"""Causal H1 feature and label generation for Gold and US-stock models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import rolling_zscore, safe_log_return

FEATURES_GOLD = ["flow_gc", "flow_vx", "flow_zn", "flow_rty", "smallcap_stress", "breadth_div", "dxy_z", "us10y_z"]
FEATURES_STOCK = ["flow_vx", "flow_rty", "smallcap_stress", "breadth_div", "dxy_z", "us10y_z", "risk_score"]
GOLD_TARGETS = ("target_gold_up_1", "target_gold_up_2", "target_gold_up_3")
STOCK_TARGETS = ("target_spx_down_1", "target_spx_down_2", "target_spx_down_3")


class AssetFeatureError(ValueError):
    """Raised when H1 source alignment or causal quality rules are violated."""


def _flow(price_z: pd.Series, volume_z: pd.Series) -> pd.Series:
    return np.sign(price_z) * np.abs(price_z) * np.log1p(np.abs(volume_z))


def _multiasset_wide(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"RTY", "GC", "DXY", "US10Y"}
    if not required.issubset(set(raw["asset"])):
        raise AssetFeatureError("MULTIASSET_UNIVERSE_INCOMPLETE")
    frame = raw.copy()
    if "roll_event" not in raw.columns:
        raise AssetFeatureError("MULTIASSET_ROLL_METADATA_MISSING")
    frame["datetime_utc"] = pd.to_datetime(frame["source_bar_start_utc"], utc=True) + pd.Timedelta(hours=1)
    if frame.duplicated(["asset", "datetime_utc"]).any():
        raise AssetFeatureError("MULTIASSET_DUPLICATE_BAR")
    close = frame.pivot(index="datetime_utc", columns="asset", values="close")
    volume = frame.pivot(index="datetime_utc", columns="asset", values="volume")
    rolls = frame.pivot(index="datetime_utc", columns="asset", values="roll_event")
    if close[list(required)].isna().any().any() or volume[list(required)].isna().any().any():
        raise AssetFeatureError("MULTIASSET_ALIGNMENT_INCOMPLETE")
    result = pd.DataFrame(index=close.index)
    for asset in sorted(required):
        result[f"close_{asset.lower()}"] = close[asset]
        result[f"volume_{asset.lower()}"] = volume[asset]
        result[f"roll_{asset.lower()}"] = rolls[asset].astype(bool)
    return result.reset_index()


def build_asset_features(base_features: pd.DataFrame, multiasset_raw: pd.DataFrame, zscore_window: int = 480, label_horizon: int = 24) -> pd.DataFrame:
    """Join completed H1 sources strictly and compute no-lookahead model inputs."""
    required_base = {"datetime_utc", "close_es", "close_nq", "flow_vx", "flow_zn", "risk_score", "roll_es"}
    if not required_base.issubset(base_features.columns):
        raise AssetFeatureError("USDJPY_FEATURE_SOURCE_INCOMPLETE")
    base = base_features.copy()
    base["datetime_utc"] = pd.to_datetime(base["datetime_utc"], utc=True)
    frame = base.merge(_multiasset_wide(multiasset_raw), on="datetime_utc", how="left", validate="one_to_one")
    source_columns = ["close_rty", "close_gc", "close_dxy", "close_us10y", "volume_rty", "volume_gc", "volume_dxy", "volume_us10y"]
    if frame[source_columns].isna().any().any():
        raise AssetFeatureError("MULTIASSET_H1_JOIN_INCOMPLETE")
    for asset in ("rty", "gc", "dxy", "us10y"):
        frame[f"r_{asset}"] = safe_log_return(frame[f"close_{asset}"])
        frame[f"z_{asset}"] = rolling_zscore(frame[f"r_{asset}"], zscore_window)
        volume_z = rolling_zscore(np.log1p(frame[f"volume_{asset}"]), zscore_window)
        frame[f"flow_{asset}"] = _flow(frame[f"z_{asset}"], volume_z)
    frame["dxy_z"] = frame["z_dxy"]
    frame["us10y_z"] = frame["z_us10y"]
    frame["smallcap_stress"] = -frame["z_rty"]
    frame["breadth_div"] = frame["z_rty"] - 0.5 * frame["z_es"] - 0.5 * frame["z_nq"]
    frame["smallcap_warning"] = ((frame["breadth_div"] < -2) | (frame["smallcap_stress"] > 2)).astype("int8")
    frame["gold_score"] = 0.30 * frame["flow_gc"] - 0.20 * frame["dxy_z"] + 0.15 * frame["flow_zn"] + 0.15 * frame["flow_vx"] - 0.10 * frame["us10y_z"] + 0.10 * frame["smallcap_stress"]
    future_gc = np.log(frame["close_gc"].shift(-label_horizon) / frame["close_gc"])
    future_es = np.log(frame["close_es"].shift(-label_horizon) / frame["close_es"])
    future_gc_roll = frame["roll_gc"].iloc[::-1].rolling(label_horizon + 1, min_periods=1).max().iloc[::-1].astype(bool)
    future_es_roll = frame["roll_es"].iloc[::-1].rolling(label_horizon + 1, min_periods=1).max().iloc[::-1].astype(bool)
    for target, threshold in zip(GOLD_TARGETS, (0.01, 0.02, 0.03), strict=True):
        frame[target] = (future_gc > threshold).astype("float64")
        frame.loc[future_gc.isna() | future_gc_roll, target] = np.nan
    for target, threshold in zip(STOCK_TARGETS, (-0.01, -0.02, -0.03), strict=True):
        frame[target] = (future_es < threshold).astype("float64")
        frame.loc[future_es.isna() | future_es_roll, target] = np.nan
    roll_columns = [column for column in frame if column.startswith("roll_")]
    frame["roll_event"] = frame[roll_columns].any(axis=1)
    # A raw continuous-contract discontinuity is not a predictive return.
    model_features = list(dict.fromkeys(FEATURES_GOLD + FEATURES_STOCK))
    frame.loc[frame["roll_event"], model_features] = np.nan
    frame["feature_timestamp"] = frame["datetime_utc"]
    frame["max_feature_source_timestamp"] = frame["datetime_utc"]
    return frame


def prepare_asset_training_rows(frame: pd.DataFrame, feature_columns: list[str], targets: tuple[str, ...]) -> pd.DataFrame:
    """Remove warm-up and terminal rows; reject feature loss above 5 percent."""
    result = frame.dropna(subset=list(targets)).copy()
    # Initial trailing z-score warm-up is expected and is not a source failure.
    warmup_columns = result.filter(regex=r"^dxy_z$")
    result = result.loc[~warmup_columns.isna().any(axis=1)].copy()
    missing = result[feature_columns].isna().mean(axis=1)
    if (missing > 0.05).any():
        raise AssetFeatureError("REQUIRED_FEATURE_MISSING")
    return result.loc[missing == 0].copy()


def write_monthly_asset_features(frame: pd.DataFrame, directory: Path) -> list[Path]:
    """Write bounded monthly Parquet partitions for model training."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for period, group in frame.groupby(frame["datetime_utc"].dt.to_period("M")):
        path = directory / f"asset_features_{period.year:04d}_{period.month:02d}.parquet"
        group.to_parquet(path, index=False)
        paths.append(path)
    return paths
