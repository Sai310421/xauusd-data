"""Causal H1 feature engineering and label generation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_provider import REQUIRED_SYMBOLS
from .utils import (
    rolling_zscore,
    safe_log_return,
    safe_log_return_excluding_events,
)

FEATURES = [
    "flow_6j_aligned",
    "flow_vx",
    "flow_zn",
    "equity_stress",
    "jpy_div",
    "risk_score",
    "best_lag_bars",
    "leadlag_corr",
]
FEATURE_DTYPES = {
    "flow_6j_aligned": "float64",
    "flow_vx": "float64",
    "flow_zn": "float64",
    "equity_stress": "float64",
    "jpy_div": "float64",
    "risk_score": "float64",
    "best_lag_bars": "int8",
    "leadlag_corr": "float64",
}


class FeatureQualityError(ValueError):
    """Raised when causal feature generation cannot meet quality rules."""


def _wide(raw: pd.DataFrame) -> pd.DataFrame:
    required = set(REQUIRED_SYMBOLS)
    if not required.issubset(set(raw["symbol"])):
        raise FeatureQualityError("one or more required symbols are absent")
    closes = raw.pivot(index="datetime_utc", columns="symbol", values="close")
    volumes = raw.pivot(index="datetime_utc", columns="symbol", values="volume")
    rolls = raw.pivot(index="datetime_utc", columns="symbol", values="roll_event")
    if closes[list(REQUIRED_SYMBOLS)].isna().any().any():
        raise FeatureQualityError("bar alignment is incomplete; forward fill is forbidden")
    result = pd.DataFrame(index=closes.index)
    for symbol in REQUIRED_SYMBOLS:
        key = symbol.lower()
        result[f"close_{key}"] = closes[symbol]
        result[f"volume_{key}"] = volumes[symbol]
        result[f"roll_{key}"] = rolls[symbol].astype(bool)
    return result.reset_index()


def _flow(z_price: pd.Series, z_volume: pd.Series) -> pd.Series:
    return np.sign(z_price) * np.abs(z_price) * np.log1p(np.abs(z_volume))


def _lead_lag(
    frame: pd.DataFrame,
    lookback: int,
    minimum: int,
    roll_tolerance_bars: int = 1,
) -> pd.DataFrame:
    if not 0 <= roll_tolerance_bars < minimum:
        raise FeatureQualityError("invalid lead-lag roll tolerance")
    effective_minimum = minimum - roll_tolerance_bars
    best: list[int] = []
    corr: list[float] = []
    available: list[int] = []
    r6 = frame["r_6j"]
    rfx = frame["r_fx"]
    for index in range(len(frame)):
        if index < lookback:
            best.append(0)
            corr.append(0.0)
            available.append(0)
            continue
        x = r6.iloc[index - lookback:index]
        y = rfx.iloc[index - lookback:index]
        candidates: list[tuple[int, float]] = []
        for lag in (0, 1):
            paired = pd.concat([x.shift(lag), y], axis=1).dropna()
            if (
                len(paired) < effective_minimum
                or paired.iloc[:, 0].std() == 0
                or paired.iloc[:, 1].std() == 0
            ):
                continue
            value = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
            if np.isfinite(value):
                candidates.append((lag, value))
        if not candidates:
            best.append(0)
            corr.append(0.0)
            available.append(0)
        else:
            lag, value = max(candidates, key=lambda item: abs(item[1]))
            best.append(lag)
            corr.append(value)
            available.append(1)
    result = pd.DataFrame(
        {"best_lag_bars": best, "leadlag_corr": corr, "lead_lag_available": available},
        index=frame.index,
    )
    result["best_lag_bars"] = result["best_lag_bars"].astype("int8")
    return result


def build_features(
    raw: pd.DataFrame,
    zscore_window: int = 480,
    label_horizon: int = 24,
    leadlag_lookback: int = 240,
    leadlag_minimum: int = 240,
    roll_tolerance_bars: int = 1,
) -> pd.DataFrame:
    """Build H1 features using only data available at each bar end."""
    frame = _wide(raw).sort_values("datetime_utc").reset_index(drop=True)
    names = ("6j", "vx", "es", "nq", "zn")
    for name in names:
        frame[f"r_{name}"] = safe_log_return_excluding_events(
            frame[f"close_{name}"], frame[f"roll_{name}"]
        )
        frame[f"z_{name}"] = rolling_zscore(
            frame[f"r_{name}"],
            zscore_window,
            min_periods=zscore_window - 1,
        )
        volume_z = rolling_zscore(np.log1p(frame[f"volume_{name}"]), zscore_window)
        frame[f"flow_{name}"] = _flow(frame[f"z_{name}"], volume_z)
    frame["r_fx"] = -safe_log_return(frame["close_usdjpy"])
    frame["z_fx"] = rolling_zscore(frame["r_fx"], zscore_window)
    frame = pd.concat(
        [
            frame,
            _lead_lag(
                frame,
                leadlag_lookback,
                leadlag_minimum,
                roll_tolerance_bars,
            ),
        ],
        axis=1,
    )
    for source, aligned in (
        ("r_6j", "r_6j_aligned"),
        ("z_6j", "z_6j_aligned"),
        ("flow_6j", "flow_6j_aligned"),
    ):
        frame[aligned] = [
            frame[source].iloc[index - lag] if index >= lag else 0.0
            for index, lag in enumerate(frame["best_lag_bars"])
        ]
    unavailable = frame["lead_lag_available"] == 0
    aligned_columns = ["r_6j_aligned", "z_6j_aligned", "flow_6j_aligned"]
    frame.loc[unavailable, aligned_columns] = 0.0
    frame["jpy_div"] = frame["z_6j_aligned"] - frame["z_fx"]
    frame["equity_stress"] = -0.5 * frame["z_es"] - 0.5 * frame["z_nq"]
    frame["vol_stress"] = frame["z_vx"] + 0.5 * frame["flow_vx"]
    frame["safe_haven"] = 0.6 * frame["flow_6j_aligned"] + 0.4 * frame["flow_zn"]
    frame["risk_score"] = (
        0.30 * frame["safe_haven"]
        + 0.25 * frame["vol_stress"]
        + 0.20 * frame["equity_stress"]
        + 0.15 * frame["jpy_div"]
        + 0.10 * frame["z_fx"]
    )
    roll_columns = [column for column in frame if column.startswith("roll_")]
    rolls = frame[roll_columns].any(axis=1)
    frame["roll_event"] = rolls
    frame["future_return"] = np.log(
        frame["close_usdjpy"].shift(-label_horizon) / frame["close_usdjpy"]
    )
    for threshold, target in (
        (-0.005, "target_05"),
        (-0.010, "target_10"),
        (-0.020, "target_20"),
    ):
        frame[target] = (frame["future_return"] < threshold).astype("float64")
        frame.loc[frame["future_return"].isna(), target] = np.nan
    frame["feature_timestamp"] = frame["datetime_utc"]
    frame["max_feature_source_timestamp"] = frame["datetime_utc"]
    invalidated = [name for name in FEATURES if name != "best_lag_bars"]
    frame.loc[rolls, invalidated] = np.nan
    for name, dtype in FEATURE_DTYPES.items():
        frame[name] = frame[name].astype(dtype)
    return frame


def validate_feature_integrity(frame: pd.DataFrame) -> None:
    """Reject any feature whose recorded source timestamp is in the future."""
    source = pd.to_datetime(frame["max_feature_source_timestamp"], utc=True)
    prediction = pd.to_datetime(frame["datetime_utc"], utc=True)
    if (source > prediction).any():
        raise FeatureQualityError("FEATURE_INTEGRITY_FAILURE")


def prepare_training_rows(
    frame: pd.DataFrame,
    max_missing_rate: float = 0.05,
) -> pd.DataFrame:
    """Remove warmup/terminal rows, then reject material feature gaps."""
    eligible = frame.dropna(subset=["future_return"]).copy()
    # Initial rolling windows are expected warmup, not a data-quality event.
    eligible = eligible.loc[eligible["z_fx"].notna()].copy()
    missing_rate = eligible[FEATURES].isna().mean(axis=1)
    if (missing_rate > max_missing_rate).any():
        raise FeatureQualityError("REQUIRED_FEATURE_MISSING")
    result = eligible.loc[missing_rate == 0].copy()
    validate_feature_integrity(result)
    return result


def write_monthly_features(frame: pd.DataFrame, directory: Path) -> list[Path]:
    """Write monthly feature-store Parquet partitions."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for period, group in frame.groupby(frame["datetime_utc"].dt.to_period("M")):
        path = directory / f"features_{period.year:04d}_{period.month:02d}.parquet"
        group.to_parquet(path, index=False)
        paths.append(path)
    return paths


def main() -> None:
    """Generate features from a raw file or raw partition directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.is_dir():
        paths = sorted(args.input.glob("raw_*.parquet"))
        raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    else:
        raw = pd.read_parquet(args.input)
    features = build_features(raw)
    training = prepare_training_rows(features)
    paths = write_monthly_features(training, args.output)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()