"""Causal Gold/Stock feature and model contract tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asset_features import (
    FEATURES_GOLD,
    FEATURES_STOCK,
    GOLD_TARGETS,
    build_asset_features,
    prepare_asset_training_rows,
)
from src.asset_modeling import fit_asset_target, predict_asset_probability


def _sources(rows: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01 01:00:00", periods=rows, freq="h", tz="UTC")
    base = pd.DataFrame({"datetime_utc": timestamps, "close_es": 5000 * np.exp(np.cumsum(rng.normal(0, 0.002, rows))), "close_nq": 17000 * np.exp(np.cumsum(rng.normal(0, 0.002, rows))), "flow_vx": rng.normal(size=rows), "flow_zn": rng.normal(size=rows), "risk_score": rng.normal(size=rows), "z_es": rng.normal(size=rows), "z_nq": rng.normal(size=rows), "roll_es": False})
    records: list[dict[str, object]] = []
    for asset, start in (("RTY", 2000), ("GC", 2000), ("DXY", 100), ("US10Y", 4)):
        price = start * np.exp(np.cumsum(rng.normal(0, 0.004, rows)))
        for timestamp, close in zip(timestamps - pd.Timedelta(hours=1), price, strict=True):
            records.append({"asset": asset, "source_bar_start_utc": timestamp, "close": close, "volume": int(rng.integers(100, 1000)), "roll_event": False})
    return base, pd.DataFrame(records)


def test_asset_features_and_labels_are_causal() -> None:
    base, raw = _sources()
    frame = build_asset_features(base, raw, zscore_window=30, label_horizon=24)
    prepared = prepare_asset_training_rows(frame, FEATURES_GOLD, ("target_gold_up_1", "target_gold_up_2", "target_gold_up_3"))
    assert set(FEATURES_GOLD).issubset(frame.columns)
    assert set(FEATURES_STOCK).issubset(frame.columns)
    assert (frame["max_feature_source_timestamp"] <= frame["datetime_utc"]).all()
    assert not prepared.empty


def test_roll_gap_is_excluded_from_features_and_target_horizon() -> None:
    base, raw = _sources()
    gc_index = raw.index[raw["asset"] == "GC"][4]
    raw.loc[gc_index, "roll_event"] = True
    frame = build_asset_features(base, raw, zscore_window=30, label_horizon=24)
    event = raw.loc[(raw["asset"] == "GC") & raw["roll_event"], "source_bar_start_utc"].iloc[0] + pd.Timedelta(hours=1)
    assert frame.loc[frame["datetime_utc"] == event, FEATURES_GOLD].isna().all().all()
    affected = frame.loc[(frame["datetime_utc"] >= event - pd.Timedelta(hours=24)) & (frame["datetime_utc"] <= event), GOLD_TARGETS]
    assert affected.isna().all().all()


def test_asset_model_uses_its_own_feature_schema() -> None:
    base, raw = _sources()
    frame = build_asset_features(base, raw, zscore_window=30, label_horizon=24)
    frame = prepare_asset_training_rows(frame, FEATURES_STOCK, ("target_spx_down_1", "target_spx_down_2", "target_spx_down_3"))
    fitted = fit_asset_target(frame, "target_spx_down_1", FEATURES_STOCK, n_estimators=20)
    probability = predict_asset_probability(fitted, frame.iloc[-5:])
    assert len(probability) == 5
    assert np.all((probability >= 0) & (probability <= 1))
