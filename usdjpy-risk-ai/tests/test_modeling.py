"""Tests for chronological model fitting and calibration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES
from src.modeling import FoldDefinition, fit_target, predict_probability, split_fold, walk_forward


def model_frame(rows: int = 900) -> pd.DataFrame:
    """Create deterministic, balanced synthetic feature rows for model tests."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(rows, len(FEATURES)))
    frame = pd.DataFrame(data, columns=FEATURES)
    frame["best_lag_bars"] = rng.integers(0, 2, size=rows).astype("int8")
    score = frame["flow_vx"] - frame["equity_stress"] + rng.normal(0, 0.4, rows)
    target = (score > np.median(score)).astype(int)
    frame["target_05"] = target
    frame["target_10"] = target
    frame["target_20"] = target
    frame["datetime_utc"] = pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC")
    return frame


def test_fit_target_uses_later_validation_slice() -> None:
    frame = model_frame()
    fitted = fit_target(
        frame,
        "target_05",
        n_estimators=20,
        isotonic_min_rows=10000,
        isotonic_min_positive=10000,
    )
    probability = predict_probability(fitted, frame.iloc[-20:])
    assert fitted.calibrator.method == "sigmoid"
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_fold_purges_label_horizon_before_test() -> None:
    frame = model_frame()
    fold = FoldDefinition("x", "2023-01-01", "2023-01-20", "2023-01-21", "2023-01-31")
    train, test = split_fold(frame, fold, embargo_bars=24)
    assert train["datetime_utc"].max() < test["datetime_utc"].min()


def test_walkforward_uses_explicit_fold_only() -> None:
    frame = model_frame()
    fold = FoldDefinition("x", "2023-01-01", "2023-01-20", "2023-01-21", "2023-02-05")
    report = walk_forward(
        frame,
        targets=("target_05",),
        folds=(fold,),
        min_positive_samples=10,
        n_estimators=20,
    )
    assert report.iloc[0]["fold"] == "x"
    assert report.iloc[0]["status"] == "PASS"
