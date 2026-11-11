"""Contracts for fail-closed prediction and drift monitoring."""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURES
from src.monitoring import append_prediction_log, psi_report, psi_status
from src.safety import (
    PredictionStopped,
    validate_prediction_input,
    validate_probability_order,
)


def _manifest() -> dict[str, object]:
    return {"version":"candidate-v1", "created_at":"2026-01-01T00:00:00Z", "target":"target_05", "feature_schema":FEATURES, "feature_order":FEATURES, "model_uri":"oci://bucket/m", "calibrator_uri":"oci://bucket/c", "model_sha256":"0"*64, "calibrator_sha256":"1"*64, "train_start":"", "train_end":"", "metrics":{}, "psi_reference_uri":"oci://bucket/p", "data_hash":"2"*64, "config_version":"v1"}


def _row() -> pd.Series:
    row = pd.Series({feature: 0.0 for feature in FEATURES})
    row["best_lag_bars"] = 1
    row["leadlag_corr"] = 0.2
    row["lead_lag_available"] = 1
    row["datetime_utc"] = pd.Timestamp("2026-01-01T00:00:00Z")
    return row


def test_stale_and_probability_order_stop() -> None:
    with pytest.raises(PredictionStopped, match="STALE_DATA"):
        validate_prediction_input(_row(), _manifest(), now=datetime(2026, 1, 1, 2, tzinfo=UTC))
    with pytest.raises(PredictionStopped, match="ORDER"):
        validate_probability_order(0.5, 0.6, 0.1)


def test_prediction_input_passes_only_current_complete_schema() -> None:
    result = validate_prediction_input(_row(), _manifest(), now=datetime(2026, 1, 1, 1, tzinfo=UTC))
    assert result.missing_rate == 0


def test_psi_and_append_only_log(tmp_path: Path) -> None:
    baseline = pd.DataFrame({feature: np.arange(100) for feature in FEATURES})
    report = psi_report(baseline, baseline, pd.Timestamp("2026-01-01T00:00:00Z"), "candidate-v1")
    assert report["psi"].max() < 1e-9
    assert psi_status(0.26) == "CRITICAL"
    path = tmp_path / "prediction_log.parquet"
    record = {"datetime_utc":"2026-01-01T00:00:00Z", "prob_05":0.6, "prob_10":0.4, "prob_20":0.2, "regime":"RISK_OFF", "model_version":"candidate-v1", "top_driver_1":"flow_vx", "top_driver_2":"risk_score", "top_driver_3":"jpy_div"}
    append_prediction_log(path, record)
    with pytest.raises(ValueError, match="DUPLICATE"):
        append_prediction_log(path, record)
