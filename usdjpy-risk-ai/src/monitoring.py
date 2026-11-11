"""PSI drift measurement and append-only prediction audit log."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PSI_FEATURES = ["flow_6j_aligned", "flow_vx", "flow_zn", "equity_stress", "jpy_div", "best_lag_bars", "leadlag_corr", "risk_score"]


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI with fixed reference quantile bins and zero-safe proportions."""
    baseline = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    observed = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(baseline) < bins or len(observed) == 0:
        raise ValueError("PSI_INSUFFICIENT_DATA")
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        raise ValueError("PSI_ZERO_VARIANCE")
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(baseline, bins=edges)[0] / len(baseline)
    actual = np.histogram(observed, bins=edges)[0] / len(observed)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, as_of: pd.Timestamp, model_version: str) -> pd.DataFrame:
    """Create the required PSI audit table for the fixed production schema."""
    records = [{"feature": feature, "psi": population_stability_index(reference[feature], current[feature]), "as_of": as_of, "model_version": model_version} for feature in PSI_FEATURES]
    return pd.DataFrame(records)


def psi_status(psi: float) -> str:
    """Classify PSI using the fixed production thresholds."""
    return "NORMAL" if psi <= 0.10 else "WATCH" if psi <= 0.25 else "CRITICAL"


def append_prediction_log(path: Path, record: dict[str, object]) -> None:
    """Append a record by rewriting a new Parquet file; never mutate a row."""
    required = {"datetime_utc", "prob_05", "prob_10", "prob_20", "regime", "model_version", "top_driver_1", "top_driver_2", "top_driver_3"}
    if required.difference(record):
        raise ValueError("PREDICTION_LOG_SCHEMA_INVALID")
    incoming = pd.DataFrame([record])
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=incoming.columns)
    result = pd.concat([existing, incoming], ignore_index=True)
    if result.duplicated(["datetime_utc", "model_version"]).any():
        raise ValueError("DUPLICATE_ENTRY")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
