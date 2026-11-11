"""Fail-closed inference gates for USDJPY risk probabilities."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .features import FEATURE_DTYPES, FEATURES
from .storage import ObjectIntegrityError, validate_manifest


class PredictionStopped(RuntimeError):
    """Raised when any production safety rule prevents inference."""


@dataclass(frozen=True)
class PredictionGateResult:
    """A successful gate result recorded with the prediction."""

    as_of: pd.Timestamp
    missing_rate: float


def _stop(reason: str) -> None:
    raise PredictionStopped(reason)


def validate_prediction_input(
    row: pd.Series,
    manifest: Mapping[str, Any],
    now: datetime | None = None,
) -> PredictionGateResult:
    """Enforce freshness, schema, lead-lag, and missing-data checks."""
    try:
        validate_manifest(manifest)
    except ObjectIntegrityError as error:
        _stop(str(error))
    if list(manifest["feature_order"]) != FEATURES or list(manifest["feature_schema"]) != FEATURES:
        _stop("FEATURE_SCHEMA_MISMATCH")
    missing = int(row.reindex(FEATURES).isna().sum())
    missing_rate = missing / len(FEATURES)
    if missing_rate > 0.05:
        _stop("MISSING_DATA")
    for feature, dtype in FEATURE_DTYPES.items():
        if feature not in row:
            _stop("FEATURE_SCHEMA_MISMATCH")
        if dtype.startswith("int") and not float(row[feature]).is_integer():
            _stop("FEATURE_SCHEMA_MISMATCH")
    if int(row.get("lead_lag_available", 0)) != 1 or not np.isfinite(row.get("leadlag_corr", np.nan)):
        _stop("LEADLAG_FAILURE")
    as_of = pd.Timestamp(row["datetime_utc"], tz="UTC") if pd.Timestamp(row["datetime_utc"]).tzinfo is None else pd.Timestamp(row["datetime_utc"]).tz_convert("UTC")
    current = pd.Timestamp(now or datetime.now(UTC))
    if current - as_of > pd.Timedelta(minutes=90):
        _stop("STALE_DATA")
    return PredictionGateResult(as_of=as_of, missing_rate=missing_rate)


def validate_probability_order(prob_05: float, prob_10: float, prob_20: float) -> None:
    """Reject invalid bounds and non-nested crash probabilities."""
    values = (prob_05, prob_10, prob_20)
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        _stop("PROBABILITY_ORDER_ERROR")
    if not prob_20 <= prob_10 <= prob_05:
        _stop("PROBABILITY_ORDER_ERROR")


def regime(prob_05: float) -> str:
    """Classify one validated USDJPY crash probability."""
    if prob_05 > 0.70:
        return "CRISIS"
    if prob_05 > 0.55:
        return "RISK_OFF"
    if prob_05 < 0.30:
        return "RISK_ON"
    return "NEUTRAL"
