"""Shared safe numerical and time helpers."""
from __future__ import annotations

import hashlib
from datetime import UTC

import numpy as np
import pandas as pd


def as_utc(values: pd.Series) -> pd.Series:
    """Return timestamps normalized to timezone-aware UTC."""
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    return converted.dt.tz_convert(UTC)


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while converting zero denominators and non-finite results to NaN."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.where(np.isfinite(result), np.nan)


def safe_log_return(price: pd.Series) -> pd.Series:
    """Compute log return without accepting zero, negative, or non-finite prices."""
    valid = price.where((price > 0) & np.isfinite(price))
    result = np.log(valid / valid.shift(1))
    return result.where(np.isfinite(result), np.nan)


def safe_log_return_excluding_events(
    price: pd.Series,
    events: pd.Series,
) -> pd.Series:
    """Compute log returns while excluding known mechanical contract rolls."""
    if not price.index.equals(events.index):
        raise ValueError("price and events indexes must match")
    return safe_log_return(price).mask(events.fillna(False).astype(bool))


def rolling_zscore(value: pd.Series, window: int) -> pd.Series:
    """Compute trailing rolling z-score; no future observations are used."""
    mean = value.rolling(window=window, min_periods=window).mean()
    std = value.rolling(window=window, min_periods=window).std(ddof=0)
    return safe_div(value - mean, std)


def frame_checksum(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 checksum for an ordered dataframe."""
    payload = frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%SZ").encode()
    return hashlib.sha256(payload).hexdigest()