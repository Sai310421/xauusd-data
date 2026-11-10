"""Provider contracts, fixture input, and raw H1 data quality validation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from .utils import as_utc, frame_checksum

REQUIRED_RAW_COLUMNS = {
    "symbol",
    "source_bar_start_utc",
    "close",
    "volume",
    "provider",
    "dataset",
    "actual_contract",
    "instrument_id",
    "roll_event",
}
REQUIRED_SYMBOLS = ("6J", "VX", "ES", "NQ", "ZN", "USDJPY")


class DataQualityError(ValueError):
    """Raised when data violates the fixed H1 market-data contract."""


class ProviderCredentialsError(RuntimeError):
    """Raised instead of silently falling back when production credentials are absent."""


class MarketDataProvider(Protocol):
    """Read completed bars for a symbol and UTC time interval."""

    def fetch_bars(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Fetch only completed source bars."""


@dataclass(frozen=True)
class FixtureProvider:
    """Read deterministic fixture raw bars from CSV or Parquet."""

    path: Path

    def fetch_bars(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Return a symbol slice after fixed-schema validation."""
        if interval != "1h":
            raise DataQualityError(f"unsupported fixture interval: {interval}")
        frame = (
            pd.read_parquet(self.path)
            if self.path.suffix == ".parquet"
            else pd.read_csv(self.path)
        )
        normalized = normalize_raw(frame)
        mask = (
            (normalized["symbol"] == symbol)
            & (normalized["datetime_utc"] >= pd.Timestamp(start, tz="UTC"))
            & (normalized["datetime_utc"] <= pd.Timestamp(end, tz="UTC"))
        )
        return normalized.loc[mask].copy()


def normalize_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source-start bars to UTC bar-end rows and validate quality."""
    missing = REQUIRED_RAW_COLUMNS.difference(frame.columns)
    if missing:
        raise DataQualityError(f"raw schema missing columns: {sorted(missing)}")
    result = frame.copy()
    result["source_bar_start_utc"] = as_utc(result["source_bar_start_utc"])
    if result["source_bar_start_utc"].isna().any():
        raise DataQualityError("invalid source_bar_start_utc")
    result["datetime_utc"] = result["source_bar_start_utc"] + pd.Timedelta(hours=1)
    for column in ("close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if (result["close"] <= 0).any() or result["close"].isna().any():
        raise DataQualityError("close must be finite and positive")
    if (result["volume"] < 0).any() or result["volume"].isna().any():
        raise DataQualityError("volume must be finite and non-negative")
    result["roll_event"] = result["roll_event"].astype(bool)
    duplicate = result.duplicated(["datetime_utc", "symbol"], keep=False)
    if duplicate.any():
        raise DataQualityError("duplicate datetime_utc + symbol")
    result = result.sort_values(["symbol", "datetime_utc"]).reset_index(drop=True)
    for symbol, group in result.groupby("symbol", sort=False):
        if not group["datetime_utc"].is_monotonic_increasing:
            raise DataQualityError(f"non-monotonic timestamps: {symbol}")
    unknown = set(result["symbol"]).difference(REQUIRED_SYMBOLS)
    if unknown:
        raise DataQualityError(f"unexpected symbols: {sorted(unknown)}")
    result["checksum"] = frame_checksum(
        result[["symbol", "datetime_utc", "close", "volume", "roll_event"]]
    )
    return result


def write_monthly_raw(frame: pd.DataFrame, directory: Path) -> list[Path]:
    """Write immutable monthly raw Parquet partitions."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for period, group in frame.groupby(frame["datetime_utc"].dt.to_period("M")):
        path = directory / f"raw_{period.year:04d}_{period.month:02d}.parquet"
        group.sort_values(["datetime_utc", "symbol"]).to_parquet(path, index=False)
        paths.append(path)
    return paths
