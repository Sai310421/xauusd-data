"""Databento H1 collector for the fixed multi-asset model universe."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .data_provider import (
    DataQualityError,
    ProviderCredentialsError,
    ProviderResponseError,
)

MULTIASSET_COLUMNS = ("asset", "source_bar_start_utc", "close", "volume", "provider", "dataset", "continuous_symbol", "instrument_id", "roll_event")
MULTIASSET_ASSETS = ("RTY", "GC", "DXY", "US10Y")


@dataclass(frozen=True)
class AssetSpec:
    """One provider-owned continuous future definition."""

    asset: str
    dataset: str
    continuous_symbol: str
    schema: str = "ohlcv-1h"


@dataclass
class MultiAssetDatabentoProvider:
    """Fetch H1 data without self-managed rolls or adjusted prices."""

    api_key: str
    specs: Mapping[str, AssetSpec]
    client: Any
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    @classmethod
    def from_environment(cls, config: Mapping[str, Any]) -> MultiAssetDatabentoProvider:
        """Construct from the canonical config and the Databento secret."""
        if not os.environ.get("DATABENTO_API_KEY"):
            raise ProviderCredentialsError("missing required production secret: DATABENTO_API_KEY")
        try:
            import databento as db
        except ImportError as exc:
            raise ProviderResponseError("databento package is required in production") from exc
        assets = config["multiasset"]["assets"]
        specs = {name.upper(): AssetSpec(name.upper(), details["dataset"], details["continuous_symbol"], details["schema"]) for name, details in assets.items()}
        if set(specs) != set(MULTIASSET_ASSETS):
            raise DataQualityError("multiasset universe is incomplete")
        return cls(os.environ["DATABENTO_API_KEY"], specs, db.Historical(os.environ["DATABENTO_API_KEY"]))

    def fetch_bars(self, asset: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch complete H1 bars using provider continuous front symbology."""
        if asset not in self.specs:
            raise DataQualityError(f"unsupported multiasset: {asset}")
        spec = self.specs[asset]
        response = self.client.timeseries.get_range(dataset=spec.dataset, schema=spec.schema, symbols=spec.continuous_symbol, stype_in="continuous", stype_out="instrument_id", start=start, end=end)
        source = response.to_df().reset_index()
        timestamp = next((key for key in ("ts_event", "timestamp", "datetime") if key in source), None)
        if timestamp is None or not {"close", "volume"}.issubset(source.columns):
            raise ProviderResponseError("multiasset OHLCV response is incomplete")
        contracts = source["instrument_id"].astype(str) if "instrument_id" in source else pd.Series(spec.continuous_symbol, index=source.index)
        result = pd.DataFrame({"asset": asset, "source_bar_start_utc": pd.to_datetime(source[timestamp], utc=True), "close": source["close"], "volume": source["volume"], "provider": "DATABENTO", "dataset": spec.dataset, "continuous_symbol": spec.continuous_symbol, "instrument_id": contracts})
        previous = result["instrument_id"].shift()
        result["roll_event"] = previous.notna() & result["instrument_id"].ne(previous)
        cutoff = self.now() - timedelta(hours=1)
        return result.loc[result["source_bar_start_utc"] <= cutoff, list(MULTIASSET_COLUMNS)].copy()


def write_monthly_multiasset(frame: pd.DataFrame, directory: Path) -> list[Path]:
    """Write feature-source partitions without mixing them into USDJPY raw data."""
    if set(frame.columns) != set(MULTIASSET_COLUMNS):
        raise DataQualityError("multiasset raw schema mismatch")
    if set(frame["asset"]) != set(MULTIASSET_ASSETS):
        raise DataQualityError("multiasset raw universe mismatch")
    result = frame.copy()
    result["source_bar_start_utc"] = pd.to_datetime(result["source_bar_start_utc"], utc=True)
    if result.duplicated(["asset", "source_bar_start_utc"]).any():
        raise DataQualityError("duplicate multiasset H1 bar")
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for period, group in result.groupby(result["source_bar_start_utc"].dt.to_period("M")):
        path = directory / f"multiasset_{period.year:04d}_{period.month:02d}.parquet"
        group.sort_values(["asset", "source_bar_start_utc"]).to_parquet(path, index=False)
        paths.append(path)
    return paths
