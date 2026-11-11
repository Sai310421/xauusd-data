"""Offline provider contract tests for the multi-asset data universe."""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multiasset_provider import AssetSpec, MultiAssetDatabentoProvider


class Store:
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({"ts_event": pd.to_datetime(["2026-01-01T00:00:00Z"]), "close": [1.0], "volume": [10], "instrument_id": [42]})


class Timeseries:
    def get_range(self, **kwargs: object) -> Store:
        self.kwargs = kwargs
        return Store()


class Client:
    def __init__(self) -> None:
        self.timeseries = Timeseries()


def test_uses_continuous_front_and_h1_schema() -> None:
    client = Client()
    provider = MultiAssetDatabentoProvider("test", {"GC": AssetSpec("GC", "GLBX.MDP3", "GC.v.0")}, client, lambda: datetime(2026, 1, 1, 2, tzinfo=UTC))
    frame = provider.fetch_bars("GC", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 2, tzinfo=UTC))
    assert client.timeseries.kwargs["stype_in"] == "continuous"
    assert client.timeseries.kwargs["symbols"] == "GC.v.0"
    assert client.timeseries.kwargs["schema"] == "ohlcv-1h"
    assert len(frame) == 1
