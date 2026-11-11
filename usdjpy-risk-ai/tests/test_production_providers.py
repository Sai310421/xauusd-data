"""Offline contract tests for production market-data adapters."""
from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_provider import DatabentoProvider, OandaProvider, ProviderCredentialsError, require_environment

UTC = timezone.utc


class FakeStore:
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({"ts_event": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]), "close": [1.1, 1.2], "volume": [10, 11], "instrument_id": [100, 101]})


class FakeTimeseries:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def get_range(self, **kwargs: object) -> FakeStore:
        self.kwargs = kwargs
        return FakeStore()


class FakeClient:
    def __init__(self) -> None:
        self.timeseries = FakeTimeseries()


class FakeResponse:
    status_code = 200

    def json(self) -> dict[str, object]:
        return {"candles": [{"time": "2026-01-01T00:00:00.000000000Z", "complete": True, "volume": 7, "mid": {"c": "155.1"}}, {"time": "2026-01-01T02:00:00.000000000Z", "complete": False, "volume": 8, "mid": {"c": "155.2"}}]}


class FakeSession:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def get(self, _url: str, **kwargs: object) -> FakeResponse:
        self.kwargs = kwargs
        return FakeResponse()


def test_missing_secrets_fail_without_fixture_fallback() -> None:
    with pytest.raises(ProviderCredentialsError, match="DATABENTO_API_KEY"):
        require_environment(("DATABENTO_API_KEY",), {})


def test_databento_uses_continuous_front_contract_and_discards_open_bar() -> None:
    client = FakeClient()
    provider = DatabentoProvider(api_key="test", symbols={"6J": "6J.v.0"}, datasets={"6J": "GLBX.MDP3"}, schemas={"6J": "ohlcv-1h"}, client=client, now=lambda: datetime(2026, 1, 1, 2, 30, tzinfo=UTC))
    frame = provider.fetch_bars("6J", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 3, tzinfo=UTC), "1h")
    assert client.timeseries.kwargs["stype_in"] == "continuous"
    assert client.timeseries.kwargs["symbols"] == "6J.v.0"
    assert client.timeseries.kwargs["schema"] == "ohlcv-1h"
    assert len(frame) == 2


def test_oanda_uses_midpoint_h1_and_discards_incomplete_bar() -> None:
    session = FakeSession()
    provider = OandaProvider(token="test", session=session, now=lambda: datetime(2026, 1, 1, 2, 30, tzinfo=UTC))
    frame = provider.fetch_bars("USDJPY", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 3, tzinfo=UTC), "1h")
    assert session.kwargs["params"] == {"from": "2026-01-01T00:00:00Z", "to": "2026-01-01T03:00:00Z", "granularity": "H1", "price": "M", "includeFirst": "true"}
    assert len(frame) == 1
    assert frame.iloc[0]["close"] == "155.1"
