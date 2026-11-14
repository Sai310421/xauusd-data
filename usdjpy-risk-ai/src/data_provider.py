"""Production and fixture market-data adapters for the fixed H1 contract.

Production adapters never substitute fixture data.  They use Databento
continuous front contracts for futures and OANDA midpoint candles for USDJPY.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .utils import as_utc, frame_checksum

REQUIRED_RAW_COLUMNS = {
    "symbol", "source_bar_start_utc", "close", "volume", "provider",
    "dataset", "actual_contract", "instrument_id", "roll_event",
}
REQUIRED_SYMBOLS = ("6J", "VX", "ES", "NQ", "ZN", "USDJPY")
FUTURES_SYMBOLS = ("6J", "VX", "ES", "NQ", "ZN")
class DataQualityError(ValueError):
    """Raised when data violates the fixed H1 market-data contract."""


class ProviderCredentialsError(RuntimeError):
    """Raised when production credentials are absent; never fall back to fixtures."""


class ProviderResponseError(RuntimeError):
    """Raised when a provider response cannot satisfy the H1 contract."""


class MarketDataProvider(Protocol):
    """Read completed bars for a symbol and UTC time interval."""

    def fetch_bars(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Fetch only completed source bars."""


def require_environment(names: tuple[str, ...], environ: Mapping[str, str] | None = None) -> None:
    """Fail fast with missing variable names only, never secret values."""
    values = os.environ if environ is None else environ
    missing = [name for name in names if not values.get(name)]
    if missing:
        raise ProviderCredentialsError(
            "missing required production secrets: " + ", ".join(missing)
        )


def _utc(value: datetime) -> datetime:
    """Return one timezone-aware UTC datetime."""
    if value.tzinfo is None:
        raise DataQualityError("start and end must be timezone-aware UTC timestamps")
    return value.astimezone(UTC)


def _completed_before_now(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Discard a source bar that has not completely elapsed."""
    cutoff = _utc(now or datetime.now(UTC)) - timedelta(hours=1)
    stamps = pd.to_datetime(frame["source_bar_start_utc"], utc=True)
    return frame.loc[stamps <= cutoff].copy()


@dataclass(frozen=True)
class FixtureProvider:
    """Read deterministic fixture raw bars from CSV or Parquet."""

    path: Path

    def fetch_bars(self, symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        """Return a symbol slice after fixed-schema validation."""
        if interval != "1h":
            raise DataQualityError(f"unsupported fixture interval: {interval}")
        frame = pd.read_parquet(self.path) if self.path.suffix == ".parquet" else pd.read_csv(self.path)
        normalized = normalize_raw(frame)
        mask = (
            (normalized["symbol"] == symbol)
            & (normalized["datetime_utc"] >= pd.Timestamp(_utc(start)))
            & (normalized["datetime_utc"] <= pd.Timestamp(_utc(end)))
        )
        return normalized.loc[mask].copy()


@dataclass
class DatabentoProvider:
    """Databento continuous-contract H1 reader; no self-managed futures rolls."""

    api_key: str
    symbols: Mapping[str, str]
    datasets: Mapping[str, str]
    schemas: Mapping[str, str]
    client: Any | None = None
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    @classmethod
    def from_environment(cls, config: Mapping[str, Any]) -> DatabentoProvider:
        """Construct the client only after the Databento secret is present."""
        require_environment(("DATABENTO_API_KEY",))
        try:
            import databento as db
        except ImportError as exc:
            raise ProviderResponseError("databento package is required in production") from exc
        section = config["databento"]
        return cls(
            api_key=os.environ["DATABENTO_API_KEY"],
            symbols={"6J": section["continuous_contract"]["symbols"]["six_j"], "ES": section["continuous_contract"]["symbols"]["es"], "NQ": section["continuous_contract"]["symbols"]["nq"], "ZN": section["continuous_contract"]["symbols"]["zn"], "VX": section["continuous_contract"]["symbols"]["vx"]},
            datasets={"6J": section["datasets"]["futures"], "ES": section["datasets"]["futures"], "NQ": section["datasets"]["futures"], "ZN": section["datasets"]["futures"], "VX": section["datasets"]["volatility"]},
            schemas={"6J": section["schema"]["futures"], "ES": section["schema"]["futures"], "NQ": section["schema"]["futures"], "ZN": section["schema"]["futures"], "VX": section["schema"]["volatility"]},
            client=db.Historical(os.environ["DATABENTO_API_KEY"]),
        )

    def fetch_bars(self, symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        """Read provider-defined front contract using continuous symbology only."""
        if symbol not in FUTURES_SYMBOLS:
            raise DataQualityError(f"Databento symbol unsupported: {symbol}")
        if interval != "1h":
            raise DataQualityError(f"Databento interval must be 1h, got {interval}")
        if self.client is None:
            raise ProviderResponseError("Databento client is not initialized")
        response = self.client.timeseries.get_range(
            dataset=self.datasets[symbol], schema=self.schemas[symbol], symbols=self.symbols[symbol],
            stype_in="continuous", stype_out="instrument_id", start=_utc(start), end=_utc(end),
        )
        source = response.to_df().reset_index()
        timestamp = next((name for name in ("ts_event", "timestamp", "datetime") if name in source.columns), None)
        if timestamp is None or "close" not in source.columns or "volume" not in source.columns:
            raise ProviderResponseError("Databento OHLCV response missing timestamp, close, or volume")
        identity = (
            source["instrument_id"].astype(str)
            if "instrument_id" in source
            else source.get("symbol", pd.Series(self.symbols[symbol], index=source.index)).astype(str)
        )
        frame = pd.DataFrame({
            "symbol": symbol,
            "source_bar_start_utc": pd.to_datetime(source[timestamp], utc=True),
            "close": source["close"], "volume": source["volume"], "provider": "DATABENTO",
            "dataset": self.datasets[symbol], "actual_contract": identity,
            "instrument_id": identity,
        })
        previous_contract = frame["actual_contract"].shift()
        frame["roll_event"] = previous_contract.notna() & frame["actual_contract"].ne(previous_contract)
        return _completed_before_now(frame, self.now())


@dataclass
class OandaProvider:
    """OANDA v20 midpoint H1 reader for the canonical USDJPY label price."""

    token: str
    base_url: str = "https://api-fxpractice.oanda.com"
    instrument: str = "USD_JPY"
    session: Any | None = None
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    max_candles_per_request: int = 5000

    @classmethod
    def from_environment(cls, config: Mapping[str, Any]) -> OandaProvider:
        """Build a production reader after verifying the token is present."""
        require_environment(("OANDA_API_TOKEN",))
        section = config["oanda"]
        environment = os.environ.get("OANDA_ENVIRONMENT", section.get("environment", "practice"))
        if environment not in {"practice", "live"}:
            raise DataQualityError("OANDA_ENVIRONMENT must be practice or live")
        base_url = "https://api-fxtrade.oanda.com" if environment == "live" else "https://api-fxpractice.oanda.com"
        return cls(token=os.environ["OANDA_API_TOKEN"], base_url=base_url, instrument=section["instrument"])

    def _session(self) -> Any:
        if self.session is not None:
            return self.session
        try:
            import requests
        except ImportError as exc:
            raise ProviderResponseError("requests package is required in production") from exc
        return requests.Session()

    def fetch_bars(self, symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        """Request midpoint candles, page safely, and reject incomplete candles."""
        if symbol != "USDJPY" or interval != "1h":
            raise DataQualityError("OANDA provider only supports USDJPY at 1h")
        start_utc, end_utc = _utc(start), _utc(end)
        if end_utc <= start_utc:
            raise DataQualityError("end must be after start")
        records: list[dict[str, Any]] = []
        cursor = start_utc
        session = self._session()
        while cursor < end_utc:
            page_end = min(cursor + timedelta(hours=self.max_candles_per_request), end_utc)
            response = session.get(
                f"{self.base_url}/v3/instruments/{self.instrument}/candles",
                headers={"Authorization": f"Bearer {self.token}"},
                params={"from": cursor.isoformat().replace("+00:00", "Z"), "to": page_end.isoformat().replace("+00:00", "Z"), "granularity": "H1", "price": "M", "includeFirst": "true"},
                timeout=30,
            )
            if getattr(response, "status_code", 200) >= 400:
                raise ProviderResponseError(f"OANDA candle request failed: HTTP {response.status_code}")
            payload = response.json()
            for candle in payload.get("candles", []):
                if not candle.get("complete", False) or "mid" not in candle:
                    continue
                mid = candle["mid"]
                records.append({"symbol": "USDJPY", "source_bar_start_utc": candle["time"], "close": mid["c"], "volume": candle["volume"], "provider": "OANDA", "dataset": "oanda-v20", "actual_contract": self.instrument, "instrument_id": self.instrument, "roll_event": False})
            cursor = page_end
        return _completed_before_now(pd.DataFrame(records, columns=sorted(REQUIRED_RAW_COLUMNS)), self.now())


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
    unknown = set(result["symbol"]).difference(REQUIRED_SYMBOLS)
    if unknown:
        raise DataQualityError(f"unexpected symbols: {sorted(unknown)}")
    result["checksum"] = frame_checksum(result[["symbol", "datetime_utc", "close", "volume", "roll_event"]])
    return result


def write_monthly_raw(frame: pd.DataFrame, directory: Path) -> list[Path]:
    """Write monthly raw Parquet partitions."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for period, group in frame.groupby(frame["datetime_utc"].dt.to_period("M")):
        path = directory / f"raw_{period.year:04d}_{period.month:02d}.parquet"
        combined = pd.concat(
            [pd.read_parquet(path), group] if path.exists() else [group],
            ignore_index=True,
        ).sort_values(["datetime_utc", "symbol"])
        keys = ["datetime_utc", "symbol"]
        compared = ["close", "volume", "provider", "dataset", "actual_contract", "instrument_id"]
        duplicates = combined.loc[combined.duplicated(keys, keep=False)]
        if not duplicates.empty and (
            duplicates.groupby(keys)[compared].nunique(dropna=False) > 1
        ).any().any():
            raise DataQualityError("HISTORICAL_BAR_CONFLICT")
        combined = combined.drop_duplicates(keys, keep="last").reset_index(drop=True)
        combined["checksum"] = frame_checksum(
            combined[["symbol", "datetime_utc", "close", "volume", "roll_event"]]
        )
        combined.to_parquet(path, index=False)
        paths.append(path)
    return paths