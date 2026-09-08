"""Central compatibility layer for NautilusTrader catalog reads.

Repository policy:
- Raw Bid/Ask QuoteTick data only for strict BT lanes.
- Never fall back to OHLC-resampled data.
- Keep Nautilus version-specific API differences outside strategy logic.

The current pinned runtime is NautilusTrader 1.230.0. In this runtime
``ParquetDataCatalog.query_quote_ticks`` may be absent while the generic
``ParquetDataCatalog.query(data_cls=QuoteTick, ...)`` API is available.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import nautilus_trader
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog


@dataclass(frozen=True)
class CatalogCompatInfo:
    nautilus_version: str
    catalog_class: str
    has_query_quote_ticks: bool
    has_query: bool
    has_instruments: bool
    selected_api: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version() -> str:
    return str(getattr(nautilus_trader, "__version__", "unknown"))


def catalog_compat_info(catalog: Any) -> CatalogCompatInfo:
    has_qqt = callable(getattr(catalog, "query_quote_ticks", None))
    has_query = callable(getattr(catalog, "query", None))
    has_instruments = callable(getattr(catalog, "instruments", None))
    if has_qqt:
        selected = "query_quote_ticks"
    elif has_query:
        selected = "query(data_cls=QuoteTick)"
    else:
        selected = "UNSUPPORTED"
    return CatalogCompatInfo(
        nautilus_version=_version(),
        catalog_class=type(catalog).__name__,
        has_query_quote_ticks=has_qqt,
        has_query=has_query,
        has_instruments=has_instruments,
        selected_api=selected,
    )


def list_instruments_compat(catalog: Any) -> list[Any]:
    fn = getattr(catalog, "instruments", None)
    if not callable(fn):
        info = catalog_compat_info(catalog)
        raise RuntimeError(f"Catalog instruments API unavailable: {info.to_dict()}")
    items = list(fn())
    if not items:
        raise RuntimeError("Catalog contains no instruments")
    return items


def select_instrument_compat(catalog: Any, symbol: str) -> Any:
    target = symbol.replace("/", "").upper()
    for instrument in list_instruments_compat(catalog):
        value = instrument.id.symbol.value.replace("/", "").upper()
        if value == target:
            return instrument
    available = [x.id.symbol.value for x in list_instruments_compat(catalog)]
    raise RuntimeError(f"Instrument {symbol} missing; available={available}")


def query_quote_ticks_compat(
    catalog: Any,
    *,
    identifiers: Iterable[str] | None = None,
    start: Any = None,
    end: Any = None,
) -> list[QuoteTick]:
    """Read QuoteTicks across Nautilus catalog API variants.

    Preferred order:
    1. Native query_quote_ticks when present.
    2. Nautilus generic catalog.query(data_cls=QuoteTick, ...).

    No bar/OHLC fallback is permitted.
    """
    ids = None if identifiers is None else list(identifiers)
    errors: list[str] = []

    native = getattr(catalog, "query_quote_ticks", None)
    if callable(native):
        try:
            out = native(identifiers=ids, start=start, end=end)
            return list(out or [])
        except Exception as exc:  # preserve evidence, then try known generic API
            errors.append(f"query_quote_ticks:{type(exc).__name__}:{exc}")

    generic = getattr(catalog, "query", None)
    if callable(generic):
        try:
            kwargs: dict[str, Any] = {"data_cls": QuoteTick}
            if ids is not None:
                kwargs["identifiers"] = ids
            if start is not None:
                kwargs["start"] = start
            if end is not None:
                kwargs["end"] = end
            out = generic(**kwargs)
            return list(out or [])
        except Exception as exc:
            errors.append(f"query(data_cls=QuoteTick):{type(exc).__name__}:{exc}")

    info = catalog_compat_info(catalog)
    raise RuntimeError(
        "Unable to read raw QuoteTicks without fallback; "
        f"compat={info.to_dict()} errors={errors}"
    )


def install_query_quote_ticks_alias() -> None:
    """Legacy bridge for older repo runners which still call query_quote_ticks."""
    if hasattr(ParquetDataCatalog, "query_quote_ticks"):
        return

    def _query_quote_ticks(self, identifiers=None, start=None, end=None):
        return query_quote_ticks_compat(
            self,
            identifiers=identifiers,
            start=start,
            end=end,
        )

    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks


def validate_raw_quote_ticks(
    catalog: Any,
    *,
    symbol: str,
    min_ticks: int = 1,
) -> dict[str, Any]:
    instrument = select_instrument_compat(catalog, symbol)
    ticks = query_quote_ticks_compat(catalog, identifiers=[instrument.id.value])
    if len(ticks) < min_ticks:
        raise RuntimeError(
            f"Raw QuoteTick validation failed for {symbol}: {len(ticks)} < {min_ticks}"
        )
    first = ticks[0]
    last = ticks[-1]
    return {
        "status": "RAW_QUOTETICK_COMPAT_OK",
        "symbol": symbol,
        "instrument_id": instrument.id.value,
        "ticks": len(ticks),
        "first_ts_init": int(first.ts_init),
        "last_ts_init": int(last.ts_init),
        "ohlc_fallback_used": False,
        "compat": catalog_compat_info(catalog).to_dict(),
    }
