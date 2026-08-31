from __future__ import annotations

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def _query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
    # NautilusTrader 1.230.0 generic query requires the concrete data class.
    # Keep the payload strictly Raw QuoteTick; no OHLC fallback/resample.
    call_variants = [
        dict(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end, **kwargs),
        dict(data_cls=QuoteTick, instrument_ids=identifiers, start=start, end=end, **kwargs),
    ]
    last = None
    for params in call_variants:
        try:
            return self.query(**params)
        except TypeError as exc:
            last = exc
    raise last if last is not None else RuntimeError('Raw QuoteTick query failed')


if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks

from research.hft_boost_raw_xau_bt import main


if __name__ == "__main__":
    main()
