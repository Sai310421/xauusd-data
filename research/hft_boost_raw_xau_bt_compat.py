from __future__ import annotations

from nautilus_trader.persistence.catalog import ParquetDataCatalog


def _query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
    # NautilusTrader 1.230.0 can expose typed readers differently depending on build.
    # Use the documented generic query while preserving raw QuoteTick semantics.
    return self.query(
        data_type="quotes",
        identifiers=identifiers,
        start=start,
        end=end,
        **kwargs,
    )


if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks

from research.hft_boost_raw_xau_bt import main


if __name__ == "__main__":
    main()
