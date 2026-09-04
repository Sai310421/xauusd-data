from __future__ import annotations

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def _install_quote_tick_compat() -> str:
    # NautilusTrader catalog read API differs across releases/builds.
    # Keep the strategy runner pinned to Raw QuoteTick semantics without
    # falling back to OHLC or resampled files.
    if hasattr(ParquetDataCatalog, "query_quote_ticks"):
        return "query_quote_ticks"
    if hasattr(ParquetDataCatalog, "query"):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            # NautilusTrader v1.230.0 generic catalog query expects the concrete
            # data class, not a string discriminator. Passing QuoteTick keeps the
            # read path on the immutable raw Bid/Ask stream.
            return self.query(
                QuoteTick,
                identifiers=identifiers,
                start=start,
                end=end,
                **kwargs,
            )
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return "query(QuoteTick)"
    if hasattr(ParquetDataCatalog, "quotes"):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quotes(
                instrument_ids=identifiers,
                start=start,
                end=end,
                **kwargs,
            )
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return "quotes"
    raise RuntimeError("No Raw QuoteTick reader found on ParquetDataCatalog")


CATALOG_QUOTE_API = _install_quote_tick_compat()
print(f"CATALOG_QUOTE_API={CATALOG_QUOTE_API}")

from research.minimumspike_raw6x3_bt import main


if __name__ == "__main__":
    main()
