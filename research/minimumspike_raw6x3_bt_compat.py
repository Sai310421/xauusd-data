from __future__ import annotations

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.portfolio.portfolio import Portfolio


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


def _install_portfolio_flat_compat() -> str:
    # v1.230.0 exposes is_net_long/is_net_short but not is_net_flat.
    # Preserve the strategy's intended NETTING flat-state gate exactly.
    if hasattr(Portfolio, "is_net_flat"):
        return "is_net_flat"
    if hasattr(Portfolio, "is_net_long") and hasattr(Portfolio, "is_net_short"):
        def is_net_flat(self, instrument_id):
            return not self.is_net_long(instrument_id) and not self.is_net_short(instrument_id)
        Portfolio.is_net_flat = is_net_flat
        return "not(is_net_long|is_net_short)"
    raise RuntimeError("No compatible portfolio flat-state API found")


CATALOG_QUOTE_API = _install_quote_tick_compat()
PORTFOLIO_FLAT_API = _install_portfolio_flat_compat()
print(f"CATALOG_QUOTE_API={CATALOG_QUOTE_API}")
print(f"PORTFOLIO_FLAT_API={PORTFOLIO_FLAT_API}")

from research.minimumspike_raw6x3_bt import main


if __name__ == "__main__":
    main()
