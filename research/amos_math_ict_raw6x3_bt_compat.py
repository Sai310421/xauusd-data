from __future__ import annotations

from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.backtest.engine import BacktestEngine


def _install_quote_tick_compat() -> str:
    # NautilusTrader catalog read APIs differ across releases/builds.
    # Preserve canonical Raw Bid/Ask QuoteTick semantics with no OHLC fallback.
    if hasattr(ParquetDataCatalog, "query_quote_ticks"):
        return "query_quote_ticks"
    if hasattr(ParquetDataCatalog, "quote_ticks"):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quote_ticks(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return "quote_ticks"
    if hasattr(ParquetDataCatalog, "quotes"):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quotes(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return "quotes"
    if hasattr(ParquetDataCatalog, "query"):
        from nautilus_trader.model.data import QuoteTick
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.query(QuoteTick, identifiers=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return "query(QuoteTick)"
    raise RuntimeError("No Raw QuoteTick reader found on ParquetDataCatalog")


def _install_positions_report_compat() -> str:
    if hasattr(BacktestEngine, "generate_positions_report"):
        return "engine.generate_positions_report"
    try:
        from nautilus_trader.analysis import ReportProvider
    except Exception as exc:
        raise RuntimeError(f"ReportProvider unavailable: {exc}") from exc

    def generate_positions_report(self):
        positions = self.cache.positions()
        snapshots = self.cache.position_snapshots()
        return ReportProvider.generate_positions_report(
            positions=positions,
            snapshots=snapshots,
        )

    BacktestEngine.generate_positions_report = generate_positions_report
    return "ReportProvider.generate_positions_report(cache.positions, snapshots)"


CATALOG_QUOTE_API = _install_quote_tick_compat()
POSITIONS_REPORT_API = _install_positions_report_compat()
print(f"CATALOG_QUOTE_API={CATALOG_QUOTE_API}")
print(f"POSITIONS_REPORT_API={POSITIONS_REPORT_API}")

from research.amos_math_ict_raw6x3_bt import main


if __name__ == "__main__":
    main()
