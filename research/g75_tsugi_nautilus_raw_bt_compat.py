"""NautilusTrader 1.230.0 catalog API compatibility shim for G75 TSUGI BT."""
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    def _query_quote_ticks(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks

from research.g75_tsugi_nautilus_raw_bt import main

if __name__ == "__main__":
    main()
