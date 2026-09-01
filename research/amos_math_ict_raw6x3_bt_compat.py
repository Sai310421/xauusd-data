from __future__ import annotations

from nautilus_trader.persistence.catalog import ParquetDataCatalog


def _install_quote_tick_compat() -> str:
    if hasattr(ParquetDataCatalog, "query_quote_ticks"):
        return "query_quote_ticks"
    if hasattr(ParquetDataCatalog, "quotes"):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quotes(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return "quotes"
    raise RuntimeError("No Raw QuoteTick reader found on ParquetDataCatalog")

CATALOG_QUOTE_API = _install_quote_tick_compat()
print(f"CATALOG_QUOTE_API={CATALOG_QUOTE_API}")

from research.amos_math_ict_raw6x3_bt import main

if __name__ == "__main__":
    main()
