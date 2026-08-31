from __future__ import annotations
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.portfolio.portfolio import Portfolio

def _install_quote_tick_compat() -> str:
    if hasattr(ParquetDataCatalog, 'query_quote_ticks'):
        return 'query_quote_ticks'
    if hasattr(ParquetDataCatalog, 'quote_ticks'):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quote_ticks(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return 'quote_ticks'
    if hasattr(ParquetDataCatalog, 'quotes'):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quotes(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return 'quotes'
    if hasattr(ParquetDataCatalog, 'query'):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return 'query(QuoteTick)'
    raise RuntimeError('No Raw QuoteTick reader found on ParquetDataCatalog')

def _install_portfolio_flat_compat() -> str:
    if hasattr(Portfolio, 'is_net_flat'):
        return 'is_net_flat'
    if hasattr(Portfolio, 'is_net_long') and hasattr(Portfolio, 'is_net_short'):
        def is_net_flat(self, instrument_id):
            return (not self.is_net_long(instrument_id)) and (not self.is_net_short(instrument_id))
        Portfolio.is_net_flat = is_net_flat
        return 'derived_from_long_short'
    raise RuntimeError('No compatible net-position state API found on Portfolio')

CATALOG_QUOTE_API = _install_quote_tick_compat()
PORTFOLIO_FLAT_API = _install_portfolio_flat_compat()
print(f'CATALOG_QUOTE_API={CATALOG_QUOTE_API}')
print(f'PORTFOLIO_FLAT_API={PORTFOLIO_FLAT_API}')
from research.amos_allweather_raw_bidask_bt import main
if __name__ == '__main__':
    main()
