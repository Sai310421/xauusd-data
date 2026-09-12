from __future__ import annotations

from pathlib import Path
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def _install_quote_tick_compat() -> str:
    if hasattr(ParquetDataCatalog, 'query_quote_ticks'):
        return 'query_quote_ticks'
    if hasattr(ParquetDataCatalog, 'query'):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.query(QuoteTick, identifiers=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return 'query(QuoteTick)'
    if hasattr(ParquetDataCatalog, 'quotes'):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quotes(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return 'quotes'
    raise RuntimeError('No Raw QuoteTick reader found on ParquetDataCatalog')

CATALOG_QUOTE_API = _install_quote_tick_compat()
print(f'CATALOG_QUOTE_API={CATALOG_QUOTE_API}')

# Hotfix the first runner revision without changing its trading semantics:
# avoid shadowing Nautilus Strategy.stop() with the trade stop-price state.
runner_path = Path(__file__).with_name('rangehunter_m5_nautilus_raw_bt.py')
src = runner_path.read_text(encoding='utf-8').replace('self.stop', 'self.stop_ref')
ns = {'__name__': 'research.rangehunter_m5_nautilus_raw_bt_patched', '__file__': str(runner_path)}
exec(compile(src, str(runner_path), 'exec'), ns)
main = ns['main']

if __name__ == '__main__':
    main()
