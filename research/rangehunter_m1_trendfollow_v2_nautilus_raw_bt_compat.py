from __future__ import annotations
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

def _install():
    if hasattr(ParquetDataCatalog,'query_quote_ticks'):return 'query_quote_ticks'
    if hasattr(ParquetDataCatalog,'query'):
        def q(self,identifiers=None,start=None,end=None,**kwargs):return self.query(QuoteTick,identifiers=identifiers,start=start,end=end,**kwargs)
        ParquetDataCatalog.query_quote_ticks=q;return 'query(QuoteTick)'
    if hasattr(ParquetDataCatalog,'quotes'):
        def q(self,identifiers=None,start=None,end=None,**kwargs):return self.quotes(instrument_ids=identifiers,start=start,end=end,**kwargs)
        ParquetDataCatalog.query_quote_ticks=q;return 'quotes'
    raise RuntimeError('No Raw QuoteTick reader')
print('CATALOG_QUOTE_API='+_install())
from research.rangehunter_m1_trendfollow_v2_nautilus_raw_bt import main
if __name__=='__main__':main()
