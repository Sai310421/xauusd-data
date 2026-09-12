from __future__ import annotations
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
    def q(self,identifiers=None,start=None,end=None,**kwargs):
        return self.query(QuoteTick,identifiers=identifiers,start=start,end=end,**kwargs)
    ParquetDataCatalog.query_quote_ticks=q
from research.rangehunter_m1_trendfollow_v2_1_nautilus_raw_bt import main
if __name__=='__main__':main()
