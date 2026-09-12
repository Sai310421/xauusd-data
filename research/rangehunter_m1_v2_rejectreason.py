from __future__ import annotations
import argparse,json,pandas as pd
from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType,OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick
from research.rangehunter_m1_trendfollow_v2_nautilus_raw_bt import Strat,Config
SIM=Venue('SIM')
class DiagStrat(Strat):
    def __init__(self,cfg):
        super().__init__(cfg);self.events=[]
    def _rec(self,kind,e):
        d={'kind':kind,'event':str(e)}
        for k in ('reason','client_order_id','venue_order_id','ts_event'):
            if hasattr(e,k):d[k]=str(getattr(e,k))
        self.events.append(d);print('ORDER_EVENT',json.dumps(d))
        if kind in ('REJECTED','DENIED'):
            self.entry=self.stop_ref=self.risk=self.tp=self.side=self.entry_ts=None;self.exit_pending=False;self.armed=None
    def on_order_rejected(self,e):self._rec('REJECTED',e)
    def on_order_denied(self,e):self._rec('DENIED',e)
    def on_order_filled(self,e):self._rec('FILLED',e)

def install():
    if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
        ParquetDataCatalog.query_quote_ticks=lambda self,identifiers=None,start=None,end=None,**kw:self.query(QuoteTick,identifiers=identifiers,start=start,end=end,**kw)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();install();cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query_quote_ticks(identifiers=[inst.id.value],start='2026-07-27',end='2026-07-28');eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks);bt=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL');st=DiagStrat(Config(instrument_id=inst.id,bar_type=bt));eng.add_strategy(st);eng.run();orders=eng.trader.generate_orders_report();out=Path('results/ae-bt')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);orders.to_csv(out/'orders.csv',index=False);s={'events':st.events,'orders':orders[['side','quantity','filled_qty','status']].astype(str).to_dict(orient='records') if not orders.empty else []};(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));eng.dispose()
if __name__=='__main__':main()
