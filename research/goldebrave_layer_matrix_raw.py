from __future__ import annotations
import argparse,json
from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from goldebrave_fasttf_parity_raw import GB,Cfg,f,l1

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None):
  return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

MODES={
 'c':set('C'),
 'ac':set('AC'),
 'ab5c':{'A','B5','C'},
 'ab15c':{'A','B15','C'},
 'ab5b15c':{'A','B5','B15','C'},
}

class GBMatrix(GB):
 def __init__(self,c):
  super().__init__(c); self.enabled=MODES[c.mode]
 def on_bar(self,bar:Bar):
  s=str(bar.bar_type)
  tf=60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is None:return
  o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close])
  [self.b[tf][i].append(v) for i,v in enumerate((o,h,l,c))]
  self.cur_hour=self.hour(int(bar.ts_event))
  day=int((int(bar.ts_event)+3*3600*10**9)//(86400*10**9))
  if self.day!=day:
   self.day=day;self.entries_day=0;self.placed=[]
  allowed=self.cur_hour in (11,15,16,17,18)
  if not allowed:self.pending=[]
  if not allowed:return
  if tf==60 and 'A' in self.enabled:
   self.scan(60,'A',7 if self.vol_reg()[1]==1 else 5)
  if tf==5 and 'B5' in self.enabled:
   self.scan(5,'B5',3)
  if tf==15:
   if 'B15' in self.enabled:self.scan(15,'B15',3)
   if 'C' in self.enabled:self.boost()

def run(catalog,mode,experiment_id):
 cat=ParquetDataCatalog(catalog)
 inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
 raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
 eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=GBMatrix(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode=mode))
 eng.add_strategy(st);eng.run();eng.end()
 res={'verification':'GOLDEBRAVE_LAYER_DECOMPOSITION_RAW','mode':mode,'enabled_layers':sorted(MODES[mode]),'raw_ticks':len(raw),'ohlc_resample_used':False,'source_parity':'Same GoldeBrave v4 approximation used by FastTF parity run; only layer enablement changes.','limitation':'MT5 Examples/ZigZag remains approximated by confirmed depth-12 extrema.',**st.summary()}
 p=Path('results/goldebrave-layer-matrix')/experiment_id/f'{mode}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--mode',choices=list(MODES),required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.mode,a.experiment_id)
if __name__=='__main__':main()
