from __future__ import annotations
import argparse,json,math,sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
sys.path.insert(0,str(Path(__file__).resolve().parent))
from multiedge_next4_dynamic_atr_gate import S,Cfg,EDGES

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class SFixed(S):
 def on_bar(self,b:Bar):
  s=str(b.bar_type);tf=60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is None:return
  vals=list(map(self.f,[b.open,b.high,b.low,b.close]));[self.buf[tf][i].append(vals[i]) for i in range(4)]
  if tf==1:
   o,h,l,c=vals;self.m1_i+=1;tr=max(h-l,abs(h-self.prev) if self.prev is not None else 0,abs(l-self.prev) if self.prev is not None else 0);self.tr.append(tr);self.prev=c
   if not self.active and not self.pending:self.pending=self.signal()

def l1(ticks):
 one=Quantity.from_int(1);out=[]
 for t in ticks:
  bs=float(t.bid_size.as_double()) if hasattr(t.bid_size,'as_double') else float(t.bid_size);az=float(t.ask_size.as_double()) if hasattr(t.ask_size,'as_double') else float(t.ask_size)
  out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) if bs<=0 or az<=0 else t)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--selected',choices=EDGES,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args()
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks=l1(raw)
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
 sym=inst.id.value;st=SFixed(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{sym}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{sym}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{sym}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{sym}-1-HOUR-BID-INTERNAL'),selected=a.selected));eng.add_strategy(st);eng.run();eng.end()
 res={'verification_level':'NAUTILUS_RAW_NEXT4_DYNAMIC_ATR_FIXED_H1','raw_ticks':len(raw),'ohlc_resample_used':False,'exit_model':'DYNAMIC_ATR_NO_FIXED_TP',**st.summary()};p=Path('results/multiedge')/a.experiment_id/f'{a.selected}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
