from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def metrics(p,initial=1000.0):
 a=np.asarray(p,float); w=a[a>0]; l=a[a<0]; eq=peak=initial; dd=0.0
 for x in a: eq+=x; peak=max(peak,eq); dd=max(dd,peak-eq)
 pf=float(w.sum()/abs(l.sum())) if len(l) else (math.inf if len(w) else 0.0)
 return {'N':len(a),'WR_pct':float((a>0).mean()*100) if len(a) else 0.0,'PF':pf,'EV':float(a.mean()) if len(a) else 0.0,'Net':float(a.sum()),'Return_pct':float(a.sum()/initial*100),'MaxDD_pct':float(dd/initial*100),'RF':float(a.sum()/dd) if dd else None}

class Cfg(StrategyConfig,frozen=True):
 instrument_id:object
 mode:str='base'

class TrendPursuit(Strategy):
 def __init__(self,c):
  super().__init__(c); self.mid=deque(maxlen=256); self.pos=None; self.pnl=[]; self.last_bid=self.last_ask=None
 def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)
 def score(self):
  if len(self.mid)<65:return None
  x=np.asarray(self.mid,float); d=abs(x[-1]-x[-33]); path=np.abs(np.diff(x[-33:])).sum(); er=d/path if path else 0
  slope=(x[-1]-x[-33])/32; vol=np.std(np.diff(x[-65:])) or 1e-9
  return er, slope/vol
 def on_quote_tick(self,t:QuoteTick):
  bid,ask=f(t.bid_price),f(t.ask_price); self.last_bid,self.last_ask=bid,ask; m=(bid+ask)/2; self.mid.append(m)
  q=self.score()
  if q is None:return
  er,z=q
  if self.pos is None:
   # FE_BOT_002: independently test trend direction + pullback continuation before any G75 overlay.
   if er>=0.35 and abs(z)>=0.20:
    side=1 if z>0 else -1
    recent=np.asarray(self.mid,float); impulse=(recent[-1]-recent[-17])*side; pull=(recent[-1]-recent[-5])*side
    if impulse>0 and pull<=0:
     entry=ask if side>0 else bid; sigma=np.std(np.diff(recent[-65:]))
     self.pos={'side':side,'entry':entry,'stop':entry-side*max(6*sigma,0.5),'take':entry+side*max(12*sigma,1.0)}
  else:
   p=self.pos; mark=bid if p['side']>0 else ask
   hit=(mark<=p['stop'] or mark>=p['take']) if p['side']>0 else (mark>=p['stop'] or mark<=p['take'])
   if hit:self.pnl.append((mark-p['entry'])*p['side']);self.pos=None
 def on_stop(self):
  if self.pos:
   mark=self.last_bid if self.pos['side']>0 else self.last_ask
   self.pnl.append((mark-self.pos['entry'])*self.pos['side']);self.pos=None

def executable(xs):
 one=Quantity.from_int(1); out=[]
 for t in xs:
  out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if f(t.bid_size)<=0 else t.bid_size,ask_size=one if f(t.ask_size)<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init))
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args()
 cat=ParquetDataCatalog(a.catalog)
 raw=(cat.query_quote_ticks(identifiers=[next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD').id.value]) if hasattr(cat,'query_quote_ticks') else [])
 if not raw: raise SystemExit('no raw XAUUSD QuoteTicks')
 inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
 eng.add_instrument(inst);eng.add_data(executable(raw));st=TrendPursuit(Cfg(instrument_id=inst.id));eng.add_strategy(st);eng.run();eng.end()
 out={'verification':'FE_BOT_002_TRENDPURSUIT_RAW_BASE','raw_ticks':len(raw),'ohlc_resample_used':False,'initial_usd':1000,'leverage':2000,**metrics(st.pnl)}
 p=Path('results/fe-bot-002')/a.experiment_id/'base.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
