from __future__ import annotations
import argparse,json,math
from collections import deque,defaultdict
from decimal import Decimal
from pathlib import Path
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)

def metrics(xs, initial=1000.0):
 a=np.asarray(xs,float); w=a[a>0]; l=a[a<0]
 pf=float(w.sum()/abs(l.sum())) if len(l) else (math.inf if len(w) else 0.0)
 eq=initial; peak=initial; mdd=0.0
 for x in a:
  eq+=x; peak=max(peak,eq); mdd=max(mdd,peak-eq)
 return {'N':int(len(a)),'WR_pct':float((a>0).mean()*100) if len(a) else 0.0,'PF':pf,'Net':float(a.sum()) if len(a) else 0.0,'EV':float(a.mean()) if len(a) else 0.0,'MaxDD_virtual':float(mdd),'MaxDD_pct_virtual':float(mdd/peak*100) if peak>0 else 0.0,'RF_virtual':float(a.sum()/mdd) if mdd>0 else None}

class Cfg(StrategyConfig,frozen=True):
 instrument_id: object; m1:BarType; m5:BarType; m15:BarType

class GBCombo(Strategy):
 def __init__(self,c):
  super().__init__(c)
  self.b={1:[deque(maxlen=4000) for _ in range(4)],5:[deque(maxlen=2000) for _ in range(4)],15:[deque(maxlen=1200) for _ in range(4)]}
  self.tr=deque(maxlen=1000); self.prev=None; self.pending=[]; self.active=[]; self.pnl=defaultdict(list); self.ledger=[]
  self.last_bid=self.last_ask=None; self.day_key=None; self.daily_hi=-1e99; self.daily_lo=1e99
 def on_start(self):
  self.subscribe_quote_ticks(self.config.instrument_id)
  for x in (self.config.m1,self.config.m5,self.config.m15): self.subscribe_bars(x)
 def atr(self,n=14): return float(np.mean(list(self.tr)[-n:])) if len(self.tr)>=n else 0.0
 def swing_levels(self,tf):
  h=np.asarray(self.b[tf][1],float); l=np.asarray(self.b[tf][2],float)
  if len(h)<30:return [],[]
  buys=[];sells=[];d=12
  for i in range(max(d,len(h)-300),len(h)-d):
   if h[i]>=h[i-d:i+d+1].max():buys.append(h[i])
   if l[i]<=l[i-d:i+d+1].min():sells.append(l[i])
  cap=3 if tf==15 else 4
  return buys[-cap:],sells[-cap:]
 def on_bar(self,bar:Bar):
  s=str(bar.bar_type);tf=15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is None:return
  o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close]);[self.b[tf][i].append(v) for i,v in enumerate((o,h,l,c))]
  if tf==1:
   tr=max(h-l,abs(h-self.prev) if self.prev is not None else 0,abs(l-self.prev) if self.prev is not None else 0);self.tr.append(tr);self.prev=c
   day=int(bar.ts_event//(86400*1e9))
   if self.day_key!=day:self.day_key=day;self.daily_hi=h;self.daily_lo=l
   else:self.daily_hi=max(self.daily_hi,h);self.daily_lo=min(self.daily_lo,l)
  if tf in (5,15): self.signal(tf,c)
 def signal(self,tf,c):
  atr=self.atr()
  if atr<=0:return
  buys,sells=self.swing_levels(tf);off=.3;sl=max(4.0,1.2*atr);tp=max(9.0,2.4*atr);id='GB_M5' if tf==5 else 'GB_M15'
  existing={(x[0],x[1],round(x[2],3)) for x in self.pending}
  for p in buys:
   key=(id,1,round(p-off,3))
   if p>self.daily_hi+0.4 and p>c and key not in existing:self.pending.append((id,1,p-off,sl,tp));existing.add(key)
  for p in sells:
   key=(id,-1,round(p+off,3))
   if p<self.daily_lo-0.4 and p<c and key not in existing:self.pending.append((id,-1,p+off,sl,tp));existing.add(key)
  self.pending=self.pending[-40:]
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask
  keep=[]
  for id,side,p,sl,tp in self.pending:
   hit=(ask>=p if side>0 else bid<=p)
   if hit:self.active.append({'id':id,'side':side,'entry':ask if side>0 else bid,'sl':sl,'tp':tp})
   else:keep.append((id,side,p,sl,tp))
  self.pending=keep
  nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side
   if move<=-a['sl'] or move>=a['tp']:
    self.pnl[a['id']].append(move);self.ledger.append(move)
   else:nxt.append(a)
  self.active=nxt
 def on_stop(self):
  if self.last_bid is None:return
  for a in self.active:
   mark=self.last_bid if a['side']>0 else self.last_ask;move=(mark-a['entry'])*a['side'];self.pnl[a['id']].append(move);self.ledger.append(move)
  self.active=[]
 def summary(self):
  return {'per_logic':{k:metrics(v) for k,v in sorted(self.pnl.items())},'combined':metrics(self.ledger)}

def l1(xs):
 one=Quantity.from_int(1);o=[]
 for t in xs:
  bs=f(t.bid_size);az=f(t.ask_size);o.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) if bs<=0 or az<=0 else t)
 return o

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));sym=inst.id.value;st=GBCombo(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{sym}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{sym}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{sym}-15-MINUTE-BID-INTERNAL')));eng.add_strategy(st);eng.run();eng.end();res={'verification':'NAUTILUS_RAW_GOLDEBRAVE_M5_M15_SIMULTANEOUS','raw_ticks':len(raw),'ohlc_resample_used':False,'note':'M5 and M15 virtual legs run simultaneously from the same raw XAUUSD stream; ZigZag approximated by confirmed depth-12 local extrema.',**st.summary()};p=Path('results/goldebrave-combo')/a.experiment_id/'summary.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
