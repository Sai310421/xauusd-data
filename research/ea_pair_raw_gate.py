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
def metrics(xs):
 a=np.asarray(xs,float); w=a[a>0]; l=a[a<0]; pf=float(w.sum()/abs(l.sum())) if len(l) else (math.inf if len(w) else 0.0)
 return {'N':len(a),'WR_pct':float((a>0).mean()*100) if len(a) else 0.0,'PF':pf,'Net':float(a.sum()) if len(a) else 0.0,'EV':float(a.mean()) if len(a) else 0.0}

class Cfg(StrategyConfig,frozen=True):
 instrument_id: object; m1:BarType; m15:BarType; h1:BarType; d1:BarType; h4:BarType; mode:str

class Gate(Strategy):
 def __init__(self,c):
  super().__init__(c); self.b={1:[deque(maxlen=3000) for _ in range(4)],15:[deque(maxlen=1000) for _ in range(4)],60:[deque(maxlen=1000) for _ in range(4)],240:[deque(maxlen=1000) for _ in range(4)],1440:[deque(maxlen=500) for _ in range(4)]}; self.tr=deque(maxlen=1000);self.prev=None;self.last_bid=self.last_ask=None;self.active=[];self.pnl=defaultdict(list);self.pending=[];self.day_key=None;self.daily_hi=-1e99;self.daily_lo=1e99
 def on_start(self):
  self.subscribe_quote_ticks(self.config.instrument_id)
  for x in (self.config.m1,self.config.m15,self.config.h1,self.config.h4,self.config.d1): self.subscribe_bars(x)
 def atr(self,n=14): return float(np.mean(list(self.tr)[-n:])) if len(self.tr)>=n else 0.0
 def ema(self,tf,n):
  a=list(self.b[tf][3]);
  if len(a)<n:return None
  alpha=2/(n+1);v=a[-n]
  for z in a[-n+1:]:v=alpha*z+(1-alpha)*v
  return v
 def swing_levels(self,tf):
  h=np.asarray(self.b[tf][1],float);l=np.asarray(self.b[tf][2],float)
  if len(h)<30:return [],[]
  buys=[];sells=[];d=12
  for i in range(max(d,len(h)-200),len(h)-d):
   if h[i]>=h[i-d:i+d+1].max():buys.append(h[i])
   if l[i]<=l[i-d:i+d+1].min():sells.append(l[i])
  return buys[-5:],sells[-5:]
 def on_bar(self,bar:Bar):
  s=str(bar.bar_type);tf=1440 if '-1-DAY-' in s else 240 if '-4-HOUR-' in s else 60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is None:return
  o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close]);[self.b[tf][i].append(v) for i,v in enumerate((o,h,l,c))]
  if tf==1:
   tr=max(h-l,abs(h-self.prev) if self.prev is not None else 0,abs(l-self.prev) if self.prev is not None else 0);self.tr.append(tr);self.prev=c
   day=int(bar.ts_event//(86400*1e9));
   if self.day_key!=day:self.day_key=day;self.daily_hi=h;self.daily_lo=l
   else:self.daily_hi=max(self.daily_hi,h);self.daily_lo=min(self.daily_lo,l)
  if self.config.mode=='gold' and tf in (15,60): self.gold_signal(tf,c)
  if self.config.mode=='neko' and tf==1:self.neko_signal(int(bar.ts_event),c)
 def gold_signal(self,tf,c):
  atr=self.atr();
  if atr<=0:return
  buys,sells=self.swing_levels(tf);off=.3;sl=max(4.0,1.2*atr);tp=max(9.0,2.4*atr)
  for p in buys:
   if p>self.daily_hi+0.4 and p>c:self.pending.append(('GB_A' if tf==60 else 'GB_B',1,p-off,sl,tp))
  for p in sells:
   if p<self.daily_lo-0.4 and p<c:self.pending.append(('GB_A' if tf==60 else 'GB_B',-1,p+off,sl,tp))
  self.pending=self.pending[-20:]
 def neko_signal(self,ts,c):
  # 30d Raw gate: only calendar flow legs are valid; D1 MA100/200 pullback legs need >200d history.
  import pandas as pd
  t=pd.Timestamp(ts,unit='ns',tz='UTC').tz_convert('Asia/Tokyo'); wd=t.weekday(); day=t.day
  if wd>=5:return
  def flowday(d): return d in (5,10,15,20,25) or d>=28
  if t.hour==9 and t.minute==55 and flowday(day): self.active.append({'id':'Flow_Short','side':-1,'entry':c,'slpct':1.0,'exit_h':14,'exit_m':25,'date':t.date()})
  if t.hour==23 and t.minute==5:
   nxt=t+pd.offsets.BDay(1)
   if nxt.day in (10,15,20,25):self.active.append({'id':'Flow_Long','side':1,'entry':c,'slpct':1.5,'exit_h':9,'exit_m':50,'date':nxt.date()})
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask
  # stop-entry simulation on raw bid/ask
  keep=[]
  for id,side,p,sl,tp in self.pending:
   hit=(ask>=p if side>0 else bid<=p)
   if hit:self.active.append({'id':id,'side':side,'entry':ask if side>0 else bid,'sl':sl,'tp':tp})
   else:keep.append((id,side,p,sl,tp))
  self.pending=keep
  import pandas as pd
  now=pd.Timestamp(int(t.ts_event),unit='ns',tz='UTC').tz_convert('Asia/Tokyo')
  nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;entry=a['entry'];close=False
   if 'sl' in a:
    move=(mark-entry)*side
    if move<=-a['sl'] or move>=a['tp']:close=True
   else:
    move=(mark-entry)*side/entry*100
    if move<=-a['slpct']:close=True
    if a['id']=='Flow_Short' and now.date()==a['date'] and (now.hour>14 or (now.hour==14 and now.minute>=25)):close=True
    if a['id']=='Flow_Long' and now.date()==a['date'] and (now.hour>9 or (now.hour==9 and now.minute>=50)):close=True
   if close:self.pnl[a['id']].append((mark-entry)*side)
   else:nxt.append(a)
  self.active=nxt
 def summary(self):
  per={k:metrics(v) for k,v in self.pnl.items()};allp=[x for v in self.pnl.values() for x in v];return {'mode':self.config.mode,'per_logic':per,'overall':metrics(allp)}

def l1(xs):
 one=Quantity.from_int(1);o=[]
 for t in xs:
  bs=f(t.bid_size);az=f(t.ask_size);o.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) if bs<=0 or az<=0 else t)
 return o

def run(cat,symbol,mode):
 inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')==symbol);raw=cat.query_quote_ticks(identifiers=[inst.id.value]);eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));sym=inst.id.value;st=Gate(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{sym}-1-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{sym}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{sym}-1-HOUR-BID-INTERNAL'),h4=BarType.from_str(f'{sym}-4-HOUR-BID-INTERNAL'),d1=BarType.from_str(f'{sym}-1-DAY-BID-INTERNAL'),mode=mode));eng.add_strategy(st);eng.run();eng.end();return {'symbol':symbol,'raw_ticks':len(raw),**st.summary()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();cat=ParquetDataCatalog(a.catalog);res={'verification':'RAW_BIDASK_INTERNAL_BARS_PARITY_GATE','ohlc_resample_used':False,'limitations':['GoldeBrave ZigZag is approximated with confirmed depth-12 local extrema; exact MT5 ZigZag parity requires MT5 tester.','Neko 30d Raw cache is insufficient for SMA100/SMA200 daily pullback strategies, so this gate measures Flow_Long/Flow_Short only.'], 'GoldeBrave':run(cat,'XAUUSD','gold'),'Neko':run(cat,'USDJPY','neko')};p=Path('results/ea-pair')/a.experiment_id/'summary.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
