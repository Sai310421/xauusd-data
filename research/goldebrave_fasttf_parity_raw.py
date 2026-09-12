from __future__ import annotations
import argparse,json,math
from collections import deque,defaultdict
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
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
def met(a,initial=1000.0):
 a=np.asarray(a,float);w=a[a>0];l=a[a<0];pf=float(w.sum()/abs(l.sum())) if len(l) else (math.inf if len(w) else 0.0);eq=peak=initial;mdd=0
 for x in a:eq+=x;peak=max(peak,eq);mdd=max(mdd,peak-eq)
 return {'N':int(len(a)),'WR_pct':float((a>0).mean()*100) if len(a) else 0.0,'PF':pf,'Net':float(a.sum()) if len(a) else 0.0,'EV':float(a.mean()) if len(a) else 0.0,'MaxDD_pct_virtual':float(mdd/peak*100) if peak else 0.0,'RF_virtual':float(a.sum()/mdd) if mdd else None}
class Cfg(StrategyConfig,frozen=True):
 instrument_id:object;m1:BarType;m5:BarType;m15:BarType;h1:BarType;mode:str
class GB(Strategy):
 def __init__(self,c):
  super().__init__(c);self.b={1:[deque(maxlen=5000) for _ in range(4)],5:[deque(maxlen=2500) for _ in range(4)],15:[deque(maxlen=1500) for _ in range(4)],60:[deque(maxlen=1200) for _ in range(4)]};self.pending=[];self.active=[];self.pnl=defaultdict(list);self.ledger=[];self.last_bid=self.last_ask=None;self.entries_day=0;self.day=None;self.placed=[];self.last_scan={5:None,15:None,60:None};self.last_boost=None
 def on_start(self):
  self.subscribe_quote_ticks(self.config.instrument_id)
  for x in (self.config.m1,self.config.m5,self.config.m15,self.config.h1):self.subscribe_bars(x)
 def atr(self,n):
  h=list(self.b[60][1]);l=list(self.b[60][2]);c=list(self.b[60][3]);
  if len(c)<n+1:return None
  tr=[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(len(c)-n,len(c))];return float(np.mean(tr))
 def adx(self,n=14):
  h=np.asarray(self.b[60][1],float);l=np.asarray(self.b[60][2],float);c=np.asarray(self.b[60][3],float)
  if len(c)<2*n+2:return None
  ph=h[-(2*n+1):-1];ch=h[-2*n:];pl=l[-(2*n+1):-1];cl=l[-2*n:];pc=c[-(2*n+1):-1]
  up=ch-ph;dn=pl-cl;plus=np.where((up>dn)&(up>0),up,0);minus=np.where((dn>up)&(dn>0),dn,0);tr=np.maximum(ch-cl,np.maximum(abs(ch-pc),abs(cl-pc)))
  out=[]
  for j in range(n,2*n):
   ts=tr[j-n:j].sum();
   if ts<=0:continue
   p=100*plus[j-n:j].sum()/ts;m=100*minus[j-n:j].sum()/ts;den=p+m
   if den>0:out.append(100*abs(p-m)/den)
  return float(np.mean(out[-n:])) if out else None
 def vol_reg(self):
  a14=self.atr(14);a480=self.atr(480)
  vol=min(max(a14/a480,.6),2.5) if a14 and a480 else 1.0;adx=self.adx(14);reg=1 if adx is not None and adx>=25 else -1 if adx is not None and adx<=18 else 0
  return vol,reg,a14
 def extrema(self,tf):
  h=np.asarray(self.b[tf][1],float);l=np.asarray(self.b[tf][2],float);d=12
  if len(h)<2*d+3:return [],[]
  hi=[];lo=[]
  for i in range(max(d,len(h)-600),len(h)-d):
   if h[i]>=h[i-d:i+d+1].max():hi.append(h[i])
   if l[i]<=l[i-d:i+d+1].min():lo.append(l[i])
  return hi,lo
 def hour(self,ts):
  # Dukascopy timestamps are UTC; original EA summer server setting is GMT+3.
  return (pd.Timestamp(ts,unit='ns',tz='UTC').hour+3)%24
 def dayrange(self):
  h=list(self.b[15][1]);l=list(self.b[15][2]);
  if not h:return None
  n=max(1,min(len(h),int(self.cur_hour*4)))
  return max(h[-n:]),min(l[-n:])
 def scan(self,tf,layer,cap):
  dr=self.dayrange();vol,reg,a=self.vol_reg()
  if dr is None or a is None:return
  dhi,dlo=dr;mind=.4*vol;off=.3*vol;hib=dhi+mind;lob=dlo-mind
  sl=min(max(a*1.2,4.0),12.0);tp=min(max(a*2.4,9.0),30.0)*(1.25 if reg==1 else .8 if reg==-1 else 1.0)
  hi,lo=self.extrema(tf);nb=ns=0
  for z in reversed(hi):
   if z>hib and nb<cap:
    p=z-off;nb+=1;hib=z
    if all(abs(p-q[2])>=mind for q in self.pending):self.pending.append((layer,1,p,sl,tp,vol))
  for z in reversed(lo):
   if z<lob and ns<cap:
    p=z+off;ns+=1;lob=z
    if all(abs(p-q[2])>=mind for q in self.pending):self.pending.append((layer,-1,p,sl,tp,vol))
  self.pending=self.pending[-30:]
 def boost(self):
  if self.cur_hour<9 or self.entries_day>=3:return
  dr=self.dayrange();vol,reg,a=self.vol_reg()
  if dr is None or a is None:return
  dhi,dlo=dr;off=.2*vol*(2 if reg==-1 else 1);sl=min(max(a*1.2,4),12);tp=min(max(a*2.4,9),30)*(1.25 if reg==1 else .8 if reg==-1 else 1)
  for side,p in ((1,dhi+off),(-1,dlo-off)):self.pending.append(('C',side,p,sl,tp,vol))
  self.pending=self.pending[-30:]
 def on_bar(self,bar:Bar):
  s=str(bar.bar_type);tf=60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is None:return
  o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close]);[self.b[tf][i].append(v) for i,v in enumerate((o,h,l,c))];self.cur_hour=self.hour(int(bar.ts_event));day=int((int(bar.ts_event)+3*3600*10**9)//(86400*10**9))
  if self.day!=day:self.day=day;self.entries_day=0;self.placed=[]
  allowed=self.cur_hour in (11,15,16,17,18)
  if not allowed:self.pending=[]
  if not allowed:return
  if tf==60:self.scan(60,'A',7 if self.vol_reg()[1]==1 else 5)
  if self.config.mode in ('m5','both') and tf==5:self.scan(5,'B5',3)
  if self.config.mode in ('m15','both') and tf==15:self.scan(15,'B15',3);self.boost()
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for layer,side,p,sl,tp,vol in self.pending:
   hit=ask>=p if side>0 else bid<=p
   if hit:self.active.append({'layer':layer,'side':side,'entry':ask if side>0 else bid,'sl':sl,'tp':tp,'vol':vol,'stop':(ask if side>0 else bid)-side*sl,'take':(ask if side>0 else bid)+side*tp});self.entries_day+=1
   else:keep.append((layer,side,p,sl,tp,vol))
  self.pending=keep
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   if move>=1.2*vol:a['stop']=max(a['stop'],a['entry']+.2*vol) if side>0 else min(a['stop'],a['entry']-.2*vol)
   if move>=2.0*vol and None not in (m1h,m1l,h1h,h1l):
    gate=(m1h>=h1h-.3*vol) or (m1l<=h1l+.3*vol)
    if gate:
     ns=(m1h-3*vol) if side>0 else (m1l+3*vol);a['stop']=max(a['stop'],ns) if side>0 else min(a['stop'],ns)
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:self.pnl[a['layer']].append(move);self.ledger.append(move)
   else:nxt.append(a)
  self.active=nxt
 def on_stop(self):
  for a in self.active:
   mark=self.last_bid if a['side']>0 else self.last_ask;mv=(mark-a['entry'])*a['side'];self.pnl[a['layer']].append(mv);self.ledger.append(mv)
 def summary(self):return {'mode':self.config.mode,'per_layer':{k:met(v) for k,v in sorted(self.pnl.items())},'overall':met(self.ledger)}
def l1(xs):
 one=Quantity.from_int(1);out=[]
 for t in xs:
  bs=f(t.bid_size);az=f(t.ask_size);out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) if bs<=0 or az<=0 else t)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--mode',choices=['m5','m15','both'],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value;st=GB(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode=a.mode));eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_V4_FASTTF_PARITY_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'preserved':['PivotTF H1 Layer A','BarrierTF M15','TradeHours 11,15,16,17,18 server time approximated as UTC+3 summer','ATR H1 SL1.2/TP2.4 with 4/9 floors 12/30 caps','ATR14/ATR480 vol adaptation','ADX14 regime TP multipliers','Layer C minimum 3 entries after hour 9','BE 1.2/lock .2 scaled','M1/H1 gated trail'], 'limitation':'MT5 Examples/ZigZag is approximated by confirmed depth-12 extrema; final parity requires MT5 tester.',**st.summary()};p=Path('results/goldebrave-fasttf')/a.experiment_id/f'{a.mode}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
