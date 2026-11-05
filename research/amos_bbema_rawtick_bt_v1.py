#!/usr/bin/env python3
"""AMOS BB-RSI divergence / EMA-compression XAUUSD Raw Bid/Ask screening BT."""
from __future__ import annotations
import argparse,json,math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import numpy as np
import nautilus_trader
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,BookType,OmsType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
def num(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def rma(v,p):
 if len(v)<p:return None
 o=float(np.mean(v[:p]))
 for x in v[p:]:o=(o*(p-1)+x)/p
 return o
def ema(v,p):
 if len(v)<p:return None
 a=2/(p+1);o=float(np.mean(v[:p]))
 for x in v[p:]:o=a*x+(1-a)*o
 return o
def rsi(v,p=14):
 if len(v)<p+1:return None
 d=np.diff(np.asarray(v,float));u=[max(float(x),0) for x in d];dn=[max(float(-x),0) for x in d]
 au,ad=rma(u,p),rma(dn,p)
 if au is None or ad is None:return None
 return 100 if ad==0 else 100-100/(1+au/ad)
@dataclass
class Variant:
 name:str;use_div:bool;use_ema:bool;pending:object=None;active:object=None;signals:int=0;trades:list=None
 def __post_init__(self):self.trades=[]
class Cfg(StrategyConfig,frozen=True):
 instrument_id:object;bar_type:BarType;lot:float;contract_size:float
class S(Strategy):
 def __init__(self,c):
  super().__init__(c);self.o,self.h,self.l,self.c=(deque(maxlen=320) for _ in range(4));self.last_bid=self.last_ask=None;self.ticks=0
  self.v=[Variant("bb_rsi_divergence",True,False),Variant("ema_compression",False,True),Variant("combined_or",True,True)]
 def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id);self.subscribe_bars(self.config.bar_type)
 def plow(self,a,i,s=3):return i>=s and i+s<len(a) and all(a[i]<a[i-k] and a[i]<=a[i+k] for k in range(1,s+1))
 def phigh(self,a,i,s=3):return i>=s and i+s<len(a) and all(a[i]>a[i-k] and a[i]>=a[i+k] for k in range(1,s+1))
 def prev(self,a,start,low,look):
  for i in range(start-1,max(3,start-look)-1,-1):
   if self.plow(a,i) if low else self.phigh(a,i):return i
  return None
 def atr(self,h,l,c,i,p=14):
  if i<p:return None
  return rma([max(h[j]-l[j],abs(h[j]-c[j-1]),abs(l[j]-c[j-1])) for j in range(1,i+1)],p)
 def bands(self,c,i,p=20):
  if i+1<p:return None
  w=np.asarray(c[i-p+1:i+1],float);return float(w.mean()+2*w.std(ddof=0)),float(w.mean()-2*w.std(ddof=0))
 def div(self,h,l,c,q):
  b=self.bands(c,q);rq=rsi(c[:q+1])
  if b is None or rq is None:return False,False
  up,lo=b;buy=sell=False
  if self.plow(l,q) and l[q]<=lo:
   z=self.prev(l,q,True,80);rz=rsi(c[:z+1]) if z is not None else None;buy=z is not None and rz is not None and l[q]<l[z] and rq>rz+1
  if self.phigh(h,q) and h[q]>=up:
   z=self.prev(h,q,False,80);rz=rsi(c[:z+1]) if z is not None else None;sell=z is not None and rz is not None and h[q]>h[z] and rq<rz-1
  return buy,sell
 def comp(self,o,h,l,c,n):
  if n<206:return False,False
  for i in range(n-6,n):
   a=self.atr(h,l,c,i);e=[ema(c[:i+1],p) for p in (20,50,100,200)]
   if a is None or any(x is None for x in e) or max(e)-min(e)>a*.35:return False,False
  a=self.atr(h,l,c,n);e=[ema(c[:n+1],p) for p in (20,50,100,200)]
  if a is None or any(x is None for x in e):return False,False
  rl=self.prev(l,n,True,60);ol=self.prev(l,rl,True,60) if rl is not None else None;rh=self.prev(h,n,False,60);oh=self.prev(h,rh,False,60) if rh is not None else None
  nl=rl is not None and ol is not None and l[rl]>=l[ol]-a*.2;nh=rh is not None and oh is not None and h[rh]<=h[oh]+a*.2;body=abs(c[n]-o[n])
  return nl and l[n]<=max(e) and c[n]>max(e) and c[n]>o[n] and body>=a*.35,nh and h[n]>=min(e) and c[n]<min(e) and c[n]<o[n] and body>=a*.35
 def close(self,v,mark,why):
  x=v.active;points=(mark-x["entry"])*x["side"];risk=abs(x["entry"]-x["stop"]);v.trades.append({"direction":"BUY" if x["side"]>0 else "SELL","entry":x["entry"],"exit":mark,"pnl_usd":points*self.config.contract_size*self.config.lot,"r":points/risk if risk else None,"mae_price":x["mae"],"mfe_price":x["mfe"],"exit_reason":why,"signal_bar":x["bar"]});v.active=None
 def on_quote_tick(self,t):
  self.ticks+=1;bid,ask=num(t.bid_price),num(t.ask_price);self.last_bid,self.last_ask=bid,ask
  for v in self.v:
   if v.pending is not None and v.active is None:
    side,stop,bar=v.pending;entry=ask if side>0 else bid;v.active={"side":side,"entry":entry,"stop":stop,"take":entry+side*abs(entry-stop)*2,"bar":bar,"mae":0.,"mfe":0.};v.pending=None
   if v.active is not None:
    x=v.active;mark=bid if x["side"]>0 else ask;move=(mark-x["entry"])*x["side"];x["mae"],x["mfe"]=min(x["mae"],move),max(x["mfe"],move);sl=mark<=x["stop"] if x["side"]>0 else mark>=x["stop"];tp=mark>=x["take"] if x["side"]>0 else mark<=x["take"]
    if sl or tp:self.close(v,mark,"TP" if tp else "SL")
 def on_bar(self,b):
  o,h,l,c=map(list,(self.o,self.h,self.l,self.c));o.append(num(b.open));h.append(num(b.high));l.append(num(b.low));c.append(num(b.close));n,q=len(c)-1,len(c)-4
  if n>=210:
   db,ds=self.div(h,l,c,q);eb,es=self.comp(o,h,l,c,n);a=self.atr(h,l,c,n)
   if a is not None:
    for v in self.v:
     if v.active is not None or v.pending is not None:continue
     buy=(db if v.use_div else False) or (eb if v.use_ema else False);sell=(ds if v.use_div else False) or (es if v.use_ema else False)
     if buy==sell:continue
     side=1 if buy else -1;swing=min(l[n],l[q]) if side>0 else max(h[n],h[q]);v.pending=(side,swing-a*.25 if side>0 else swing+a*.25,n);v.signals+=1
  self.o.append(o[-1]);self.h.append(h[-1]);self.l.append(l[-1]);self.c.append(c[-1])
 def on_stop(self):
  for v in self.v:
   if v.active is not None:self.close(v,self.last_bid if v.active["side"]>0 else self.last_ask,"END_OF_DATA")
def l1(xs):
 one=Quantity.from_int(1);return [QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if num(t.bid_size)<=0 else t.bid_size,ask_size=one if num(t.ask_size)<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) if num(t.bid_size)<=0 or num(t.ask_size)<=0 else t for t in xs]
def metrics(v,initial=1000):
 p=np.asarray([x["pnl_usd"] for x in v.trades],float);w,l=p[p>0],p[p<0];eq=peak=initial;dd=0
 for x in p:eq+=float(x);peak=max(peak,eq);dd=max(dd,peak-eq)
 return {"signals":v.signals,"N":int(len(p)),"WR_pct":float((p>0).mean()*100) if len(p) else 0,"PF":float(w.sum()/abs(l.sum())) if len(l) else (None if not len(w) else "INF"),"EV_usd":float(p.mean()) if len(p) else 0,"Net_usd":float(p.sum()) if len(p) else 0,"Return_pct":float(p.sum()/initial*100) if len(p) else 0,"MaxDD_pct":float(dd/peak*100) if peak else None,"avg_R":float(np.mean([x["r"] for x in v.trades])) if v.trades else 0,"TP":sum(x["exit_reason"]=="TP" for x in v.trades),"SL":sum(x["exit_reason"]=="SL" for x in v.trades)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--catalog",action="append",required=True);p.add_argument("--experiment-id",required=True);p.add_argument("--lot",type=float,default=.02);p.add_argument("--contract-size",type=float,default=100);a=p.parse_args();raw=[];mans=[];inst=None
 for root in map(Path,a.catalog):
  mans.append(json.loads((root/"catalog_manifest.json").read_text()));cat=ParquetDataCatalog(str(root));i=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD");inst=inst or i;raw.extend(cat.query(data_cls=QuoteTick,identifiers=[i.id.value]))
 raw.sort(key=lambda x:x.ts_event)
 if not raw:raise SystemExit("No XAUUSD Raw QuoteTicks")
 e=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)));e.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"));e.add_instrument(inst);e.add_data(l1(raw));s=S(Cfg(instrument_id=inst.id,bar_type=BarType.from_str(f"{inst.id.value}-5-MINUTE-BID-INTERNAL"),lot=a.lot,contract_size=a.contract_size));e.add_strategy(s);e.run();e.end()
 r={"status":"COMPLETE","verification":"NAUTILUS_RAW_BIDASK_BBEMA_V1","engine":"NautilusTrader BacktestEngine","nautilus_version":getattr(nautilus_trader,"__version__","unknown"),"raw_ticks":len(raw),"ohlc_resample_used":False,"signal_bars":"M5 Nautilus INTERNAL BID bars built online from Raw QuoteTicks","execution":"First QuoteTick after confirmed bar: BUY at ASK / SELL at BID; exits on raw Bid/Ask QuoteTicks","account":{"initial_usd":1000,"leverage":2000,"lot":a.lot,"contract_size_oz_per_lot":a.contract_size},"entry_logic":{"bb_rsi":"BB(20,2) touch + RSI(14) confirmed regular divergence; pivot side=3","ema":"EMA(20,50,100,200) <=0.35 ATR for six prior bars + no-new-high/low + breakout body >=0.35 ATR","combined":"OR of the two independent engines"},"exit_logic":"SL = signal/current structure +/- 0.25 ATR(14); TP = 2.0R; one position per variant","vwap_gate":{"included":False,"reason":"MT5 custom VWAP market-state/BigPlayer buffers are not yet ported; results are entry-core screening only"},"catalogs":[m.get("catalog_sha256") for m in mans],"results":{v.name:metrics(v) for v in s.v},"trades":{v.name:v.trades for v in s.v}}
 out=Path("results/amos-bbema")/a.experiment_id/"summary.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
