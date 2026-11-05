#!/usr/bin/env python3
"""Nautilus Raw Bid/Ask MTF trend-gated BBEMA test."""
from __future__ import annotations
import argparse,json,math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import numpy as np
import nautilus_trader
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
NS=1_000_000_000
def f(x):return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def ema(a,p):
 if len(a)<p:return None
 v=sum(a[:p])/p;k=2/(p+1)
 for x in a[p:]:v=k*x+(1-k)*v
 return v
def rma(a,p):
 if len(a)<p:return None
 v=sum(a[:p])/p
 for x in a[p:]:v=(v*(p-1)+x)/p
 return v
def rsi(c,p=14):
 if len(c)<p+1:return None
 d=np.diff(np.asarray(c,float));u=[max(float(x),0) for x in d];dn=[max(float(-x),0) for x in d];a,b=rma(u,p),rma(dn,p)
 return None if a is None else (100 if b==0 else 100-100/(1+a/b))
def atr(h,l,c,i,p=14):
 if i<p:return None
 return rma([max(h[j]-l[j],abs(h[j]-c[j-1]),abs(l[j]-c[j-1])) for j in range(1,i+1)],p)
def plow(a,i,s=3):return i>=s and i+s<len(a) and all(a[i]<a[i-k] and a[i]<=a[i+k] for k in range(1,s+1))
def phigh(a,i,s=3):return i>=s and i+s<len(a) and all(a[i]>a[i-k] and a[i]>=a[i+k] for k in range(1,s+1))
def prev(a,start,low,look):
 for i in range(start-1,max(3,start-look)-1,-1):
  if plow(a,i) if low else phigh(a,i):return i
 return None
@dataclass
class Cfg(StrategyConfig,frozen=True):
 instrument_id:object;mode:str;low_min:int;high_min:int
class Gated(Strategy):
 def __init__(self,c):
  super().__init__(c);self.lo={k:deque(maxlen=1000) for k in "ohlc"};self.hc=deque(maxlen=1000);self.lb=self.hb=None;self.lwork=self.hwork=None;self.trend=0;self.pending=self.active=None;self.lastb=self.lasta=None;self.signals=0;self.trades=[];self.ticks=0
 def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
 def close_bucket(self,w,store):
  for k in "ohlc":store[k].append(w[k])
 def update_trend(self):
  c=list(self.hc);a=ema(c,50);b=ema(c,200);old=ema(c[:-1],50)
  self.trend=1 if a is not None and b is not None and old is not None and a>b and a>old else -1 if a is not None and b is not None and old is not None and a<b and a<old else 0
 def divsig(self,o,h,l,c,n):
  q=n-3
  if q<20:return 0,None
  w=np.asarray(c[q-19:q+1]);up,lo=float(w.mean()+2*w.std()),float(w.mean()-2*w.std());rq=rsi(c[:q+1])
  if rq is None:return 0,None
  if plow(l,q) and l[q]<=lo:
   z=prev(l,q,True,80);rz=rsi(c[:z+1]) if z is not None else None
   if z is not None and rz is not None and l[q]<l[z] and rq>rz+1 and rq<=30:return 1,min(l[n],l[q])
  if phigh(h,q) and h[q]>=up:
   z=prev(h,q,False,80);rz=rsi(c[:z+1]) if z is not None else None
   if z is not None and rz is not None and h[q]>h[z] and rq<rz-1 and rq>=70:return -1,max(h[n],h[q])
  return 0,None
 def emasig(self,o,h,l,c,n):
  if n<206:return 0,None
  for i in range(n-6,n):
   a=atr(h,l,c,i);es=[ema(c[:i+1],p) for p in (20,50,100,200)]
   if a is None or any(x is None for x in es) or max(es)-min(es)>a*.35:return 0,None
  a=atr(h,l,c,n);es=[ema(c[:n+1],p) for p in (20,50,100,200)]
  if a is None or any(x is None for x in es):return 0,None
  rl, rh=prev(l,n,True,60),prev(h,n,False,60);ol=prev(l,rl,True,60) if rl is not None else None;oh=prev(h,rh,False,60) if rh is not None else None
  nl=rl is not None and ol is not None and l[rl]>=l[ol]-a*.2;nh=rh is not None and oh is not None and h[rh]<=h[oh]+a*.2;body=abs(c[n]-o[n])
  if nl and l[n]<=max(es) and c[n]>max(es) and c[n]>o[n] and body>=a*.35:return 1,min(l[n],l[n-3])
  if nh and h[n]>=min(es) and c[n]<min(es) and c[n]<o[n] and body>=a*.35:return -1,max(h[n],h[n-3])
  return 0,None
 def low_close(self,w):
  self.close_bucket(w,self.lo);o,h,l,c=[list(self.lo[k]) for k in "ohlc"];n=len(c)-1
  if self.pending or self.active or self.trend==0:return
  side,swing=(self.emasig(o,h,l,c,n) if self.config.mode=="ema" else self.divsig(o,h,l,c,n))
  if not side or side!=self.trend:return
  a=atr(h,l,c,n)
  if a is not None:self.pending=(side,swing-a*.25 if side>0 else swing+a*.25);self.signals+=1
 def finish(self,mark,why):
  x=self.active;p=(mark-x["entry"])*x["side"];risk=abs(x["entry"]-x["stop"]);self.trades.append({"pnl_usd":p*2,"r":p/risk if risk else None,"why":why});self.active=None
 def on_quote_tick(self,t):
  self.ticks+=1;bid,ask=f(t.bid_price),f(t.ask_price);self.lastb,self.lasta=bid,ask;ts=t.ts_event
  # Low closes first: a newly closing HTF bar is not used until the next low bar, preventing look-ahead.
  for minutes,kind in ((self.config.low_min,"low"),(self.config.high_min,"high")):
   bucket=ts//(minutes*60*NS);wattr="lwork" if kind=="low" else "hwork";battr="lb" if kind=="low" else "hb";work=getattr(self,wattr);old=getattr(self,battr)
   if old is None:setattr(self,battr,bucket);setattr(self,wattr,{"o":bid,"h":bid,"l":bid,"c":bid});continue
   if bucket!=old:
    if kind=="low":self.low_close(work)
    else:self.hc.append(work["c"]);self.update_trend()
    setattr(self,battr,bucket);setattr(self,wattr,{"o":bid,"h":bid,"l":bid,"c":bid})
   else:work["h"]=max(work["h"],bid);work["l"]=min(work["l"],bid);work["c"]=bid
  if self.pending and not self.active:
   s,stop=self.pending;entry=ask if s>0 else bid;self.active={"side":s,"entry":entry,"stop":stop,"take":entry+s*abs(entry-stop)*2,"mae":0,"mfe":0};self.pending=None
  if self.active:
   x=self.active;mark=bid if x["side"]>0 else ask;move=(mark-x["entry"])*x["side"];x["mae"],x["mfe"]=min(x["mae"],move),max(x["mfe"],move)
   sl=mark<=x["stop"] if x["side"]>0 else mark>=x["stop"];tp=mark>=x["take"] if x["side"]>0 else mark<=x["take"]
   if sl or tp:self.finish(mark,"TP" if tp else "SL")
 def on_stop(self):
  if self.active:self.finish(self.lastb if self.active["side"]>0 else self.lasta,"END")
 def result(self):
  p=np.asarray([x["pnl_usd"] for x in self.trades],float);w,l=p[p>0],p[p<0];eq=peak=1000;dd=0
  for x in p:eq+=x;peak=max(peak,eq);dd=max(dd,peak-eq)
  return {"signals":self.signals,"N":len(p),"WR_pct":float((p>0).mean()*100) if len(p) else 0,"PF":float(w.sum()/abs(l.sum())) if len(l) else None,"EV_usd":float(p.mean()) if len(p) else 0,"Net_usd":float(p.sum()),"Return_pct":float(p.sum()/10),"MaxDD_pct":float(dd/peak*100) if peak else None,"avg_R":float(np.mean([x["r"] for x in self.trades])) if self.trades else 0,"TP":sum(x["why"]=="TP" for x in self.trades),"SL":sum(x["why"]=="SL" for x in self.trades)}
def l1(q):
 one=Quantity.from_int(1);out=[]
 for t in q:out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if f(t.bid_size)<=0 else t.bid_size,ask_size=one if f(t.ask_size)<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) if f(t.bid_size)<=0 or f(t.ask_size)<=0 else t)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--catalog",action="append",required=True);p.add_argument("--experiment-id",required=True);a=p.parse_args();raw=[];inst=None
 for root in a.catalog:
  cat=ParquetDataCatalog(root);i=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD");inst=inst or i;raw+=cat.query(data_cls=QuoteTick,identifiers=[i.id.value])
 raw.sort(key=lambda x:x.ts_event);e=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)));e.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"));e.add_instrument(inst);e.add_data(l1(raw));ema_s=Gated(Cfg(instrument_id=inst.id,mode="ema",low_min=1,high_min=5));bb_s=Gated(Cfg(instrument_id=inst.id,mode="bb",low_min=5,high_min=15));e.add_strategy(ema_s);e.add_strategy(bb_s);e.run();e.end()
 r={"status":"COMPLETE","engine":"NautilusTrader BacktestEngine","raw_ticks":len(raw),"ohlc_resample_used":False,"execution":"Raw QuoteTick BUY Ask/SELL Bid","trend_gate":{"M1_EMA":"last completed M5 EMA50>EMA200 and rising for buy; inverse for sell","M5_BB":"last completed M15 EMA50>EMA200 and rising for buy; inverse for sell"},"results":{"ema_m1_m5_trend":ema_s.result(),"bb_m5_m15_trend":bb_s.result()}}
 out=Path("results/amos-bbema-mtf")/a.experiment_id/"summary.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
