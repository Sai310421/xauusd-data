#!/usr/bin/env python3
"""
AMOS Raw Gate Attrition Audit v1.10
Diagnostic only. No trading rule, threshold, or gate order changes.
Replays the frozen v1.6 P5 state machine on causal M1 BID bars built from
Dukascopy/Nautilus Raw QuoteTicks and counts exactly where setups disappear.
"""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

SEQ=["sweep","mss","volume","equilibrium","pullback"]
TTL={"mss":8,"volume":8,"equilibrium":12,"pullback":8}

def fpx(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def atr(d,n=14):
 p=d.close.shift(1);tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def ranges(d):
 out={};mins=d.datetime.dt.hour*60+d.datetime.dt.minute
 for day,idx in d.groupby(d.datetime.dt.normalize()).groups.items():
  q=d.loc[idx];qm=mins.loc[idx]
  def rg(a,b):
   w=q[(qm>=a)&(qm<b)];return None if w.empty else (float(w.high.max()),float(w.low.min()))
  out[pd.Timestamp(day)]={"asia":rg(0,360),"london":rg(420,600)}
 return out
def ref(R,t):
 m=t.hour*60+t.minute;r=R.get(t.normalize(),{})
 if 420<=m<600 and r.get("asia"):return (*r["asia"],"ASIA")
 if 810<=m<990:
  x=r.get("london") or r.get("asia")
  if x:return (*x,"LONDON")
 return None
def eqzone(di,se,de):
 if di<0:
  z=se-de;return (de+z*.50,de+z*.79) if z>0 else None
 z=de-se;return (de-z*.79,de-z*.50) if z>0 else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
 if not ticks:raise SystemExit("no XAUUSD raw quote ticks")
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
 bid=np.fromiter((fpx(x.bid_price) for x in ticks),dtype=float,count=len(ticks))
 raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"bid":bid})
 g=raw.set_index("datetime").bid.resample("1min")
 d=pd.DataFrame({"open":g.first(),"high":g.max(),"low":g.min(),"close":g.last(),"volume":g.count()}).dropna().reset_index()
 d["atr"]=atr(d);d["vma"]=d.volume.shift(1).rolling(20).mean();R=ranges(d)

 counts=Counter();timeouts=Counter();by_session=defaultdict(Counter);by_dir=defaultdict(Counter);events=[]
 step=0;age=0;di=0;se=de=0.;zone=None;sweep_i=-1;sweep_time=None;session=None
 def mark(stage,t,sess,direction,extra=None):
  counts[stage]+=1;by_session[str(sess)][stage]+=1;by_dir["BUY" if direction>0 else "SELL"][stage]+=1
  row={"time":str(t),"stage":stage,"session":sess,"dir":direction}
  if extra:row.update(extra)
  events.append(row)

 for i in range(60,len(d)-61):
  b=d.iloc[i];t=b.datetime
  if not np.isfinite(b.atr) or b.atr<=0:continue
  rr=ref(R,t)
  if not rr:
   if step:
    timeouts["session_or_reference_end"]+=1
    events.append({"time":str(t),"stage":"RESET_SESSION_END","session":session,"dir":di,"waiting_for":SEQ[step]})
    step=0;age=0;di=0;zone=None
   continue
  counts["eligible_reference_bars"]+=1
  rh,rl,sess=rr;mn=b.atr*.03;mx=b.atr*.80
  sh=b.high>rh+mn and b.close<rh and b.high-rh<=mx
  sl=b.low<rl-mn and b.close>rl and rl-b.low<=mx
  if step>0:
   age+=1;want=SEQ[step]
   if age>TTL.get(want,8):
    timeouts["timeout_wait_"+want]+=1
    events.append({"time":str(t),"stage":"TIMEOUT","session":session,"dir":di,"waiting_for":want,"age":age})
    step=0;age=0;di=0;zone=None
  want=SEQ[step];event=False
  if want=="sweep":
   if sh or sl:
    di=-1 if sh else 1;se=float(b.high if sh else b.low);de=float(b.low if sh else b.high)
    sweep_i=i;sweep_time=t;session=sess;event=True
  elif di:
   se=max(se,float(b.high)) if di<0 else min(se,float(b.low))
   de=min(de,float(b.low)) if di<0 else max(de,float(b.high))
   struct=float(d.low.iloc[i-8:i].min()) if di<0 else float(d.high.iloc[i-8:i].max())
   disp=abs(b.close-b.open)>=b.atr*.55
   if want=="mss":
    event=(b.close<struct-b.atr*.02 and disp) if di<0 else (b.close>struct+b.atr*.02 and disp)
   elif want=="volume":
    vr=float(b.volume/b.vma) if np.isfinite(b.vma) and b.vma>0 else np.nan
    event=bool(np.isfinite(vr) and vr>=1.10)
   elif want=="equilibrium":
    zone=eqzone(di,se,de);event=zone is not None
   elif want=="pullback" and zone:
    lo,hi=sorted(zone);event=b.high>=lo and b.low<=hi
  if not event:continue
  mark(want,t,session or sess,di,{"bars_from_sweep":i-sweep_i})
  step+=1;age=0
  if step==len(SEQ):
   counts["completed_sequence"]+=1;by_session[str(session)]["completed_sequence"]+=1;by_dir["BUY" if di>0 else "SELL"]["completed_sequence"]+=1
   events.append({"time":str(t),"stage":"COMPLETED","session":session,"dir":di,"bars_from_sweep":i-sweep_i})
   step=0;age=0;di=0;zone=None

 stages=["sweep","mss","volume","equilibrium","pullback"]
 survival={}
 prev=None
 for st in stages:
  n=int(counts[st]);survival[st]={"count":n,"from_previous_pct":None if prev in (None,0) else 100*n/prev}
  prev=n
 result={"version":"v1.10","mode":"RAW_GATE_ATTRITION_DIAGNOSTIC_ONLY",
  "raw_period":{"start":str(pd.Timestamp(ts[0],unit="ns")),"end":str(pd.Timestamp(ts[-1],unit="ns")),
                "raw_ticks":len(ticks),"m1_bid_bars":len(d)},
  "fixed_gate_order":["Liquidity Sweep","MSS","Volume Influx","Equilibrium","Pullback"],
  "logic_change":False,"counts":dict(counts),"survival":survival,"reset_reasons":dict(timeouts),
  "by_session":{k:dict(v) for k,v in by_session.items()},"by_direction":{k:dict(v) for k,v in by_dir.items()},
  "note":"Counts are stage events in the exact frozen v1.6 sequence screen. This audit does not optimize thresholds or create entries."}
 pd.DataFrame(events).to_csv(out/"gate_events.csv",index=False)
 (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
