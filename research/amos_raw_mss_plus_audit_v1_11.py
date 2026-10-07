#!/usr/bin/env python3
"""
AMOS Raw MSS+ Gate Audit v1.11
Correction only: replaces the simplified single MSS checkpoint used by v1.10
with the repository's prior MSS+ structural definition.

Prior MSS+ structural sequence (frozen from amos_reversal V6/V8 research):
  Sweep -> CISD (<=8 bars) -> MSS (<=12 bars after CISD)
  -> Displacement >=0.65 ATR on MSS bar or next 4 bars
  -> POI exists: FVG / IFVG / BPR
Then the existing downstream diagnostic remains:
  Volume Influx -> Equilibrium -> Pullback

This is an attrition audit only. No ML threshold optimization and no entry tuning.
"""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

def fpx(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def atr(d,n=14):
 p=d.close.shift(1); tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
 return tr.rolling(n).mean()
def feature(d):
 z=d.copy();z["atr"]=atr(z);z["body"]=(z.close-z.open).abs();z["disp"]=z.body/(z.atr+1e-12)
 z["swing_hi"]=z.high.shift(1).rolling(5).max();z["swing_lo"]=z.low.shift(1).rolling(5).min()
 z["bull_fvg"]=z.low>z.high.shift(2);z["bear_fvg"]=z.high<z.low.shift(2)
 z["vma"]=z.volume.shift(1).rolling(20).mean()
 return z
def fvg_zone(z,i,side):
 if i<2:return None
 r=z.iloc[i]
 if side==1 and bool(r.bull_fvg):
  lo=float(z.high.iloc[i-2]);hi=float(r.low);return min(lo,hi),max(lo,hi),"FVG"
 if side==-1 and bool(r.bear_fvg):
  lo=float(r.high);hi=float(z.low.iloc[i-2]);return min(lo,hi),max(lo,hi),"FVG"
 return None
def prior_fvgs(z,i,side,lookback=12):
 out=[]
 for k in range(max(2,i-lookback),i):
  q=z.iloc[k]
  if side==1 and bool(q.bear_fvg):
   lo=float(q.high);hi=float(z.low.iloc[k-2]);out.append((k,min(lo,hi),max(lo,hi)))
  elif side==-1 and bool(q.bull_fvg):
   lo=float(z.high.iloc[k-2]);hi=float(q.low);out.append((k,min(lo,hi),max(lo,hi)))
 return out
def poi_zone(z,i,side):
 cur=fvg_zone(z,i,side); inv=[];r=z.iloc[i]
 for k,lo,hi in prior_fvgs(z,i,side):
  if (float(r.close)>hi if side==1 else float(r.close)<lo):inv.append((k,lo,hi))
 if cur and inv:
  clo,chi,_=cur
  ov=[(max(clo,lo),min(chi,hi)) for _,lo,hi in inv if max(clo,lo)<=min(chi,hi)]
  if ov:return (*ov[-1],"BPR")
 if cur:return cur
 if inv:
  _,lo,hi=inv[-1];return lo,hi,"IFVG"
 return None
def eqzone(di,se,de):
 if di<0:
  x=se-de;return (de+x*.50,de+x*.79) if x>0 else None
 x=de-se;return (de-x*.79,de-x*.50) if x>0 else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
 bid=np.fromiter((fpx(x.bid_price) for x in ticks),dtype=float,count=len(ticks))
 raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"bid":bid})
 g=raw.set_index("datetime").bid.resample("1min")
 d=pd.DataFrame({"open":g.first(),"high":g.max(),"low":g.min(),"close":g.last(),"volume":g.count()}).dropna().reset_index()
 z=feature(d)

 c=Counter();reset=Counter();poi=Counter();events=[];states={1:None,-1:None};down=[]
 for i in range(30,len(z)):
  r=z.iloc[i];p=z.iloc[i-1]
  if not np.isfinite(r.atr) or r.atr<=0:continue
  # expire structural state as in prior implementation
  for side in (1,-1):
   s=states[side]
   if s and i-s["sweep_i"]>28:
    wait="CISD" if s["cisd_i"] is None else ("MSS" if s["mss_i"] is None else "DISP_POI")
    reset["timeout_"+wait]+=1;states[side]=None
  bull=bool(r.low<r.swing_lo and r.close>r.swing_lo);bear=bool(r.high>r.swing_hi and r.close<r.swing_hi)
  if bull:
   states[1]={"sweep_i":i,"extreme":float(r.low),"mss_ref":float(z.high.iloc[max(0,i-3):i].max()),"cisd_i":None,"mss_i":None,"disp_seen":False};c["sweep"]+=1
  if bear:
   states[-1]={"sweep_i":i,"extreme":float(r.high),"mss_ref":float(z.low.iloc[max(0,i-3):i].min()),"cisd_i":None,"mss_i":None};c["sweep"]+=1
  for side in (1,-1):
   s=states[side]
   if not s:continue
   if s["cisd_i"] is None:
    if i>s["sweep_i"] and i-s["sweep_i"]<=8:
     hit=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
     if hit:s["cisd_i"]=i;c["cisd"]+=1
    elif i-s["sweep_i"]>8:
     reset["timeout_CISD"]+=1;states[side]=None
    continue
   if s["mss_i"] is None:
    if i-s["cisd_i"]>12:reset["timeout_MSS"]+=1;states[side]=None;continue
    if i<=s["cisd_i"]:continue
    hit=(r.close>s["mss_ref"]) if side==1 else (r.close<s["mss_ref"])
    if hit:s["mss_i"]=i;c["mss"]+=1
    else:continue
   if i<s["mss_i"] or i-s["mss_i"]>4:
    if i-s["mss_i"]>4:reset["timeout_DISP_POI"]+=1;states[side]=None
    continue
   if float(r.disp)<.65:continue
   c["displacement"]+=1
   zone=poi_zone(z,i,side)
   if zone is None:continue
   lo,hi,kind=zone;c["mss_plus"]+=1;poi[kind]+=1
   events.append({"time":str(r.datetime),"stage":"MSS_PLUS","dir":side,"poi_type":kind,"sweep_i":s["sweep_i"],"cisd_i":s["cisd_i"],"mss_i":s["mss_i"],"disp_i":i})
   down.append({"i":i,"side":side,"se":float(s["extreme"]),"de":float(r.close),"zone":zone,"stage":"volume","age":0})
   states[side]=None

 # Existing downstream P5 diagnostic, unchanged thresholds/order after corrected MSS+.
 completed=[]
 for s in down:
  for j in range(s["i"]+1,min(len(z),s["i"]+1+8+12+8+3)):
   b=z.iloc[j];s["age"]+=1
   if s["stage"]=="volume":
    vr=float(b.volume/b.vma) if np.isfinite(b.vma) and b.vma>0 else np.nan
    if np.isfinite(vr) and vr>=1.10:
     c["volume"]+=1;s["stage"]="equilibrium";s["age"]=0
    elif s["age"]>8:reset["timeout_volume"]+=1;break
   elif s["stage"]=="equilibrium":
    zone=eqzone(s["side"],s["se"],s["de"])
    if zone is not None:
     c["equilibrium"]+=1;s["eq"]=zone;s["stage"]="pullback";s["age"]=0
    elif s["age"]>12:reset["timeout_equilibrium"]+=1;break
   elif s["stage"]=="pullback":
    lo,hi=sorted(s["eq"])
    if b.high>=lo and b.low<=hi:
     c["pullback"]+=1;c["completed_sequence"]+=1;completed.append(j);break
    if s["age"]>8:reset["timeout_pullback"]+=1;break

 order=["sweep","cisd","mss","displacement","mss_plus","volume","equilibrium","pullback"]
 survival={};prev=None
 for k in order:
  n=int(c[k]);survival[k]={"count":n,"from_previous_pct":None if prev in (None,0) else 100*n/prev};prev=n
 result={"version":"v1.11","mode":"RAW_MSS_PLUS_CORRECTION_AUDIT",
  "raw_period":{"start":str(pd.Timestamp(ts[0],unit="ns")),"end":str(pd.Timestamp(ts[-1],unit="ns")),"raw_ticks":len(ticks),"m1_bid_bars":len(z)},
  "mss_plus_definition":{"sequence":["Sweep","CISD","MSS","Displacement","POI(FVG/IFVG/BPR)"],"cisd_window_bars":8,"mss_window_after_cisd_bars":12,"displacement_min_atr":0.65,"displacement_window_from_mss_bars":4,"poi_types":["FVG","IFVG","BPR"]},
  "downstream_order_unchanged":["Volume Influx","Equilibrium","Pullback"],
  "logic_change":"MSS checkpoint corrected to prior MSS+ composite only; downstream thresholds unchanged",
  "counts":dict(c),"survival":survival,"poi_type_counts":dict(poi),"reset_reasons":dict(reset),
  "note":"Diagnostic only; no ML threshold tuning and no production entry promotion."}
 pd.DataFrame(events).to_csv(out/"mss_plus_events.csv",index=False)
 (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
