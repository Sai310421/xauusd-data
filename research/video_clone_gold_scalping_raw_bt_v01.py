import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT=Path("bt_results/video_clone_gold_scalping_v01"); OUT.mkdir(parents=True,exist_ok=True)
CANDS=[
 Path("csv/XAUUSD/XAUUSD_TICK.csv"),Path("csv/XAUUSD/XAUUSD_ticks.csv"),
 Path("csv/XAUUSD/XAUUSD_RAW.csv"),Path("csv/XAUUSD/XAUUSD_tick.csv")
]
src=next((p for p in CANDS if p.exists()),None)
if src is None:
 raise SystemExit("FAIL_CLOSED: XAUUSD raw tick CSV not found; OHLC fallback is forbidden")
df=pd.read_csv(src); df.columns=[c.lower() for c in df.columns]
timecol=next((c for c in ["datetime","timestamp","time"] if c in df.columns),None)
if timecol is None: raise SystemExit("FAIL_CLOSED: timestamp column missing")
df[timecol]=pd.to_datetime(df[timecol],errors="coerce"); df=df.dropna(subset=[timecol]).sort_values(timecol)
if "bid" in df and "ask" in df: bid=df.bid.to_numpy(float); ask=df.ask.to_numpy(float)
elif "price" in df:
 bid=df.price.to_numpy(float); ask=bid.copy()
else: raise SystemExit("FAIL_CLOSED: raw bid/ask or price column missing")
ts=df[timecol].to_numpy()
LOT_SEQ=[.01,.01,.02,.06]
def pnl(d,e,x,l): return (x-e)*l*100 if d==1 else (e-x)*l*100
def run(dist,target,max_flips=3):
 eq=1000.; peak=eq; maxdd=0.; cycles=[]; i=0
 while i<len(df)-2:
  d=1 if (i==0 or (bid[i]-bid[max(0,i-10)])>=0) else -1
  entry=ask[i] if d==1 else bid[i]; legs=[(d,entry,LOT_SEQ[0])]; flips=0; start=i
  while i<len(df)-1:
   i+=1; mark=sum(pnl(dd,e,bid[i] if dd==1 else ask[i],l) for dd,e,l in legs)
   maxdd=max(maxdd,(peak-(eq+mark))/peak*100)
   if mark>=target:
    eq+=mark; peak=max(peak,eq); cycles.append((mark,flips,i-start)); break
   if flips<max_flips:
    anchor=legs[-1][1]; adverse=(bid[i]<=anchor-dist) if legs[-1][0]==1 else (ask[i]>=anchor+dist)
    if adverse:
     nd=-legs[-1][0]; ne=bid[i] if nd==-1 else ask[i]; flips+=1
     legs.append((nd,ne,LOT_SEQ[min(flips,len(LOT_SEQ)-1)]))
   if i-start>200000:
    eq+=mark; cycles.append((mark,flips,i-start)); break
  i+=1
 if not cycles:return {}
 c=pd.DataFrame(cycles,columns=["pnl","flips","ticks"]); gp=c.loc[c.pnl>0,"pnl"].sum(); gl=-c.loc[c.pnl<0,"pnl"].sum()
 return dict(N=len(c),CycleWR=(c.pnl>0).mean()*100,PF=(gp/gl if gl else 9999),EV=c.pnl.mean(),Net=c.pnl.sum(),ReturnPct=(eq/1000-1)*100,MaxDDPct=maxdd,AvgFlips=c.flips.mean(),MaxFlips=int(c.flips.max()),AvgCycleTicks=c.ticks.mean())
rows=[]
for dist in [.05,.10,.15,.20,.25,.30,.40,.50]:
 for target in [1,2,3,5,7,10]:
  rows.append(dict(stop_distance=dist,basket_target=target,**run(dist,target)))
o=pd.DataFrame(rows); o.to_csv(OUT/"grid.csv",index=False)
summary={"source":str(src),"rows":len(df),"start":str(df[timecol].min()),"end":str(df[timecol].max()),"top_by_pf":o.sort_values(["PF","MaxDDPct"],ascending=[False,True]).head(20).to_dict("records")}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
