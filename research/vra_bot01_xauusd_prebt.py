#!/usr/bin/env python3
"""VRA BOT-01 XAUUSD pre-BT. BAR DATA ONLY; NOT Raw Tick certification."""
from pathlib import Path
import csv, json, math, statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/vra-bot01-prebt"; OUT.mkdir(parents=True,exist_ok=True)
FILES={"M1":"XAUUSD_M1_2026Q1Q2.csv","M5":"XAUUSD_M5_2026Q1Q2.csv","H1":"XAUUSD_H1_2026Q1Q2.csv"}
FAST=[5,10,20,30,50]; SLOW=[50,100,150,200]; THR=[0.0,0.0005,0.001,0.002]
COSTS=[0.0,0.0001,0.00025]

def load(tf):
 p=ROOT/"csv/XAUUSD"/FILES[tf]; rows=[]
 with p.open() as f:
  for r in csv.DictReader(f):
   rows.append((r["datetime"],float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"])))
 return rows

def resample15(rows):
 out=[]
 for i in range(0,len(rows)-14,15):
  b=rows[i:i+15]; out.append((b[0][0],b[0][1],max(x[2] for x in b),min(x[3] for x in b),b[-1][4]))
 return out

def sma(x,n):
 o=[None]*len(x); s=0.0
 for i,v in enumerate(x):
  s+=v
  if i>=n:s-=x[i-n]
  if i>=n-1:o[i]=s/n
 return o

def bt(rows,fast,slow,thr,cost,start,end):
 c=[x[4] for x in rows]; a=sma(c,fast); b=sma(c,slow); eq=1.; peak=1.; dd=0.; rs=[]; prev=0; changes=0
 for i in range(max(slow,start),min(end,len(c))):
  state=1 if (a[i-1] is not None and b[i-1] and a[i-1]/b[i-1]-1>thr) else 0
  r=state*(c[i]/c[i-1]-1)
  if state!=prev:r-=cost*abs(state-prev);changes+=1;prev=state
  eq*=1+r; peak=max(peak,eq); dd=min(dd,eq/peak-1); rs.append(r)
 if not rs:return None
 mu=sum(rs)/len(rs); sd=statistics.stdev(rs) if len(rs)>1 else 0
 pf=sum(x for x in rs if x>0)/abs(sum(x for x in rs if x<0)) if any(x<0 for x in rs) else 999.
 wr=sum(x>0 for x in rs if x!=0)/max(1,sum(x!=0 for x in rs))
 return {"return":eq-1,"max_dd":dd,"pf":pf,"wr_active_bars":wr,"changes":changes,"mean_bar":mu,"sd_bar":sd}

allout={}
for tf in ["M1","M5","M15","H1"]:
 rows=resample15(load("M1")) if tf=="M15" else load(tf)
 cut=int(len(rows)*.60); candidates=[]
 for f in FAST:
  for s in SLOW:
   if f>=s:continue
   for t in THR:
    z=bt(rows,f,s,t,0.0001,s,cut)
    if z:candidates.append((z["return"]/max(0.01,abs(z["max_dd"])),f,s,t,z))
 candidates.sort(reverse=True,key=lambda q:q[0]); chosen=candidates[0]
 _,f,s,t,isr=chosen
 oos={str(c):bt(rows,f,s,t,c,cut,len(rows)) for c in COSTS}
 allout[tf]={"rows":len(rows),"split":cut,"selected":{"fast":f,"slow":s,"threshold":t},"IS":isr,"OOS_cost_scenarios":oos}
with (OUT/"summary.json").open("w") as f:json.dump(allout,f,indent=2)
with (OUT/"summary.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["tf","fast","slow","threshold","is_return","is_dd","oos_cost","oos_return","oos_dd","oos_pf","changes"])
 for tf,v in allout.items():
  for cost,z in v["OOS_cost_scenarios"].items():
   w.writerow([tf,*v["selected"].values(),v["IS"]["return"],v["IS"]["max_dd"],cost,z["return"],z["max_dd"],z["pf"],z["changes"]])
print(json.dumps(allout,indent=2))
