#!/usr/bin/env python3
"""BOT-01 Frozen Candidate #1 sizing stress. M5 20/50, adaptive threshold, vol gate, MA exit."""
from pathlib import Path
import csv,statistics,json
R=Path(__file__).resolve().parents[1];O=R/"results/vra-bot01-sizing";O.mkdir(parents=True,exist_ok=True)
rows=[]
with (R/"csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv").open() as f:
 for x in csv.DictReader(f): rows.append((x["datetime"],float(x["high"]),float(x["low"]),float(x["close"])))
c=[x[3] for x in rows];h=[x[1] for x in rows];l=[x[2] for x in rows]
def roll(a,n):
 o=[None]*len(a)
 for i in range(n-1,len(a)):o[i]=sum(a[i-n+1:i+1])/n
 return o
m20=roll(c,20);m50=roll(c,50)
tr=[0]+[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(1,len(c))]
atr=roll(tr,14);vol=roll([abs(c[i]/c[i-1]-1) if i else 0 for i in range(len(c))],20)
cut=int(len(c)*.6);COST=.0001
def run(mult):
 eq=peak=1.;dd=0.;pos=0;trade=[];cur=0.;changes=0
 for i in range(max(cut,51),len(c)):
  th=.001+1.5*(atr[i-1]/c[i-1] if atr[i-1] else 0)
  vv=[x for x in vol[max(20,i-500):i] if x is not None];med=statistics.median(vv) if vv else 0
  sig=(m20[i-1]/m50[i-1]-1)>th and vol[i-1] is not None and vol[i-1]<=med*1.75
  new=1 if sig else 0
  rr=pos*(c[i]/c[i-1]-1)*mult
  if new!=pos:
   rr-=COST*mult;changes+=1
   if not pos and new:cur=-COST*mult
   elif pos and not new:cur+=rr;trade.append(cur);cur=0
  elif pos:cur+=rr
  pos=new
  if 1+rr<=0:return {"mult":mult,"ruin":True,"return":-1,"max_dd":-1}
  eq*=1+rr;peak=max(peak,eq);dd=min(dd,eq/peak-1)
 gp=sum(x for x in trade if x>0);gl=-sum(x for x in trade if x<0)
 return {"mult":mult,"ruin":False,"return":eq-1,"max_dd":dd,"pf":gp/gl if gl else 999,"wr":sum(x>0 for x in trade)/len(trade),"trades":len(trade),"changes":changes,"return_dd":(eq-1)/abs(dd) if dd else 999}
res=[run(x) for x in [1,1.5,2,2.5,3,3.5,4,4.5,5,5.25,5.5,5.75,6]]
eligible=[x for x in res if not x["ruin"] and abs(x["max_dd"])<=.04]
best=max(eligible,key=lambda x:x["return"]) if eligible else None
with (O/"sizing_stress.json").open("w") as f:json.dump({"dd_limit":.04,"results":res,"max_return_under_dd4":best},f,indent=2)
with (O/"sizing_stress.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=res[0].keys());w.writeheader();w.writerows(res)
print(json.dumps({"dd_limit":.04,"results":res,"max_return_under_dd4":best},indent=2))
