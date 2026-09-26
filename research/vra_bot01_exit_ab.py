#!/usr/bin/env python3
"""BOT-01 M5 fixed adaptive entry, exit A/B. BAR research only; no Raw Tick certification."""
from pathlib import Path
import csv,statistics,json
R=Path(__file__).resolve().parents[1];O=R/"results/vra-bot01-exit-ab";O.mkdir(parents=True,exist_ok=True)
P=R/"csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv"
rows=[]
with P.open() as f:
 for x in csv.DictReader(f): rows.append((x["datetime"],float(x["open"]),float(x["high"]),float(x["low"]),float(x["close"])))
c=[x[4] for x in rows];h=[x[2] for x in rows];l=[x[3] for x in rows]
def roll(a,n):
 o=[None]*len(a)
 for i in range(n-1,len(a)):o[i]=sum(a[i-n+1:i+1])/n
 return o
ma20=roll(c,20);ma50=roll(c,50)
tr=[0]+[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(1,len(c))]
atr=roll(tr,14)
vol=roll([abs(c[i]/c[i-1]-1) if i else 0 for i in range(len(c))],20)
cut=int(len(c)*.6); COST=.0001
def run(st,en,mode,trail_k=0,hold=0):
 eq=peak=1.;dd=0.;pos=0;entry=0.;hi=0.;age=0;trade=[];cur=0.;changes=0
 for i in range(max(st,51),min(en,len(c))):
  th=.001+1.5*(atr[i-1]/c[i-1] if atr[i-1] else 0)
  vv=[x for x in vol[max(20,i-500):i] if x is not None];med=statistics.median(vv) if vv else 0
  sig=(ma20[i-1]/ma50[i-1]-1)>th and vol[i-1] is not None and vol[i-1]<=med*1.75
  new=pos
  if not pos and sig:new=1;entry=c[i-1];hi=c[i-1];age=0;cur=-COST
  elif pos:
   age+=1;hi=max(hi,c[i-1])
   exit_ma=not sig
   exit_tr=(trail_k>0 and atr[i-1] and c[i-1] < hi-trail_k*atr[i-1])
   exit_tm=(hold>0 and age>=hold)
   if mode=="ma": ex=exit_ma
   elif mode=="trail": ex=exit_tr
   elif mode=="ma_trail": ex=exit_ma or exit_tr
   elif mode=="time": ex=exit_ma or exit_tm
   else: ex=exit_ma
   if ex:new=0
  rr=pos*(c[i]/c[i-1]-1)
  if new!=pos:
   rr-=COST;changes+=1
   if pos and not new:cur+=rr;trade.append(cur);cur=0
  elif pos:cur+=rr
  pos=new;eq*=1+rr;peak=max(peak,eq);dd=min(dd,eq/peak-1)
 gp=sum(x for x in trade if x>0);gl=-sum(x for x in trade if x<0)
 return {"return":eq-1,"pf":gp/gl if gl else 999,"max_dd":dd,"trades":len(trade),"wr":sum(x>0 for x in trade)/len(trade) if trade else 0,"changes":changes}
cfg=[("ma",0,0)]+[("trail",k,0) for k in [.5,1,1.5,2,2.5,3]]+[("ma_trail",k,0) for k in [.5,1,1.5,2,2.5,3]]+[("time",0,n) for n in [12,24,48,96,192]]
res=[]
for m,k,n in cfg:
 a=run(0,cut,m,k,n);b=run(cut,len(c),m,k,n)
 res.append({"mode":m,"trail_k":k,"hold":n,"IS":a,"OOS":b})
with (O/"exit_ab.json").open("w") as f:json.dump(res,f,indent=2)
with (O/"exit_ab.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["mode","trail_k","hold","is_ret","is_pf","is_dd","oos_ret","oos_pf","oos_dd","oos_wr","oos_trades"])
 for x in res:w.writerow([x["mode"],x["trail_k"],x["hold"],x["IS"]["return"],x["IS"]["pf"],x["IS"]["max_dd"],x["OOS"]["return"],x["OOS"]["pf"],x["OOS"]["max_dd"],x["OOS"]["wr"],x["OOS"]["trades"]])
print(json.dumps(res,indent=2))
