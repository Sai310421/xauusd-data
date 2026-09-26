#!/usr/bin/env python3
"""BOT-01 adaptive A/B: baseline vs ATR-adaptive threshold vs trend+vol gate. BAR research only."""
from pathlib import Path
import csv,json,statistics,math
R=Path(__file__).resolve().parents[1]; O=R/"results/vra-bot01-adaptive";O.mkdir(parents=True,exist_ok=True)
FILES={"M5":"XAUUSD_M5_2026Q1Q2.csv"}
def load():
 p=R/"csv/XAUUSD"/FILES["M5"]; out=[]
 with p.open() as f:
  for x in csv.DictReader(f):out.append((x["datetime"],float(x["open"]),float(x["high"]),float(x["low"]),float(x["close"])))
 return out
def r15(a):
 z=[]
 for i in range(0,len(a)-14,15):
  b=a[i:i+15];z.append((b[0][0],b[0][1],max(x[2] for x in b),min(x[3] for x in b),b[-1][4]))
 return z
def roll(a,n,fn):
 o=[None]*len(a)
 for i in range(n-1,len(a)):o[i]=fn(a[i-n+1:i+1])
 return o
def test(rows,f,s,base,katr,gate,cost,st,en):
 c=[x[4] for x in rows]; h=[x[2] for x in rows];l=[x[3] for x in rows]
 maF=roll(c,f,statistics.mean);maS=roll(c,s,statistics.mean)
 tr=[0]+[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(1,len(c))]
 atr=roll(tr,14,statistics.mean); vol=roll([abs(c[i]/c[i-1]-1) if i else 0 for i in range(len(c))],20,statistics.mean)
 eq=peak=1.;dd=0.;pos=0;rs=[];chg=0
 for i in range(max(s,21,st),min(en,len(c))):
  th=base+(katr*(atr[i-1]/c[i-1] if atr[i-1] else 0))
  trend=(maF[i-1]/maS[i-1]-1)>th
  allow=True
  if gate:
   vv=[x for x in vol[max(20,st):i] if x is not None]
   med=statistics.median(vv[-500:]) if vv else 0
   allow=vol[i-1] is not None and vol[i-1] <= med*1.75
  new=1 if trend and allow else 0
  rr=pos*(c[i]/c[i-1]-1)
  if new!=pos:rr-=cost;chg+=1
  pos=new;eq*=1+rr;peak=max(peak,eq);dd=min(dd,eq/peak-1);rs.append(rr)
 gp=sum(x for x in rs if x>0);gl=-sum(x for x in rs if x<0)
 return {"return":eq-1,"max_dd":dd,"pf":gp/gl if gl else 999,"changes":chg}
rows5=load(); data={"M5":rows5,"M15":r15(rows5)};out={}
for tf,rows in data.items():
 cut=int(len(rows)*.6); configs=[]
 for f in [10,20,30,50]:
  for s in [50,100,150,200]:
   if f>=s:continue
   for base in [0,.0005,.001,.002]:
    for k in [0,.25,.5,1,1.5]:
     for gate in [False,True]:
      z=test(rows,f,s,base,k,gate,.0001,s,cut)
      score=z["return"]/max(.01,abs(z["max_dd"]))
      configs.append((score,f,s,base,k,gate,z))
 configs.sort(reverse=True,key=lambda x:x[0])
 top=configs[:10]; evals=[]
 for _,f,s,b,k,g,isr in top:
  oo=test(rows,f,s,b,k,g,.0001,cut,len(rows))
  evals.append({"f":f,"s":s,"base":b,"atr_k":k,"vol_gate":g,"IS":isr,"OOS":oo})
 out[tf]={"rows":len(rows),"split":cut,"top10":evals}
with (O/"adaptive_ab.json").open("w") as f:json.dump(out,f,indent=2)
with (O/"adaptive_ab.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["tf","rank","fast","slow","base","atr_k","vol_gate","is_ret","is_pf","is_dd","oos_ret","oos_pf","oos_dd","changes"])
 for tf,v in out.items():
  for n,x in enumerate(v["top10"],1):w.writerow([tf,n,x["f"],x["s"],x["base"],x["atr_k"],x["vol_gate"],x["IS"]["return"],x["IS"]["pf"],x["IS"]["max_dd"],x["OOS"]["return"],x["OOS"]["pf"],x["OOS"]["max_dd"],x["OOS"]["changes"]])
print(json.dumps(out,indent=2))
