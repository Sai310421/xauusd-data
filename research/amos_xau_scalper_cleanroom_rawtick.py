from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from amos_gold_5in1_v101_rawtick_bt import load, days, bars

def prep(b):
 b=b.copy(); c=b.close
 b["ema20"]=c.ewm(span=20,adjust=False).mean(); b["ema50"]=c.ewm(span=50,adjust=False).mean()
 d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/7,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/7,adjust=False).mean()
 b["rsi"]=100-100/(1+up/dn.replace(0,np.nan))
 p=c.shift(1); tr=pd.concat([b.high-b.low,(b.high-p).abs(),(b.low-p).abs()],axis=1).max(axis=1)
 b["atr"]=tr.ewm(alpha=1/14,adjust=False).mean(); b["atr_med50"]=b.atr.rolling(50).median()
 ll=b.low.rolling(5).min(); hh=b.high.rolling(5).max(); rk=100*(c-ll)/(hh-ll).replace(0,np.nan)
 b["k"]=rk.rolling(3).mean(); b["d"]=b.k.rolling(3).mean()
 return b

def plow(b,i,p=2):
 return i>=p and i+p<len(b) and b.low.iloc[i]<b.low.iloc[i-p:i].min() and b.low.iloc[i]<b.low.iloc[i+1:i+p+1].min()
def phigh(b,i,p=2):
 return i>=p and i+p<len(b) and b.high.iloc[i]>b.high.iloc[i-p:i].max() and b.high.iloc[i]>b.high.iloc[i+1:i+p+1].max()

def div(b,i,side,lookback,p=2):
 pts=[]
 for j in range(i-p,max(p,i-lookback-p*2)-1,-1):
  if (plow(b,j,p) if side==1 else phigh(b,j,p)):
   pts.append(j)
   if len(pts)==2: break
 if len(pts)<2:return False
 n,o=pts
 if side==1:return b.low.iloc[n]<b.low.iloc[o] and b.rsi.iloc[n]>b.rsi.iloc[o]
 return b.high.iloc[n]>b.high.iloc[o] and b.rsi.iloc[n]<b.rsi.iloc[o]

def sigs(b,lookback,stoch,gate):
 out=[]
 for i in range(55,len(b)-1):
  r,p=b.iloc[i],b.iloc[i-1]
  if not 13<=r.t.hour<17:continue
  if gate=="median50" and (pd.isna(r.atr_med50) or r.atr<r.atr_med50):continue
  bc=p.k<=p.d and r.k>r.d and min(p.k,r.k)<=stoch
  sc=p.k>=p.d and r.k<r.d and max(p.k,r.k)>=100-stoch
  if r.ema20>r.ema50 and bc and div(b,i,1,lookback):out.append((b.t.iloc[i+1],1,float(r.atr)))
  elif r.ema20<r.ema50 and sc and div(b,i,-1,lookback):out.append((b.t.iloc[i+1],-1,float(r.atr)))
 return out

def sim(t,sg,capital=1000,lot=.01):
 mult=100*lot; eq=peak=capital; mdd=0.; op=[]; tr=[]; si=0
 for row in t.itertuples():
  tm,bid,ask=row.t,row.bid,row.ask; keep=[]
  for p in op:
   px=bid if p["s"]==1 else ask
   sl=px<=p["sl"] if p["s"]==1 else px>=p["sl"]; tp=px>=p["tp"] if p["s"]==1 else px<=p["tp"]
   if sl or tp:
    pnl=(px-p["e"])*p["s"]*mult;eq+=pnl;tr.append(pnl)
   else:keep.append(p)
  op=keep
  while si<len(sg) and sg[si][0]<=tm:
   st,s,a=sg[si];si+=1
   if len(op)>=2 or ask-bid>.50:continue
   e=ask if s==1 else bid;op.append({"s":s,"e":e,"sl":e-s*a,"tp":e+s*1.5*a})
  fl=sum(((bid if p["s"]==1 else ask)-p["e"])*p["s"]*mult for p in op);cur=eq+fl;peak=max(peak,cur);mdd=max(mdd,(peak-cur)/peak)
 if len(t):
  for p in op:
   px=t.bid.iloc[-1] if p["s"]==1 else t.ask.iloc[-1]; pnl=(px-p["e"])*p["s"]*mult;eq+=pnl;tr.append(pnl)
 a=np.array(tr,float)
 if not len(a):return {"N":0,"WR_pct":0.,"PF":None,"EV_USD":0.,"Net_USD":0.,"Return_pct":0.,"MaxDD_pct":mdd*100}
 gp=a[a>0].sum();gl=-a[a<0].sum()
 return {"N":len(a),"WR_pct":float((a>0).mean()*100),"PF":float(gp/gl) if gl>0 else None,"EV_USD":float(a.mean()),"Net_USD":float(a.sum()),"Return_pct":float(a.sum()/capital*100),"MaxDD_pct":float(mdd*100)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--start",default="2026-07-27");ap.add_argument("--days",type=int,default=21);ap.add_argument("--workers",type=int,default=8);ap.add_argument("--cache",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t=load(days(a.start,a.days),a.workers,a.cache);b=prep(bars(t,"5min"))
 vv=[]
 for lb in (8,12,20):
  for st in (20.,30.,40.):
   for g in ("none","median50"):
    vv.append({"lookback":lb,"stoch":st,"atr_gate":g,**sim(t,sigs(b,lb,st,g))})
 pd.DataFrame(vv).to_csv(out/"variant_kpi.csv",index=False)
 default=next(x for x in vv if x["lookback"]==12 and x["stoch"]==30. and x["atr_gate"]=="none")
 eligible=[x for x in vv if x["N"]>=10 and x["PF"] is not None]
 best=max(eligible,key=lambda x:x["PF"]) if eligible else max(vv,key=lambda x:x["N"])
 rep={"verification":"INDEPENDENT_RAW_BIDASK_FALSIFICATION","source":"Dukascopy BI5","raw_ticks":len(t),"period":{"start":a.start,"business_days":a.days},"public_claim":{"N":847,"WR_pct":77.3,"PF":2.31,"MaxDD_pct":4.2,"claim_period":"2023-01 to 2024-04"},"default_cleanroom":default,"best_in_sample_variant":best,"limits":["Original proprietary code not executed.","Exact ATR threshold and RSI-divergence definition are not specified in README.","Repository backtester references Yahoo GC=F futures 5m, while this audit uses XAUUSD spot raw bid/ask.","Best variant is in-sample exploration, not OOS evidence."]}
 (out/"summary.json").write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=="__main__":main()

# workflow-trigger: audit-v1
