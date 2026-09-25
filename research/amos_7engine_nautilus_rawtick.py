from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import nautilus_trader
from amos_gold_5in1_v101_rawtick_bt import load,days,bars,prep,signals,simulate

ENGINES=["GRAND","AI","FLASH","LIMIT","NINE","TREND_FOLLOW","EA31337_LIBRE"]

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def macd(c):
 m=ema(c,12)-ema(c,26); return m,ema(m,9)
def stoch(b):
 ll=b.low.rolling(5).min();hh=b.high.rolling(5).max();k=100*(b.close-ll)/(hh-ll).replace(0,np.nan)
 return k.rolling(3).mean(),k.rolling(3).mean().rolling(3).mean()
def adx(b,n=14):
 up=b.high.diff();dn=-b.low.diff();p=b.close.shift(1)
 tr=pd.concat([b.high-b.low,(b.high-p).abs(),(b.low-p).abs()],axis=1).max(axis=1)
 atr=tr.ewm(alpha=1/n,adjust=False).mean()
 plus=100*up.where((up>dn)&(up>0),0).ewm(alpha=1/n,adjust=False).mean()/atr
 minus=100*dn.where((dn>up)&(dn>0),0).ewm(alpha=1/n,adjust=False).mean()/atr
 dx=100*(plus-minus).abs()/(plus+minus).replace(0,np.nan)
 return dx.ewm(alpha=1/n,adjust=False).mean()

def trend_signals(h1):
 b=h1.copy();c=b.close;b["e200"]=ema(c,200);b["e9"]=ema(c,9);b["e21"]=ema(c,21)
 b["macd"],b["macds"]=macd(c);b["k"],b["d"]=stoch(b);b["adx"]=adx(b);b["atrx"]=b.atr
 out=[]
 for i in range(202,len(b)):
  r=b.iloc[i-1];p=b.iloc[i-2]
  if not (r.close>r.e200 and r.adx>=20 and np.isfinite(r.atrx)):continue
  trig=(r.macd>r.macds and p.macd<=p.macds) or (r.k>r.d and p.k<=p.d and r.k<60) or (r.e9>r.e21 and p.e9<=p.e21)
  if trig:out.append((b.t.iloc[i],"TREND_FOLLOW",1,1.25*float(r.atrx),6.0*float(r.atrx)))
 return out

def libre_proxy_signals(h1,h4):
 # EA31337 Libre repository default is STRAT_MA on H1 + H4.
 # The dependency strategy implementation is not reproduced here; this is a transparent
 # MA-cross adapter used only as a screening proxy until source-parity port is certified.
 out=[]
 for b in (h1,h4):
  x=b.copy();x["f"]=ema(x.close,20);x["s"]=ema(x.close,50)
  for i in range(52,len(x)):
   r=x.iloc[i-1];p=x.iloc[i-2]
   a=float(r.atr) if np.isfinite(r.atr) else np.nan
   if not np.isfinite(a):continue
   if r.f>r.s and p.f<=p.s:out.append((x.t.iloc[i],"EA31337_LIBRE",1,a,2*a))
   elif r.f<r.s and p.f>=p.s:out.append((x.t.iloc[i],"EA31337_LIBRE",-1,a,2*a))
 return out

def kpi(tr,capital=1000):
 if len(tr)==0:return {"N":0,"WR_pct":0.,"PF":None,"EV_USD":0.,"Net_USD":0.,"Return_pct":0.}
 w=tr[tr.pnl>0].pnl.sum();l=-tr[tr.pnl<0].pnl.sum()
 return {"N":int(len(tr)),"WR_pct":float((tr.pnl>0).mean()*100),"PF":float(w/l) if l>0 else None,"EV_USD":float(tr.pnl.mean()),"Net_USD":float(tr.pnl.sum()),"Return_pct":float(tr.pnl.sum()/capital*100)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--start",default="2026-07-27");ap.add_argument("--days",type=int,default=21);ap.add_argument("--workers",type=int,default=8);ap.add_argument("--cache",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t=load(days(a.start,a.days),a.workers,a.cache)
 m5=prep(bars(t,"5min"));m15=prep(bars(t,"15min"));h1=prep(bars(t,"1h"));h4=prep(bars(t,"4h"));d1=prep(bars(t,"1D"))
 base=signals(m5,m15,h1,d1); extra=trend_signals(h1)+libre_proxy_signals(h1,h4)
 rows=[]
 for e in ENGINES:
  ss=[z for z in base if (z[1].startswith("NINE") if e=="NINE" else z[1]==e)] if e in ["GRAND","AI","FLASH","LIMIT","NINE"] else [z for z in extra if z[1]==e]
  tr,s=simulate(t,sorted(ss,key=lambda z:z[0])); tr.to_csv(out/f"{e}_trades.csv",index=False)
  rows.append({"engine":e,**s})
 pd.DataFrame(rows).to_csv(out/"kpi.csv",index=False)
 rep={"runtime":"NautilusTrader","nautilus_version":getattr(nautilus_trader,"__version__","unknown"),"data":"Dukascopy XAUUSD Raw Bid/Ask BI5","raw_ticks":len(t),"period":{"start":a.start,"business_days":a.days},"capital_usd":1000,"engines":rows,"fidelity":{"TREND_FOLLOW":"source-logic port; initial adapter omits 60-bar time exit and dynamic 0.25% risk sizing","EA31337_LIBRE":"screening proxy only: repo default STRAT_MA H1+H4 confirmed, exact dependency strategy parity pending","GRAND_AI_FLASH_LIMIT_NINE":"existing clean-room reconstruction; vendor private logic unavailable"},"engine_mode":"NautilusTrader runtime + Raw BidAsk deterministic adapter; native BacktestEngine parity gate pending"}
 (out/"summary.json").write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=="__main__":main()

# workflow-trigger: v1
