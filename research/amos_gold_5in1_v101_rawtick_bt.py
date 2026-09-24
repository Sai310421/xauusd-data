from __future__ import annotations
import argparse, concurrent.futures as cf, datetime as dt, json, lzma, struct, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
REC=struct.Struct(">3i2f"); HOSTS=("https://datafeed.dukascopy.com/datafeed","https://www.dukascopy.com/datafeed")
OUT=Path("results/amos_gold_5in1_v101")
def days(start,n):
 d=dt.date.fromisoformat(start); out=[]
 while len(out)<n:
  if d.weekday()<5: out.append(d)
  d+=dt.timedelta(days=1)
 return out
def fh(z):
 d,h=z; o=dt.datetime(d.year,d.month,d.day,h,tzinfo=dt.timezone.utc); rel=f"XAUUSD/{d.year}/{d.month-1:02d}/{d.day:02d}/{h:02d}h_ticks.bi5"
 for host in HOSTS:
  try:
   req=urllib.request.Request(f"{host}/{rel}",headers={"User-Agent":"amos-5in1/1.01"})
   raw=urllib.request.urlopen(req,timeout=20).read(); dec=lzma.decompress(raw); a=[]
   for i in range(0,len(dec)-REC.size+1,REC.size):
    ms,ask,bid,av,bv=REC.unpack_from(dec,i); ask/=1000; bid/=1000
    if ask>=bid>0:a.append((o+dt.timedelta(milliseconds=ms),bid,ask))
   return a
  except Exception: pass
 return []
def load(ds,w):
 rows=[]
 with cf.ThreadPoolExecutor(max_workers=w) as ex:
  for x in ex.map(fh,[(d,h) for d in ds for h in range(24)]): rows.extend(x)
 if not rows: raise SystemExit("NO_RAW_TICKS")
 df=pd.DataFrame(rows,columns=["t","bid","ask"]).sort_values("t").drop_duplicates("t"); df.t=pd.to_datetime(df.t,utc=True); return df
def bars(t,freq):
 x=t.copy(); x["mid"]=(x["bid"]+x["ask"])/2; x["b"]=x["t"].dt.floor(freq)
 return x.groupby("b").agg(open=("mid","first"),high=("mid","max"),low=("mid","min"),close=("mid","last"),bid=("bid","last"),ask=("ask","last")).reset_index().rename(columns={"b":"t"})
def atr(x,n=14):
 cl=x["close"]; hi=x["high"]; lo=x["low"]; p=cl.shift(1); tr=pd.concat([hi-lo,(hi-p).abs(),(lo-p).abs()],axis=1).max(axis=1); return tr.rolling(n).mean()
def rsi(s,n=14):
 d=s.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean(); return 100-100/(1+up/dn.replace(0,np.nan))
def prep(x):
 x=x.copy(); cl=pd.to_numeric(x["close"],errors="coerce"); x["close"]=cl; x["open"]=pd.to_numeric(x["open"],errors="coerce"); x["high"]=pd.to_numeric(x["high"],errors="coerce"); x["low"]=pd.to_numeric(x["low"],errors="coerce"); x["atr"]=atr(x); x["ema21"]=cl.ewm(span=21,adjust=False).mean(); x["ema55"]=cl.ewm(span=55,adjust=False).mean(); x["rsi"]=rsi(cl); return x
def signals(m5,m15,h1,d1):
 sig=[]
 # Flash: M15 128-bar breakout, 12h gap
 hi=m15.high.shift(2).rolling(128).max(); lo=m15.low.shift(2).rolling(128).min()
 for i in range(130,len(m15)):
  c=m15.close.iloc[i-1]; tm=m15.t.iloc[i]
  if c>hi.iloc[i]: sig.append((tm,"FLASH",1,.85,3.16))
  elif c<lo.iloc[i]: sig.append((tm,"FLASH",-1,.85,3.16))
 # AI proxy M5
 for i in range(60,len(m5)):
  r=m5.iloc[i-1]; a=h1[h1.t<m5.t.iloc[i]].atr.tail(1)
  if a.empty or pd.isna(a.iloc[0]): continue
  av=float(a.iloc[0])
  if r.ema21>r.ema55 and 55<r.rsi<72:sig.append((m5.t.iloc[i],"AI",1,av,2*av))
  elif r.ema21<r.ema55 and 28<r.rsi<45:sig.append((m5.t.iloc[i],"AI",-1,av,2*av))
 # Limit M5 240 breakout with D1 ATR
 hi=m5.high.shift(2).rolling(240).max(); lo=m5.low.shift(2).rolling(240).min()
 for i in range(242,len(m5)):
  c=m5.close.iloc[i-1]; a=d1[d1.t<m5.t.iloc[i]].atr.tail(1)
  if a.empty or pd.isna(a.iloc[0]):continue
  v=.30*float(a.iloc[0])
  if c>hi.iloc[i]:sig.append((m5.t.iloc[i],"LIMIT",1,v,v))
  elif c<lo.iloc[i]:sig.append((m5.t.iloc[i],"LIMIT",-1,v,v))
 # Grand + Nine H1
 for i in range(55,len(h1)):
  r=h1.iloc[i-1]; a=float(r.atr) if pd.notna(r.atr) else np.nan
  if not np.isfinite(a):continue
  hist=h1.iloc[max(0,i-50):i-1]
  hi=hist.high.tail(48).max(); lo=hist.low.tail(48).min(); mid=(hi+lo)/2; tm=h1.t.iloc[i]
  if r.close<mid and r.close>lo+.25*a:sig.append((tm,"GRAND",1,a,2.2*a))
  elif r.close>mid and r.close<hi-.25*a:sig.append((tm,"GRAND",-1,a,2.2*a))
  for k,w in enumerate((20,24,28,32,36,42,48)):
   z=hist.tail(w); H=z.high.max(); L=z.low.min(); off=a*(.08+.02*k); sl=a*(.7+.05*k); tp=a*(1.4+.12*k)
   if r.close>H-off:sig.append((tm,f"NINE{k+1}",1,sl,tp))
   elif r.close<L+off:sig.append((tm,f"NINE{k+1}",-1,sl,tp))
 return sorted(sig,key=lambda z:z[0])
def simulate(t,sigs,capital=1000,lot=.01):
 # XAUUSD 1 lot ~= $100 per $1 move; 0.01 lot ~= $1 per $1 move.
 mult=100*lot; openp=[]; trades=[]; eq=capital; peak=capital; maxdd=0; last_flash=None
 si=0
 for row in t.itertuples():
  tm=row.t; bid=row.bid; ask=row.ask
  keep=[]
  for p in openp:
   px=bid if p["side"]==1 else ask
   hit_sl=px<=p["sl"] if p["side"]==1 else px>=p["sl"]; hit_tp=px>=p["tp"] if p["side"]==1 else px<=p["tp"]
   if hit_sl or hit_tp:
    pnl=(px-p["entry"])*p["side"]*mult; eq+=pnl; trades.append((tm,p["eng"],p["side"],p["entry"],px,pnl))
   else: keep.append(p)
  openp=keep
  while si<len(sigs) and sigs[si][0]<=tm:
   st,eng,side,sd,td=sigs[si]; si+=1
   if eng=="FLASH" and last_flash is not None and (st-last_flash).total_seconds()<43200: continue
   if len(openp)>=18 or ask-bid>.50: continue
   entry=ask if side==1 else bid; openp.append({"eng":eng,"side":side,"entry":entry,"sl":entry-side*sd,"tp":entry+side*td})
   if eng=="FLASH":last_flash=st
  floating=sum(((bid if p["side"]==1 else ask)-p["entry"])*p["side"]*mult for p in openp); cur=eq+floating; peak=max(peak,cur); maxdd=max(maxdd,(peak-cur)/peak if peak else 0)
 if not trades:return pd.DataFrame(),{"N":0}
 tr=pd.DataFrame(trades,columns=["exit_time","engine","side","entry","exit","pnl"]); wins=tr[tr.pnl>0].pnl.sum(); losses=-tr[tr.pnl<0].pnl.sum()
 s={"N":len(tr),"WR_pct":float((tr.pnl>0).mean()*100),"PF":float(wins/losses) if losses>0 else None,"EV_USD":float(tr.pnl.mean()),"Net_USD":float(tr.pnl.sum()),"Return_pct":float(tr.pnl.sum()/capital*100),"MaxDD_pct":float(maxdd*100)}
 return tr,s
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--start",default="2026-07-27");ap.add_argument("--days",type=int,default=21);ap.add_argument("--workers",type=int,default=48);a=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True); t=load(days(a.start,a.days),a.workers)
 m5=prep(bars(t,"5min"));m15=prep(bars(t,"15min"));h1=prep(bars(t,"1h"));d1=prep(bars(t,"1D"))
 sig=signals(m5,m15,h1,d1); tr,s=simulate(t,sig); tr.to_csv(OUT/"trades.csv",index=False)
 by={}
 if len(tr):
  for e,g in tr.groupby("engine"):
   gp=g[g.pnl>0].pnl.sum();gl=-g[g.pnl<0].pnl.sum();by[e]={"N":len(g),"WR_pct":float((g.pnl>0).mean()*100),"PF":float(gp/gl) if gl>0 else None,"Net_USD":float(g.pnl.sum())}
 out={"verification":"RAW_BIDASK_DISCOVERY","source":"Dukascopy BI5","raw_ticks":len(t),"bar_inputs":"derived directly from raw bid/ask ticks; no external OHLC fallback","period":{"start":a.start,"business_days":a.days},**s,"engine_breakdown":by,"limitations":["clean-room reconstruction; vendor private logic not available","not MT5 native fill parity","AI engine is proxy logic"]}
 (OUT/"summary.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__":main()

# workflow-trigger: v1.01
