import json, runpy
from pathlib import Path
import numpy as np, pandas as pd

OUT=Path('bt_results/abc_b_mtf_composite_v2'); OUT.mkdir(parents=True,exist_ok=True)
INIT=1000.0; START=pd.Timestamp('2026-02-25'); END=pd.Timestamp('2026-05-26 23:59:59')

# Reuse already-validated reconstructed A/C generators and shared Math-DD portfolio.
ab=runpy.run_path('scripts/abc_math_dd_controller_v1.py')
A_EVENTS=ab['a_events']; C_EVENTS=ab['c_events']; BASE_B=ab['b_events']; PORT=ab['portfolio_math']

# Shared controller = chosen v1 profile that produced ~239.95% Month21 / ~4.50% DD in proxy.
P=dict(d1=3.0,d2=3.5,d3=4.25,d4=4.5,m1=.95,m2=.80,m3=.60,m4=.40,
       q1=2.5,q2=3.0,q3=3.5,q_m1=.90,q_m2=.70,q_m3=.40,
       boost_stop_dd=3.5,boost_stop_daily=3.0,hard_dd=4.5,hard_daily=3.5,floor=.05)

def rsi(s,n=14):
 d=s.diff(); u=d.clip(lower=0); v=(-d).clip(lower=0)
 au=u.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); av=v.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
 rs=au/av.replace(0,np.nan); return 100-100/(1+rs)

def atr(d,n=14):
 pc=d.close.shift(1); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def load_mtf():
 book=json.loads(Path('duka/books/xauusd_mtf.json').read_text())['tfs']; out={}
 for tf in ['M5','M15']:
  q=pd.DataFrame(book[tf]).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close'})
  q['datetime']=pd.to_datetime(q.ts,unit='s',utc=True).dt.tz_convert(None)
  q=q[(q.datetime>=START)&(q.datetime<=END)].sort_values('datetime').reset_index(drop=True)
  for n in (9,14,21): q[f'rsi{n}']=rsi(q.close,n)
  q['ema9']=q.close.ewm(span=9,adjust=False).mean(); q['atr14']=atr(q)
  q['atrp']=q.atr14.rolling(100,min_periods=30).rank(pct=True)
  q['rsi_pct']=q.rsi14.rolling(100,min_periods=30).rank(pct=True)
  q['rsi_slope']=q.rsi14-q.rsi14.shift(1)
  out[tf]=q
 return out

def m5_edge_events(data):
 # M5 independent composite = percentile OR adaptive RSI. Independent entry N retained; duplicate same-bar direction deduped.
 q=data['M5']; ar=np.where(q.atrp>=.70,q.rsi9,np.where(q.atrp<=.30,q.rsi21,q.rsi14))
 buy=((q.rsi_pct<=.10)|(ar<=36.8)); sell=((q.rsi_pct>=.90)|(ar>=67.8))
 lot=.001; ev=[]; blocked=-1
 for i in range(30,len(q)-2):
  if i<=blocked: continue
  dr=1 if buy.iloc[i] else (-1 if sell.iloc[i] else 0)
  if dr==0 or (bool(buy.iloc[i]) and bool(sell.iloc[i])): continue
  ei=i+1; en=float(q.open.iloc[ei]); xi=min(len(q)-1,ei+160); ex=float(q.close.iloc[xi]); mae=0.0
  for j in range(ei,xi+1):
   lo=float(q.low.iloc[j]); hi=float(q.high.iloc[j]); ema=float(q.ema9.iloc[j]); adv=lo if dr>0 else hi
   mtm=((adv-en) if dr>0 else (en-adv))*lot*100; mae=min(mae,mtm)
   if lo<=ema<=hi: ex=ema; xi=j; break
  pnl=((ex-en) if dr>0 else (en-ex))*lot*100
  ev.append(dict(strategy='B_M5',symbol='XAUUSD',direction=dr,entry_time=q.datetime.iloc[ei],exit_time=q.datetime.iloc[xi],pnl_frac=pnl/INIT,mae_frac=mae/INIT))
  blocked=xi
 return ev

def centered_pivots(x,w=3):
 a=x.to_numpy(); hi=np.zeros(len(a),bool); lo=np.zeros(len(a),bool)
 for i in range(w,len(a)-w):
  if not np.isfinite(a[i]): continue
  hi[i]=a[i]>=np.nanmax(a[i-w:i+w+1]); lo[i]=a[i]<=np.nanmin(a[i-w:i+w+1])
 return hi,lo

def causal_div(q,w=3,sep=8,delay=2):
 ph,pl=centered_pivots(q.close,w); rh,rl=centered_pivots(q.rsi14,w)
 bull=np.zeros(len(q),bool); bear=np.zeros(len(q),bool); lp=None; lh=None
 for c in range(len(q)):
  if pl[c] and rl[c]:
   if lp is not None and c-lp>=sep and q.close.iloc[c]<q.close.iloc[lp] and q.rsi14.iloc[c]>q.rsi14.iloc[lp]:
    f=c+w+delay
    if f<len(q): bull[f]=True
   lp=c
  if ph[c] and rh[c]:
   if lh is not None and c-lh>=sep and q.close.iloc[c]>q.close.iloc[lh] and q.rsi14.iloc[c]<q.rsi14.iloc[lh]:
    f=c+w+delay
    if f<len(q): bear[f]=True
   lh=c
 return bull,bear

def child_confirm(data,parent):
 m=data['M5'].copy(); ar=np.where(m.atrp>=.70,m.rsi9,np.where(m.atrp<=.30,m.rsi21,m.rsi14))
 b=(((m.rsi_pct<=.10)|(ar<=36.8))&(m.rsi_slope>0)).astype(bool)
 s=(((m.rsi_pct>=.90)|(ar>=67.8))&(m.rsi_slope<0)).astype(bool)
 x=m[['datetime']].copy(); x['b']=pd.Series(b).rolling(3,min_periods=1).max().astype(bool); x['s']=pd.Series(s).rolling(3,min_periods=1).max().astype(bool)
 z=pd.merge_asof(parent[['datetime']],x[['datetime','b','s']],on='datetime',direction='backward')
 return z.b.fillna(False).to_numpy(bool),z.s.fillna(False).to_numpy(bool)

def divergence_lt_events(data):
 # Audited HC profile: M15 causal divergence w3/sep8/delay2 -> LT-B 0.5ATR peak trail; G75 add=.25ATR max5; M5 child timing.
 q=data['M15']; bull,bear=causal_div(q); cb,cs=child_confirm(data,q)
 lot=.001; trig=.35; add=.25; maxadds=5; trail=.50; ev=[]; blocked=-1
 for i in range(40,len(q)-2):
  if i<=blocked: continue
  dr=1 if bull[i] else (-1 if bear[i] else 0)
  if dr==0 or (bull[i] and bear[i]): continue
  ei=i+1; en=float(q.open.iloc[ei]); av=float(q.atr14.iloc[i]) if np.isfinite(q.atr14.iloc[i]) else 0
  if av<=0: continue
  entries=[(en,lot)]; last=en; peak=en; trough=en; mae=0.0; xi=min(len(q)-1,ei+160); ex=float(q.close.iloc[xi]); activated=False
  for j in range(ei,xi+1):
   hi=float(q.high.iloc[j]); lo=float(q.low.iloc[j]); cl=float(q.close.iloc[j]); peak=max(peak,hi); trough=min(trough,lo)
   adv=lo if dr>0 else hi; mtm=sum(((adv-px) if dr>0 else (px-adv))*v*100 for px,v in entries); mae=min(mae,mtm)
   fav=(peak-en) if dr>0 else (en-trough)
   if fav>=trig*av: activated=True
   if activated:
    px=peak if dr>0 else trough
    while len(entries)-1<maxadds and dr*(px-last)>=add*av:
     ok=cb[j] if dr>0 else cs[j]
     if not ok: break
     last+=dr*add*av; entries.append((last,lot))
    if dr>0 and peak-lo>=trail*av: ex=max(lo,peak-trail*av); xi=j; break
    if dr<0 and hi-trough>=trail*av: ex=min(hi,trough+trail*av); xi=j; break
  pnl=sum(((ex-px) if dr>0 else (px-ex))*v*100 for px,v in entries)
  ev.append(dict(strategy='B_DIV_LT',symbol='XAUUSD',direction=dr,entry_time=q.datetime.iloc[ei],exit_time=q.datetime.iloc[xi],pnl_frac=pnl/INIT,mae_frac=mae/INIT,adds=len(entries)-1))
  blocked=xi
 return ev

def summary(mode,events):
 s,rows=PORT(events,P)
 s['MODE']=mode
 s['B_M5_N']=sum(1 for e in events if e['strategy']=='B_M5')
 s['B_DIV_LT_N']=sum(1 for e in events if e['strategy']=='B_DIV_LT')
 return s,rows

data=load_mtf(); A=A_EVENTS(); CX=C_EVENTS('XAUUSD'); CE=C_EVENTS('EURUSD'); B0=BASE_B(); BM5=m5_edge_events(data); BDLT=divergence_lt_events(data)
variants={
 'BASE_ABC_MATHDD': A+B0+CX+CE,
 'ADD_DIV_LT_ONLY': A+B0+BDLT+CX+CE,
 'REPLACE_B_WITH_MTFV2': A+BM5+BDLT+CX+CE,
 'FULL_KEEP_B_PLUS_MTFV2': A+B0+BM5+BDLT+CX+CE,
}
outs=[]
for name,ev in variants.items():
 s,rows=summary(name,ev); outs.append(s)
 pd.DataFrame(rows).to_csv(OUT/f'trades_{name}.csv',index=False)
pd.DataFrame(outs).to_csv(OUT/'comparison.csv',index=False)
(Path(OUT/'meta.json')).write_text(json.dumps({'verification':'ABC_B_MTF_COMPOSITE_V2_PROXY','native_tf':'M5/M15 stored; C native M1 parquet; no resampling','divergence':'causal w3 sep8 delay2','div_lt':'M15 LT-B 0.5ATR trail, G75 .35 trigger/.25 add/max5, M5 child confirm','m5':'percentile OR adaptive RSI -> EMA9','shared':'same Math-DD + overlap Boost as v1','costs':'none in integration pass','warning':'reconstructed proxy; shared chronology/MAE model remains approximate; not MT5 real-tick'},indent=2))
print(pd.DataFrame(outs).to_string(index=False))
