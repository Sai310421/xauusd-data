import json, math
from pathlib import Path
import numpy as np, pandas as pd

OUT=Path('bt_results/b_rsi_mtf_edge_lab_v1'); OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-02-25'); END=pd.Timestamp('2026-05-26 23:59:59'); INIT=1000.0
TF_ORDER=['M5','M15','H1','H4','D1']
LEVELS={
'M5': {'buy':[30.3,34.8,36.8,20.9], 'sell':[67.8,75.7,78.1,84.9]},
'M15':{'buy':[10.7,14.2,21.5,23.4,34.5,36.05], 'sell':[63.08,67.2,70.08,77.3,82.8,84.9]},
'H1': {'buy':[21.6,24.7,27.04,30.5,33.6,35.1], 'sell':[64.1,70.6,75.3,77.6,80.5,87.2]},
'H4': {'buy':[14.7,15.2,29.5,32.7,35.3], 'sell':[65.1,68.0,70.2,73.1,74.0,75.7]},
'D1': {'buy':[18.02,22.8,30.1,34.4,35.3], 'sell':[62.7,66.3,68.1,70.8,76.5,78.6]},
}
MAXBARS={'M5':160,'M15':80,'H1':40,'H4':20,'D1':10}

def rsi(s,n=14):
 d=s.diff(); u=d.clip(lower=0); v=(-d).clip(lower=0)
 au=u.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); av=v.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
 rs=au/av.replace(0,np.nan); return 100-100/(1+rs)

def atr(d,n=14):
 pc=d.close.shift(1); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def load():
 book=json.loads(Path('duka/books/xauusd_mtf.json').read_text())['tfs']; out={}
 for tf in TF_ORDER:
  q=pd.DataFrame(book[tf]).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close'})
  q['datetime']=pd.to_datetime(q.ts,unit='s',utc=True).dt.tz_convert(None)
  q=q[(q.datetime>=START)&(q.datetime<=END)].sort_values('datetime').reset_index(drop=True)
  for n in (9,14,21): q[f'rsi{n}']=rsi(q.close,n)
  q['ema9']=q.close.ewm(span=9,adjust=False).mean(); q['atr14']=atr(q)
  q['atrp']=q.atr14.rolling(100,min_periods=30).rank(pct=True)
  q['rsi_slope']=q.rsi14-q.rsi14.shift(1); q['rsi_vel']=q.rsi_slope-q.rsi_slope.shift(1)
  q['rsi_pct']=q.rsi14.rolling(100,min_periods=30).rank(pct=True)
  out[tf]=q
 return out

def pivots(x,w=3):
 a=x.to_numpy(); hi=np.zeros(len(a),bool); lo=np.zeros(len(a),bool)
 for i in range(w,len(a)-w):
  hi[i]=a[i]>=np.nanmax(a[i-w:i+w+1]); lo[i]=a[i]<=np.nanmin(a[i-w:i+w+1])
 return hi,lo

def divergence_flags(q):
 ph,pl=pivots(q.close,3); rh,rl=pivots(q.rsi14,3); bull=np.zeros(len(q),bool); bear=np.zeros(len(q),bool)
 last_pl=None; last_ph=None
 for i in range(len(q)):
  if pl[i] and rl[i]:
   if last_pl is not None and q.close.iloc[i]<q.close.iloc[last_pl] and q.rsi14.iloc[i]>q.rsi14.iloc[last_pl]: bull[i]=True
   last_pl=i
  if ph[i] and rh[i]:
   if last_ph is not None and q.close.iloc[i]>q.close.iloc[last_ph] and q.rsi14.iloc[i]<q.rsi14.iloc[last_ph]: bear[i]=True
   last_ph=i
 return bull,bear

def failure_swing(q):
 r=q.rsi14.to_numpy(); bull=np.zeros(len(q),bool); bear=np.zeros(len(q),bool)
 for i in range(4,len(q)):
  # bullish: oversold, bounce, higher low above oversold, break bounce high
  a=r[i-4:i+1]
  bull[i]=(np.nanmin(a[:2])<30 and a[2]>a[1] and a[3]>30 and a[3]>a[1] and a[4]>a[2])
  bear[i]=(np.nanmax(a[:2])>70 and a[2]<a[1] and a[3]<70 and a[3]<a[1] and a[4]<a[2])
 return bull,bear

def adaptive_rsi(q):
 # high vol -> faster RSI9, low vol -> slower RSI21, middle -> RSI14
 return np.where(q.atrp>=.70,q.rsi9,np.where(q.atrp<=.30,q.rsi21,q.rsi14))

def regime_levels(q,tf):
 # expand thresholds in high vol; contract in low vol around TF source extrema
 b=max(LEVELS[tf]['buy']); s=min(LEVELS[tf]['sell']); p=q.atrp.fillna(.5)
 buy=np.where(p>=.70,b-3,np.where(p<=.30,b+2,b)); sell=np.where(p>=.70,s+3,np.where(p<=.30,s-2,s))
 return buy,sell

def align_mtf(data,tf):
 idx=TF_ORDER.index(tf); q=data[tf][['datetime','rsi14']].copy().rename(columns={'rsi14':'selfr'})
 for ht in TF_ORDER[idx+1:]:
  h=data[ht][['datetime','rsi14']].rename(columns={'rsi14':f'r_{ht}'})
  q=pd.merge_asof(q.sort_values('datetime'),h.sort_values('datetime'),on='datetime',direction='backward')
 return q

def signals(data,tf,edge):
 q=data[tf].copy(); b=max(LEVELS[tf]['buy']); s=min(LEVELS[tf]['sell']); buy=np.zeros(len(q),bool); sell=np.zeros(len(q),bool)
 if edge=='FIXED14': buy=q.rsi14<=b; sell=q.rsi14>=s
 elif edge=='ADAPTIVE_RSI':
  ar=adaptive_rsi(q); buy=ar<=b; sell=ar>=s
 elif edge=='REGIME_THRESH':
  bl,sl=regime_levels(q,tf); buy=q.rsi14<=bl; sell=q.rsi14>=sl
 elif edge=='SLOPE_VELOCITY':
  buy=(q.rsi14<=b)&(q.rsi_slope>0)&(q.rsi_vel>0); sell=(q.rsi14>=s)&(q.rsi_slope<0)&(q.rsi_vel<0)
 elif edge=='DIVERGENCE': buy,sell=divergence_flags(q)
 elif edge=='FAILURE_SWING': buy,sell=failure_swing(q)
 elif edge=='PERCENTILE_NORM': buy=q.rsi_pct<=.10; sell=q.rsi_pct>=.90
 elif edge=='MTF_ALIGN':
  a=align_mtf(data,tf); samebuy=np.ones(len(q),bool); samesell=np.ones(len(q),bool)
  cols=[c for c in a.columns if c.startswith('r_')]
  if cols:
   # higher TF alignment is directional, not extreme: below/above 50
   samebuy=(a[cols].lt(50).sum(axis=1)>=max(1,math.ceil(len(cols)/2))).to_numpy()
   samesell=(a[cols].gt(50).sum(axis=1)>=max(1,math.ceil(len(cols)/2))).to_numpy()
  buy=(q.rsi14<=b).to_numpy()&samebuy; sell=(q.rsi14>=s).to_numpy()&samesell
 return np.asarray(buy,bool),np.asarray(sell,bool)

def bt(data,tf,edge):
 q=data[tf]; buy,sell=signals(data,tf,edge); pnl=[]; hold=[]; blocked=-1
 for i in range(25,len(q)-2):
  if i<=blocked: continue
  dr=1 if buy[i] else (-1 if sell[i] else 0)
  if dr==0 or (buy[i] and sell[i]): continue
  ei=i+1; en=float(q.open.iloc[ei]); ex=en; xi=min(len(q)-1,ei+MAXBARS[tf])
  for j in range(ei,xi+1):
   ema=float(q.ema9.iloc[j]); lo=float(q.low.iloc[j]); hi=float(q.high.iloc[j])
   if lo<=ema<=hi: ex=ema; xi=j; break
  if xi>=len(q): xi=len(q)-1
  if ex==en: ex=float(q.close.iloc[xi])
  ret=dr*(ex-en)/max(abs(en),1e-12)
  pnl.append(ret); hold.append(xi-ei+1); blocked=xi
 a=np.array(pnl,float); N=len(a); gp=a[a>0].sum(); gl=-a[a<0].sum(); pf=gp/gl if gl>0 else (9999 if gp>0 else 0); wr=(a>0).mean()*100 if N else 0
 # normalized 0.25%-risk-like score to compare edges, not broker PnL
 eq=INIT; peak=INIT; mdd=0
 for r in a:
  x=np.clip(r*100, -0.025, .05) # common bounded transfer function
  eq*=1+x; peak=max(peak,eq); mdd=max(mdd,100*(peak-eq)/peak)
 biz=max(1,len(pd.bdate_range(START.date(),END.date()))); month=((eq/INIT)**(21/biz)-1)*100
 return {'TF':tf,'EDGE':edge,'N':N,'WR':wr,'PF':pf,'MeanRawRet':a.mean() if N else 0,'NetScore':eq-INIT,'Month21Score':month,'MaxDDScore':mdd,'AvgHoldBars':np.mean(hold) if hold else 0}

data=load(); edges=['FIXED14','ADAPTIVE_RSI','REGIME_THRESH','SLOPE_VELOCITY','DIVERGENCE','MTF_ALIGN','FAILURE_SWING','PERCENTILE_NORM']
rows=[]
for tf in TF_ORDER:
 for e in edges: rows.append(bt(data,tf,e))
r=pd.DataFrame(rows); r.to_csv(OUT/'edge_matrix.csv',index=False)
# robust ranking: demand N>=5, reward PF/WR/return, penalize DD; this is diagnostic only
z=r.copy(); z['Score']=np.where(z.N>=5, np.log1p(z.PF.clip(0,20))*2 + z.WR/50 + z.Month21Score/20 - z.MaxDDScore/10, -999)
z=z.sort_values('Score',ascending=False); z.to_csv(OUT/'ranked.csv',index=False)
# aggregate each edge across TFs by trade-count weighted diagnostics
agg=[]
for e,g in r.groupby('EDGE'):
 n=g.N.sum(); w=np.where(g.N>0,g.N,0); agg.append({'EDGE':e,'N':int(n),'WR_w':np.average(g.WR,weights=w) if n else 0,'PF_median':g.PF.replace(9999,np.nan).median(),'Month21Score_sum':g.Month21Score.sum(),'MaxDDScore_max':g.MaxDDScore.max()})
pd.DataFrame(agg).sort_values('Month21Score_sum',ascending=False).to_csv(OUT/'edge_aggregate.csv',index=False)
meta={'verification':'B_RSI_MTF_EDGE_DIAGNOSTIC_PROXY','data':'stored M5/M15/H1/H4/D1 from xauusd-duka-feed; no resampling','exit':'common EMA9 touch / TF max-bars','g75':False,'costs':'none','purpose':'isolate direction/entry EDGE before G75/Math-DD integration'}
(OUT/'meta.json').write_text(json.dumps(meta,indent=2)); print(z.head(20).to_string(index=False))