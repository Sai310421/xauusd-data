import json, math
from pathlib import Path
import numpy as np, pandas as pd

OUT=Path('bt_results/divergence_g75_longtime_v1'); OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-02-25'); END=pd.Timestamp('2026-05-26 23:59:59'); INIT=1000.0
PARENTS=['M15','H1']; MAXBARS={'M15':160,'H1':96}

# Causal divergence specification carried from audit v2
DIV_W=3; DIV_SEP=8; DIV_DELAY=2

# G75 Long-Time research grid
TRIGGER_ATR=.35
ADD_ATRS=[.15,.20,.25]
MAX_ADDS=[5,10]
TRAIL_ATRS=[.50,.75,1.00]


def rsi(s,n=14):
 d=s.diff(); u=d.clip(lower=0); v=(-d).clip(lower=0)
 au=u.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); av=v.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
 rs=au/av.replace(0,np.nan); return 100-100/(1+rs)

def atr(d,n=14):
 pc=d.close.shift(1); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def load():
 book=json.loads(Path('duka/books/xauusd_mtf.json').read_text())['tfs']; out={}
 for tf in ['M5','M15','H1']:
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

def centered_pivots(x,w):
 a=x.to_numpy(); hi=np.zeros(len(a),bool); lo=np.zeros(len(a),bool)
 for i in range(w,len(a)-w):
  hi[i]=np.isfinite(a[i]) and a[i]>=np.nanmax(a[i-w:i+w+1])
  lo[i]=np.isfinite(a[i]) and a[i]<=np.nanmin(a[i-w:i+w+1])
 return hi,lo

def causal_divergence(q,w=DIV_W,minsep=DIV_SEP,delay=DIV_DELAY):
 # Pivots become knowable only w bars after the center; signal adds requested delay.
 ph,pl=centered_pivots(q.close,w); rh,rl=centered_pivots(q.rsi14,w)
 bull=np.zeros(len(q),bool); bear=np.zeros(len(q),bool); last_pl=None; last_ph=None
 for center in range(len(q)):
  if pl[center] and rl[center]:
   if last_pl is not None and center-last_pl>=minsep and q.close.iloc[center]<q.close.iloc[last_pl] and q.rsi14.iloc[center]>q.rsi14.iloc[last_pl]:
    fire=center+w+delay
    if fire<len(q): bull[fire]=True
   last_pl=center
  if ph[center] and rh[center]:
   if last_ph is not None and center-last_ph>=minsep and q.close.iloc[center]>q.close.iloc[last_ph] and q.rsi14.iloc[center]<q.rsi14.iloc[last_ph]:
    fire=center+w+delay
    if fire<len(q): bear[fire]=True
   last_ph=center
 return bull,bear

def m5_child_flags(m5):
 # Child timing remains a separate EDGE. Require the original-style buy/sell extreme AND recovering slope.
 ar=np.where(m5.atrp>=.70,m5.rsi9,np.where(m5.atrp<=.30,m5.rsi21,m5.rsi14))
 buy=((m5.rsi_pct<=.10)|(ar<=36.8)) & (m5.rsi_slope>0)
 sell=((m5.rsi_pct>=.90)|(ar>=67.8)) & (m5.rsi_slope<0)
 return np.asarray(buy,bool),np.asarray(sell,bool)

def m5_confirm_map(parent,m5):
 mb,ms=m5_child_flags(m5)
 x=m5[['datetime']].copy(); x['mb']=mb; x['ms']=ms
 # map latest M5 state into parent timestamps, also allow last 2 M5 bars via rolling max
 x['mb2']=pd.Series(mb).rolling(3,min_periods=1).max().astype(bool)
 x['ms2']=pd.Series(ms).rolling(3,min_periods=1).max().astype(bool)
 z=pd.merge_asof(parent[['datetime']].sort_values('datetime'),x[['datetime','mb2','ms2']].sort_values('datetime'),on='datetime',direction='backward')
 return z.mb2.fillna(False).to_numpy(bool),z.ms2.fillna(False).to_numpy(bool)

def opposite_div_flags(q):
 return causal_divergence(q)

def choose_exit(q,ei,dir,mode,trail_k,opp_bull,opp_bear):
 en=float(q.open.iloc[ei]); av=float(q.atr14.iloc[ei-1]) if ei>0 else float(q.atr14.iloc[ei]);
 if not np.isfinite(av) or av<=0: return None
 end=min(len(q)-1,ei+MAXBARS[q.attrs['tf']])
 peak=en; trough=en; partial_i=None; partial_px=None
 for j in range(ei,end+1):
  hi=float(q.high.iloc[j]); lo=float(q.low.iloc[j]); close=float(q.close.iloc[j]); ema=float(q.ema9.iloc[j])
  peak=max(peak,hi); trough=min(trough,lo)
  if mode=='EMA9':
   if lo<=ema<=hi: return {'xi':j,'px':ema,'partial':None,'atr':av}
  elif mode=='LT_A':
   if partial_i is None and lo<=ema<=hi:
    partial_i=j; partial_px=ema
   if partial_i is not None:
    if dir>0 and peak-lo>=trail_k*av: return {'xi':j,'px':max(lo,peak-trail_k*av),'partial':(partial_i,partial_px,.50),'atr':av}
    if dir<0 and hi-trough>=trail_k*av: return {'xi':j,'px':min(hi,trough+trail_k*av),'partial':(partial_i,partial_px,.50),'atr':av}
  elif mode=='LT_B':
   fav=(peak-en if dir>0 else en-trough)
   if fav>=TRIGGER_ATR*av:
    if dir>0 and peak-lo>=trail_k*av: return {'xi':j,'px':max(lo,peak-trail_k*av),'partial':None,'atr':av}
    if dir<0 and hi-trough>=trail_k*av: return {'xi':j,'px':min(hi,trough+trail_k*av),'partial':None,'atr':av}
  elif mode=='LT_C':
   # opposite confirmed divergence OR 5-bar structure break, whichever arrives first
   if (dir>0 and opp_bear[j]) or (dir<0 and opp_bull[j]): return {'xi':j,'px':close,'partial':None,'atr':av}
   if j>=ei+5:
    if dir>0 and close<float(q.low.iloc[j-5:j].min()): return {'xi':j,'px':close,'partial':None,'atr':av}
    if dir<0 and close>float(q.high.iloc[j-5:j].max()): return {'xi':j,'px':close,'partial':None,'atr':av}
 # timeout
 return {'xi':end,'px':float(q.close.iloc[end]),'partial':((partial_i,partial_px,.50) if partial_i is not None else None),'atr':av}

def trade_return(q,ei,dir,exit_info,add_atr,max_adds,child_confirm,child_buy,child_sell):
 en=float(q.open.iloc[ei]); av=exit_info['atr']; xi=exit_info['xi']; xp=exit_info['px']
 entries=[en]; last_add=en
 # profit-direction only; optional M5 child confirmation is required when child_confirm=True
 for j in range(ei,xi+1):
  hi=float(q.high.iloc[j]); lo=float(q.low.iloc[j])
  favorable=(hi-en if dir>0 else en-lo)
  if favorable<TRIGGER_ATR*av: continue
  px=(hi if dir>0 else lo)
  while len(entries)-1<max_adds and dir*(px-last_add)>=add_atr*av:
   ok=True
   if child_confirm: ok=(child_buy[j] if dir>0 else child_sell[j])
   if not ok: break
   last_add=last_add+dir*add_atr*av; entries.append(last_add)
 # equal nominal per layer, matching G75 research convention
 if exit_info['partial'] is None:
  rs=[dir*(xp-e)/max(abs(e),1e-12) for e in entries]
 else:
  _,pp,frac=exit_info['partial']; rs=[]
  for e in entries:
   rs.append(frac*dir*(pp-e)/max(abs(e),1e-12)+(1-frac)*dir*(xp-e)/max(abs(e),1e-12))
 return float(np.sum(rs)),len(entries)-1

def run_one(data,tf,mode,add_atr,max_adds,trail_k,child_confirm,cost_bps=0.0):
 q=data[tf].copy(); q.attrs['tf']=tf; bull,bear=causal_divergence(q); opp_bull,opp_bear=bull,bear
 cb,cs=m5_confirm_map(q,data['M5'])
 pnl=[]; adds=[]; blocked=-1; trades=[]
 for i in range(max(40,DIV_W+DIV_DELAY+5),len(q)-2):
  if i<=blocked: continue
  dr=1 if bull[i] else (-1 if bear[i] else 0)
  if dr==0 or (bull[i] and bear[i]): continue
  ei=i+1
  ex=choose_exit(q,ei,dr,mode,trail_k,opp_bull,opp_bear)
  if ex is None: continue
  ret,na=trade_return(q,ei,dr,ex,add_atr,max_adds,child_confirm,cb,cs)
  # cost per base+add layer, round-turn approximation
  ret-=((1+na)*2*cost_bps/10000.0)
  pnl.append(ret); adds.append(na); blocked=ex['xi']
  trades.append((q.datetime.iloc[i],q.datetime.iloc[ei],q.datetime.iloc[ex['xi']],dr,ret,na))
 a=np.array(pnl,float); N=len(a); gp=a[a>0].sum(); gl=-a[a<0].sum(); pf=gp/gl if gl>0 else (9999 if gp>0 else 0); wr=(a>0).mean()*100 if N else 0
 # common bounded equity transfer; diagnostic, not broker PnL
 eq=INIT; peak=INIT;mdd=0
 for r in a:
  x=np.clip(r*35,-.05,.15)
  eq*=1+x;peak=max(peak,eq);mdd=max(mdd,100*(peak-eq)/peak)
 biz=max(1,len(pd.bdate_range(START.date(),END.date()))); month=((eq/INIT)**(21/biz)-1)*100
 cut=int(N*2/3); tr=a[:cut]; oo=a[cut:]
 def sub(x):
  if len(x)==0:return (0,0,0)
  g=x[x>0].sum();l=-x[x<0].sum();p=g/l if l>0 else (9999 if g>0 else 0);return (len(x),(x>0).mean()*100,p)
 tn,tw,tp=sub(tr);on,ow,op=sub(oo)
 return {'TF':tf,'MODE':mode,'AddATR':add_atr,'MaxAdds':max_adds,'TrailATR':trail_k,'M5ChildConfirm':child_confirm,'CostBps':cost_bps,'N':N,'WR':wr,'PF':pf,'MeanRet':a.mean() if N else 0,'AvgAdds':np.mean(adds) if adds else 0,'MaxAddsSeen':max(adds) if adds else 0,'Month21Score':month,'MaxDDScore':mdd,'TrainN':tn,'TrainWR':tw,'TrainPF':tp,'OOSN':on,'OOSWR':ow,'OOSPF':op}

data=load(); rows=[]
for tf in PARENTS:
 # baseline parent divergence -> EMA9 (no G75), plus long-time families with G75
 rows.append(run_one(data,tf,'EMA9',.20,0,.75,False,0))
 for mode in ['LT_A','LT_B','LT_C']:
  trails=TRAIL_ATRS if mode in ['LT_A','LT_B'] else [.75]
  for ad in ADD_ATRS:
   for ma in MAX_ADDS:
    for tk in trails:
     for cc in [False,True]:
      rows.append(run_one(data,tf,mode,ad,ma,tk,cc,0))

r=pd.DataFrame(rows)
# robustness gate: enough total/OOS trades, PF on both train and OOS, cap diagnostic DD
r['PASS']=((r.N>=20)&(r.OOSN>=6)&(r.TrainPF>1.0)&(r.OOSPF>1.0)&(r.PF>1.05)&(r.MaxDDScore<=20))
r['Score']=np.where(r.PASS, np.log1p(r.PF.clip(0,10))*2 + r.OOSPF.clip(0,10) + r.Month21Score/30 - r.MaxDDScore/15, -999)
r.sort_values(['PASS','Score'],ascending=[False,False]).to_csv(OUT/'grid.csv',index=False)
r[r.PASS].sort_values('Score',ascending=False).to_csv(OUT/'passed.csv',index=False)

# Cost stress on top 8 passing settings, 1 and 2 bps
stress=[]
for _,z in r[r.PASS].sort_values('Score',ascending=False).head(8).iterrows():
 for c in [1.0,2.0]:
  stress.append(run_one(data,z.TF,z.MODE,float(z.AddATR),int(z.MaxAdds),float(z.TrailATR),bool(z.M5ChildConfirm),c))
pd.DataFrame(stress).to_csv(OUT/'cost_stress.csv',index=False)

meta={'verification':'CAUSAL_DIVERGENCE_G75_LONGTIME_PROXY_V1','parent_tfs':PARENTS,'divergence':{'w':DIV_W,'min_sep':DIV_SEP,'delay':DIV_DELAY},'g75':'profit-direction adds only','child':'M5 percentile/adaptive extreme + recovery slope, optional add timing only','exits':{'EMA9':'baseline full exit','LT_A':'first EMA9 partial 50%, runner ATR trail','LT_B':'ATR peak trail','LT_C':'opposite confirmed divergence or 5-bar structure break'},'data':'stored native TFs from xauusd-duka-feed; no OHLC resampling','costs':'grid=0 bps, top settings stress=1/2 bps','warning':'diagnostic proxy, not MT5 real-tick'}
(OUT/'meta.json').write_text(json.dumps(meta,indent=2))
print(r.sort_values(['PASS','Score'],ascending=[False,False]).head(25).to_string(index=False))