import json, math
from pathlib import Path
import numpy as np, pandas as pd

OUT=Path('bt_results/b_rsi_divergence_causal_audit_v2'); OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-02-25'); END=pd.Timestamp('2026-05-26 23:59:59'); INIT=1000.0
TFS=['M5','M15','H1','H4']; MAXBARS={'M5':160,'M15':80,'H1':40,'H4':20}

def rsi(s,n=14):
 d=s.diff(); u=d.clip(lower=0); v=(-d).clip(lower=0)
 au=u.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); av=v.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
 rs=au/av.replace(0,np.nan); return 100-100/(1+rs)

def load():
 book=json.loads(Path('duka/books/xauusd_mtf.json').read_text())['tfs']; out={}
 for tf in TFS:
  q=pd.DataFrame(book[tf]).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close'})
  q['datetime']=pd.to_datetime(q.ts,unit='s',utc=True).dt.tz_convert(None)
  q=q[(q.datetime>=START)&(q.datetime<=END)].sort_values('datetime').reset_index(drop=True)
  q['rsi14']=rsi(q.close,14); q['ema9']=q.close.ewm(span=9,adjust=False).mean(); out[tf]=q
 return out

def centered_pivots(a,w):
 a=np.asarray(a,float); hi=np.zeros(len(a),bool); lo=np.zeros(len(a),bool)
 for i in range(w,len(a)-w):
  z=a[i-w:i+w+1]
  if np.isfinite(a[i]):
   hi[i]=a[i]>=np.nanmax(z); lo[i]=a[i]<=np.nanmin(z)
 return hi,lo

def causal_divergence_signals(q,w=3,min_sep=3,delay=0):
 # A pivot at index i is only knowable after right-side w bars, so the earliest signal is i+w.
 ph,pl=centered_pivots(q.close,w); rh,rl=centered_pivots(q.rsi14,w)
 buy=np.zeros(len(q),bool); sell=np.zeros(len(q),bool)
 last_pl=None; last_ph=None
 for i in range(w,len(q)-w):
  confirm=i+w+delay
  if confirm>=len(q)-1: break
  if pl[i] and rl[i]:
   if last_pl is not None and i-last_pl>=min_sep:
    if q.close.iloc[i] < q.close.iloc[last_pl] and q.rsi14.iloc[i] > q.rsi14.iloc[last_pl]: buy[confirm]=True
   last_pl=i
  if ph[i] and rh[i]:
   if last_ph is not None and i-last_ph>=min_sep:
    if q.close.iloc[i] > q.close.iloc[last_ph] and q.rsi14.iloc[i] < q.rsi14.iloc[last_ph]: sell[confirm]=True
   last_ph=i
 return buy,sell

def trades(q,tf,w,min_sep,delay):
 buy,sell=causal_divergence_signals(q,w,min_sep,delay); out=[]; blocked=-1
 for i in range(30,len(q)-2):
  if i<=blocked: continue
  dr=1 if buy[i] else (-1 if sell[i] else 0)
  if dr==0 or (buy[i] and sell[i]): continue
  ei=i+1; en=float(q.open.iloc[ei]); xi=min(len(q)-1,ei+MAXBARS[tf]); ex=float(q.close.iloc[xi])
  for j in range(ei,xi+1):
   ema=float(q.ema9.iloc[j]); lo=float(q.low.iloc[j]); hi=float(q.high.iloc[j])
   if np.isfinite(ema) and lo<=ema<=hi:
    ex=ema; xi=j; break
  raw=dr*(ex-en)/max(abs(en),1e-12)
  out.append((q.datetime.iloc[ei],raw,dr,xi-ei+1)); blocked=xi
 return pd.DataFrame(out,columns=['entry_time','raw_ret','dir','hold'])

def metrics(t,cost_bps=0.0):
 if len(t)==0:return {'N':0,'WR':0,'PF':0,'MeanRet':0,'NetScore':0,'Month21Score':0,'MaxDDScore':0}
 a=t.raw_ret.to_numpy(float)-cost_bps/10000.0
 N=len(a); gp=a[a>0].sum(); gl=-a[a<0].sum(); pf=gp/gl if gl>0 else (9999 if gp>0 else 0); wr=(a>0).mean()*100
 eq=INIT; peak=INIT; mdd=0
 for r in a:
  x=np.clip(r*100,-.025,.05); eq*=1+x; peak=max(peak,eq); mdd=max(mdd,100*(peak-eq)/peak)
 biz=max(1,len(pd.bdate_range(START.date(),END.date()))); mo=((eq/INIT)**(21/biz)-1)*100
 return {'N':N,'WR':wr,'PF':pf,'MeanRet':a.mean(),'NetScore':eq-INIT,'Month21Score':mo,'MaxDDScore':mdd}

def split_metrics(t):
 split=START+(END-START)*2/3
 tr=t[t.entry_time<split]; te=t[t.entry_time>=split]
 a=metrics(tr); b=metrics(te)
 return {**{f'Train_{k}':v for k,v in a.items()},**{f'Test_{k}':v for k,v in b.items()}}

data=load(); rows=[]; stress=[]
for tf,q in data.items():
 for w in [2,3,4,5]:
  for sep in [3,5,8]:
   for delay in [0,1,2]:
    t=trades(q,tf,w,sep,delay); m=metrics(t); sm=split_metrics(t)
    rows.append({'TF':tf,'w':w,'min_sep':sep,'delay':delay,**m,**sm})
    if w==3 and sep==3 and delay==0:
     for c in [0,1,2,5]:stress.append({'TF':tf,'cost_bps':c,**metrics(t,c)})
r=pd.DataFrame(rows)
# robust pass: enough OOS trades, positive OOS PF, not dependent on exact pivot window
r['OOS_PASS']=(r.Test_N>=10)&(r.Test_PF>1.10)&(r.Test_WR>50)
r['RobustScore']=np.where(r.OOS_PASS,np.log1p(r.Test_PF.clip(0,20))*2+r.Test_WR/50+r.Test_Month21Score/20-r.Test_MaxDDScore/10,-999)
r.sort_values(['OOS_PASS','RobustScore'],ascending=[False,False]).to_csv(OUT/'audit_matrix.csv',index=False)
pd.DataFrame(stress).to_csv(OUT/'cost_stress.csv',index=False)
# summarize parameter-neighborhood survival by TF
summ=[]
for tf,g in r.groupby('TF'):
 summ.append({'TF':tf,'variants':len(g),'oos_passes':int(g.OOS_PASS.sum()),'pass_rate':float(g.OOS_PASS.mean()),'median_test_pf':float(g.Test_PF.replace(9999,np.nan).median()),'median_test_wr':float(g.Test_WR.median()),'median_test_N':float(g.Test_N.median())})
pd.DataFrame(summ).to_csv(OUT/'robustness_summary.csv',index=False)
meta={'verification':'CAUSAL_DIVERGENCE_AUDIT_V2','critical_fix':'pivot signal delayed by w right-side bars; no centered-window lookahead at entry','walk_forward':'first 2/3 vs final 1/3','robustness_grid':'w=2..5, min_sep=3/5/8, signal delay=0/1/2 after causal confirmation','cost_stress_bps':[0,1,2,5],'exit':'EMA9 touch / TF maxbars','g75':False,'boost':False,'costs_in_main':False}
(OUT/'meta.json').write_text(json.dumps(meta,indent=2)); print(r.sort_values('RobustScore',ascending=False).head(20).to_string(index=False)); print(pd.DataFrame(summ).to_string(index=False))