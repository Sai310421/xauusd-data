import json, math
from pathlib import Path
import pandas as pd
import numpy as np

OUT=Path('bt_results/c0_mtf_standalone_v1'); OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-02-25 00:00:00'); END=pd.Timestamp('2026-05-26 23:59:59')
TF_ORDER=['M1','M5','M15','H1','H2','H4','D1']
PROFILES={
 'image_tight':{'B':(.697,.706),'C':(.654,.667),'D':(.825,.835)},
 'balanced':{'B':(.62,.78),'C':(.55,.78),'D':(.78,.90)},
}
PIVOT_LR={'M1':2,'M5':2,'M15':2,'H1':3,'H2':3,'H4':3,'D1':2}
WAIT={'M1':12,'M5':12,'M15':10,'H1':8,'H2':8,'H4':6,'D1':4}
MAX_HOLD={'M1':180,'M5':120,'M15':80,'H1':48,'H2':36,'H4':24,'D1':12}
TOTAL_LOT=.001; CONTRACT=100.; INITIAL=1000.

def load_data():
 out={}
 p=Path('csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv')
 d=pd.read_csv(p); d.columns=[x.lower() for x in d.columns]; d['datetime']=pd.to_datetime(d.datetime)
 out['M1']=d[(d.datetime>=START)&(d.datetime<=END)].reset_index(drop=True)
 book=json.loads(Path('duka/books/xauusd_mtf.json').read_text())
 for tf in ['M5','M15','H1','H4','D1']:
  rows=book['tfs'].get(tf,[])
  if not rows: continue
  q=pd.DataFrame(rows).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close'})
  q['datetime']=pd.to_datetime(q.ts,unit='s',utc=True).dt.tz_convert(None)
  q=q[(q.datetime>=START)&(q.datetime<=END)][['datetime','open','high','low','close']].reset_index(drop=True)
  out[tf]=q
 return out

def atr14(df):
 pc=df.close.shift(1)
 tr=pd.concat([(df.high-df.low),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()

def confirmed_pivots(df,lr):
 H=df.high.to_numpy(); L=df.low.to_numpy(); n=len(df); events=[]
 for i in range(2*lr,n):
  j=i-lr
  wh=H[j-lr:j+lr+1]; wl=L[j-lr:j+lr+1]
  if H[j]>=np.max(wh): events.append((i,j,'H',float(H[j])))
  if L[j]<=np.min(wl): events.append((i,j,'L',float(L[j])))
 events.sort(key=lambda z:(z[0],z[1],z[2]))
 # Alternate pivots; if same type, keep more extreme one.
 alt=[]
 for e in events:
  if not alt or alt[-1][2]!=e[2]: alt.append(e)
  else:
   if (e[2]=='H' and e[3]>=alt[-1][3]) or (e[2]=='L' and e[3]<=alt[-1][3]): alt[-1]=e
 return alt

def pattern_candidates(df,tf,profile):
 piv=confirmed_pivots(df,PIVOT_LR[tf]); bands=PROFILES[profile]; out=[]
 for k in range(4,len(piv)):
  five=piv[k-4:k+1]; typ=''.join(x[2] for x in five)
  if typ not in ('LHLHL','HLHLH'): continue
  X,A,B,C,D=[x[3] for x in five]
  xa=abs(A-X); ab=abs(A-B)
  if xa<=0 or ab<=0: continue
  rb=abs(A-B)/xa; rc=abs(C-B)/ab; rd=abs(A-D)/xa
  if not (bands['B'][0]<=rb<=bands['B'][1] and bands['C'][0]<=rc<=bands['C'][1] and bands['D'][0]<=rd<=bands['D'][1]): continue
  bull=(typ=='LHLHL')
  qm=(D<B and C<A) if bull else (D>B and C>A)
  if not qm: continue
  conf_i=five[-1][0]; d_i=five[-1][1]
  # D itself is the liquidity sweep; wait for causal local structure break.
  pre0=max(0,d_i-3)
  choch=float(df.high.iloc[pre0:d_i].max()) if bull else float(df.low.iloc[pre0:d_i].min())
  if not np.isfinite(choch): continue
  out.append(dict(conf_i=conf_i,d_i=d_i,dir=1 if bull else -1,X=X,A=A,B=B,C=C,D=D,rb=rb,rc=rc,rd=rd,choch=choch))
 return out

def simulate_trade(df,entry_i,direction,D,A,atr):
 if entry_i>=len(df): return None
 entry=float(df.open.iloc[entry_i]); buf=max(.15,float(atr.iloc[entry_i])*.15 if np.isfinite(atr.iloc[entry_i]) else .15)
 if direction==1:
  sl=D-buf; R=entry-sl
  if R<=0:return None
  t1=entry+R; t2=entry+2*R; tr=max(A,entry+3*R)
 else:
  sl=D+buf; R=sl-entry
  if R<=0:return None
  t1=entry-R; t2=entry-2*R; tr=min(A,entry-3*R)
 parts=[.5,.3,.2]; targets=[t1,t2,tr]; alive=[True,True,True]; pnl=0.; end_i=entry_i; min_open=0.
 end=min(len(df)-1,entry_i+MAX_HOLD_CUR)
 for i in range(entry_i,end+1):
  hi=float(df.high.iloc[i]); lo=float(df.low.iloc[i])
  # floating MAE for all currently open size
  open_frac=sum(parts[j] for j,a in enumerate(alive) if a)
  adverse=(lo-entry) if direction==1 else (entry-hi)
  min_open=min(min_open,adverse*TOTAL_LOT*open_frac*CONTRACT)
  slhit=(lo<=sl) if direction==1 else (hi>=sl)
  if slhit:
   for j,a in enumerate(alive):
    if a:
     pnl += ((sl-entry) if direction==1 else (entry-sl))*TOTAL_LOT*parts[j]*CONTRACT; alive[j]=False
   end_i=i; break
  # conservative: target processing after SL check
  for j,t in enumerate(targets):
   if alive[j] and ((hi>=t) if direction==1 else (lo<=t)):
    pnl += ((t-entry) if direction==1 else (entry-t))*TOTAL_LOT*parts[j]*CONTRACT; alive[j]=False
  end_i=i
  if not any(alive): break
 if any(alive):
  ex=float(df.close.iloc[end_i])
  for j,a in enumerate(alive):
   if a: pnl += ((ex-entry) if direction==1 else (entry-ex))*TOTAL_LOT*parts[j]*CONTRACT
 return dict(pnl=pnl,end_i=end_i,mae=min_open,entry=entry,sl=sl,R=R)

def run_tf(df,tf,profile):
 global MAX_HOLD_CUR
 MAX_HOLD_CUR=MAX_HOLD[tf]
 df=df.copy(); df['atr']=atr14(df); candidates=pattern_candidates(df,tf,profile)
 trades=[]; blocked_until=-1
 for c in candidates:
  if c['conf_i']<=blocked_until: continue
  sig=None
  for i in range(c['conf_i'],min(len(df)-1,c['conf_i']+WAIT[tf]+1)):
   if c['dir']==1 and float(df.close.iloc[i])>c['choch']: sig=i; break
   if c['dir']==-1 and float(df.close.iloc[i])<c['choch']: sig=i; break
  if sig is None or sig+1>=len(df): continue
  r=simulate_trade(df,sig+1,c['dir'],c['D'],c['A'],df.atr)
  if not r: continue
  r.update({k:c[k] for k in ['dir','rb','rc','rd']}); r['entry_time']=df.datetime.iloc[sig+1]
  trades.append(r); blocked_until=r['end_i']
 if not trades:
  return dict(TF=tf,profile=profile,rows=len(df),N=0,WR=0,PF=0,Net=0,Final=INITIAL,MaxDD_closed=0,MaxDD_float=0,Month21_pct=0)
 t=pd.DataFrame(trades); gp=t.loc[t.pnl>0,'pnl'].sum(); gl=-t.loc[t.pnl<0,'pnl'].sum(); eq=INITIAL+t.pnl.cumsum(); peak=eq.cummax(); dd=(peak-eq)/peak*100
 # Approx sequential floating DD using each trade MAE against equity peak before that trade.
 bal=INITIAL; pk=INITIAL; mfd=0.
 for _,r in t.iterrows():
  mfd=max(mfd,100*(pk-(bal+r.mae))/pk); bal+=r.pnl; pk=max(pk,bal)
 biz=len(set(x.date() for x in df.datetime if x.weekday()<5)); final=INITIAL+t.pnl.sum(); mo=((final/INITIAL)**(21/max(1,biz))-1)*100 if final>0 else -100
 return dict(TF=tf,profile=profile,rows=len(df),N=len(t),WR=(t.pnl>0).mean()*100,PF=gp/gl if gl>0 else 9999,Net=t.pnl.sum(),Final=final,MaxDD_closed=dd.max(),MaxDD_float=mfd,Month21_pct=mo,AvgB=t.rb.mean(),AvgC=t.rc.mean(),AvgD=t.rd.mean())

data=load_data(); rows=[]
for tf in TF_ORDER:
 if tf=='H2' or tf not in data:
  for profile in PROFILES: rows.append(dict(TF=tf,profile=profile,status='UNAVAILABLE_NATIVE_DATA',rows=0,N=0,WR=np.nan,PF=np.nan,Net=np.nan,Final=np.nan,MaxDD_closed=np.nan,MaxDD_float=np.nan,Month21_pct=np.nan))
  continue
 for profile in PROFILES:
  r=run_tf(data[tf],tf,profile); r['status']='OK'; rows.append(r)
out=pd.DataFrame(rows); out.to_csv(OUT/'per_tf.csv',index=False)
# Combined is arithmetic independent-ledger reference only; no correlation/netting assumptions.
comb=[]
for p in PROFILES:
 q=out[(out.profile==p)&(out.status=='OK')].copy()
 comb.append(dict(profile=p,available_TFs=','.join(q.TF),N=int(q.N.sum()),Net=float(q.Net.sum()),mean_PF=float(q[q.PF<9999].PF.mean()) if len(q[q.PF<9999]) else np.nan,max_singleTF_DD=float(q.MaxDD_float.max()),note='Independent TF ledgers; not a shared-equity portfolio DD.'))
pd.DataFrame(comb).to_csv(OUT/'combined_reference.csv',index=False)
summary={'verification':'C0_RECONSTRUCTED_PROXY_OHLC_CAUSAL','period':[str(START),str(END)],'native_available':list(data.keys()),'H2':'UNAVAILABLE_NATIVE_DATA_NO_RESAMPLE','no_g75':True,'costs':'none','profiles':PROFILES,'rows':out.to_dict('records')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
print(out.to_string(index=False)); print('\n',json.dumps(summary,indent=2,default=str)[:12000])
