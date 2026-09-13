import json, math, runpy
from pathlib import Path
import pandas as pd, numpy as np

OUT=Path('bt_results/abc_shared_boost_v1'); OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-02-25'); END=pd.Timestamp('2026-05-26 23:59:59'); INIT=1000.0

def rsi_wilder(s,n=14):
 d=s.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0)
 au=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); ad=dn.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
 rs=au/ad.replace(0,np.nan); return 100-100/(1+rs)

def atr14(df):
 pc=df.close.shift(1); tr=pd.concat([df.high-df.low,(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()

def a_events():
 p=Path('csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv'); d=pd.read_csv(p); d.columns=[x.lower() for x in d.columns]; d['datetime']=pd.to_datetime(d.datetime); d=d[(d.datetime>=START)&(d.datetime<=END)].reset_index(drop=True)
 H=d.high.to_numpy();L=d.low.to_numpy();C=d.close.to_numpy(); n=len(d); hs=[];ls=[]; r1=np.full(n,np.nan);r2=np.full(n,np.nan);s1=np.full(n,np.nan);s2=np.full(n,np.nan)
 P=.1; KP=np.array([80,100,110,130,150,160,170,190,210,220,230,250,270.]); KV=np.array([.07,.08,.09,.10,.11,.12,.13,.14,.15,.16,.17,.18,.20])
 for i in range(n):
  j=i-3
  if j>=3 and j+3<n:
   if H[j]>max(H[j-3:j]) and H[j]>=max(H[j+1:j+4]): hs=(hs+[H[j]])[-2:]
   if L[j]<min(L[j-3:j]) and L[j]<=min(L[j+1:j+4]): ls=(ls+[L[j]])[-2:]
  if hs:r1[i]=hs[-1]
  if len(hs)>1:r2[i]=hs[-2]
  if ls:s1[i]=ls[-1]
  if len(ls)>1:s2[i]=ls[-2]
 def K(x):return float(KV[np.argmin(abs(KP-x))])
 trigger=.08; add=.025; rev=.20; maxadds=10; g2=60; g3=130; lot=.0013
 pos=None; ev=[]
 for i in range(10,n):
  if pos:
   avg=sum(x*y for x,y in pos['e'])/sum(y for x,y in pos['e']); x=C[i]; ex=None
   if not pos['a']:
    sl=avg-3 if pos['d']==1 else avg+3; tp=avg+6 if pos['d']==1 else avg-6
    if (L[i]<=sl if pos['d']==1 else H[i]>=sl): ex=sl
    elif (H[i]>=tp if pos['d']==1 else L[i]<=tp): ex=tp
    elif (x>=avg+trigger if pos['d']==1 else x<=avg-trigger): pos['a']=1; pos['pk']=x; pos['la']=avg+trigger if pos['d']==1 else avg-trigger
   if ex is None and pos['a']:
    if pos['d']==1:
     pos['pk']=max(pos['pk'],x)
     while pos['n']<maxadds and x>=pos['la']+add: pos['la']+=add; pos['e'].append((pos['la'],lot)); pos['n']+=1
     if x<=pos['pk']-rev: ex=x
    else:
     pos['pk']=min(pos['pk'],x)
     while pos['n']<maxadds and x<=pos['la']-add: pos['la']-=add; pos['e'].append((pos['la'],lot)); pos['n']+=1
     if x>=pos['pk']+rev: ex=x
   # adverse bar estimate
   adv=L[i] if pos['d']==1 else H[i]; mtm=sum(((adv-e[0]) if pos['d']==1 else (e[0]-adv))*e[1]*100 for e in pos['e']); pos['mae']=min(pos['mae'],mtm)
   if ex is not None:
    pnl=sum(((ex-px) if pos['d']==1 else (px-ex))*v*100 for px,v in pos['e']); ev.append(dict(strategy='A',symbol='XAUUSD',direction=pos['d'],entry_time=pos['t'],exit_time=d.datetime.iloc[i],pnl_frac=pnl/INIT,mae_frac=pos['mae']/INIT)); pos=None; continue
  if pos is None and all(np.isfinite([r1[i-1],r2[i-1],s1[i-1],s2[i-1]])) and r1[i-1]>s1[i-1]:
   k=K((r1[i-1]-s1[i-1])/P); b=(math.sqrt(r1[i-1])-k)**2; s=(math.sqrt(s1[i-1])+k)**2; ods=[(1,b),(1,b-g2*P),(1,b-(g2+g3)*P),(-1,s),(-1,s+g2*P),(-1,s+(g2+g3)*P)]; v=[]
   for dr,px in ods:
    if dr==1 and px<C[i-1] and L[i]<=px<=H[i]:v.append((dr,px))
    if dr==-1 and px>C[i-1] and L[i]<=px<=H[i]:v.append((dr,px))
   ds=set(dr for dr,px in v)
   if len(ds)==1:
    dr=next(iter(ds)); es=[(px,lot) for dd,px in v if dd==dr]; pos={'d':dr,'e':es,'a':0,'pk':C[i],'la':sum(px for px,vv in es)/len(es),'n':0,'t':d.datetime.iloc[i],'mae':0.0}
 return ev

def b_events():
 book=json.loads(Path('duka/books/xauusd_mtf.json').read_text()); q=pd.DataFrame(book['tfs']['M15']).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close'}); q['datetime']=pd.to_datetime(q.ts,unit='s',utc=True).dt.tz_convert(None); q=q[(q.datetime>=START)&(q.datetime<=END)].reset_index(drop=True)
 q['rsi']=rsi_wilder(q.close); q['ema9']=q.close.ewm(span=9,adjust=False).mean(); q['atr']=atr14(q)
 buys=[10.7,14.2,21.5,23.4,34.5,36.05]; sells=[63.08,67.2,70.08,77.3,82.8,84.9]
 lot=.001; trig=.35; add=.15; rev=.35; maxadds=10; ev=[]; pos=None
 for i in range(15,len(q)-1):
  if pos:
   hi=float(q.high.iloc[i]);lo=float(q.low.iloc[i]);cl=float(q.close.iloc[i]); ema=float(q.ema9.iloc[i]); pos['pk']=max(pos['pk'],hi) if pos['d']==1 else pos['pk']; pos['tr']=min(pos['tr'],lo) if pos['d']==-1 else pos['tr']
   adv=lo if pos['d']==1 else hi; mtm=sum(((adv-px) if pos['d']==1 else (px-adv))*v*100 for px,v in pos['e']); pos['mae']=min(pos['mae'],mtm)
   fav=pos['d']*(cl-pos['base']); av=pos['atr']
   if fav>=trig*av:
    while pos['n']<maxadds and pos['d']*(cl-pos['last'])>=add*av: pos['last']+=pos['d']*add*av; pos['e'].append((pos['last'],lot)); pos['n']+=1
   revv=(pos['pk']-cl) if pos['d']==1 else (cl-pos['tr'])
   exit_cond=(lo<=ema<=hi) or (revv>=rev*av and pos['n']>0) or i-pos['i']>=80
   if exit_cond:
    ex=ema if lo<=ema<=hi else cl; pnl=sum(((ex-px) if pos['d']==1 else (px-ex))*v*100 for px,v in pos['e']); ev.append(dict(strategy='B',symbol='XAUUSD',direction=pos['d'],entry_time=pos['t'],exit_time=q.datetime.iloc[i],pnl_frac=pnl/INIT,mae_frac=pos['mae']/INIT)); pos=None; continue
  r=float(q.rsi.iloc[i]) if np.isfinite(q.rsi.iloc[i]) else 50
  dr=1 if r<=max(buys) else (-1 if r>=min(sells) else 0)
  if dr:
   px=float(q.open.iloc[i+1]); av=float(q.atr.iloc[i]) if np.isfinite(q.atr.iloc[i]) else 1.; pos={'d':dr,'base':px,'e':[(px,lot)],'last':px,'n':0,'pk':px,'tr':px,'atr':av,'i':i+1,'t':q.datetime.iloc[i+1],'mae':0.0}
 return ev

def c_events(sym):
 ns=runpy.run_path('scripts/c0_multisymbol_m1m5_bt_v1.py'); normalize=ns['normalize']; cands=ns['candidates']; wait=ns['WAIT']; df=normalize(pd.read_parquet(Path(f'parquet/{sym}_M1_90d_20260526.parquet'))); df['atr']=atr14(df); ev=[]; blocked=-1; par=dict(tr=.35,ad=.15,rv=.35,ma=10,af=1.0)
 for c in cands(df,'M1'):
  if c['conf_i']<=blocked:continue
  sig=None
  for i in range(c['conf_i'],min(len(df)-1,c['conf_i']+wait['M1']+1)):
   if (c['dir']==1 and df.close.iloc[i]>c['choch']) or (c['dir']==-1 and df.close.iloc[i]<c['choch']):sig=i;break
  if sig is None:continue
  ei=sig+1; entry=float(df.open.iloc[ei]); av=float(df.atr.iloc[ei]) if np.isfinite(df.atr.iloc[ei]) else abs(entry)*.0005; buf=max(av*.15,abs(entry)*1e-7); sl=c['D']-buf if c['dir']==1 else c['D']+buf; R=(entry-sl) if c['dir']==1 else (sl-entry)
  if R<=0:continue
  risk=INIT*.0025; base_v=risk/R; parts=[.5,.3,.2]; alive=[1,1,1]; tg=[entry+c['dir']*R,entry+c['dir']*2*R,max(c['A'],entry+3*R) if c['dir']==1 else min(c['A'],entry-3*R)]; adds=[]; pk=entry;tr=entry;last=entry;act=False; mae=0.; end=min(len(df)-1,ei+180); exi=end
  for i in range(ei,end+1):
   hi=float(df.high.iloc[i]);lo=float(df.low.iloc[i]);cl=float(df.close.iloc[i]);pk=max(pk,hi);tr=min(tr,lo); adverse=((lo-entry) if c['dir']==1 else (entry-hi))*sum(parts[j] for j,a in enumerate(alive) if a)*base_v+sum((((lo-x[0]) if c['dir']==1 else (x[0]-hi))*x[1]) for x in adds); mae=min(mae,adverse)
   if (lo<=sl if c['dir']==1 else hi>=sl):
    pnl=-risk*sum(parts[j] for j,a in enumerate(alive) if a)+sum(c['dir']*(sl-x[0])*x[1] for x in adds); exi=i; break
   pnl=0
   for j,t in enumerate(tg):
    if alive[j] and (hi>=t if c['dir']==1 else lo<=t):pnl+=parts[j]*risk*abs(t-entry)/R;alive[j]=0
   if not act and c['dir']*(cl-entry)>=par['tr']*av:act=True
   if act:
    while len(adds)<par['ma'] and c['dir']*(cl-last)>=par['ad']*av:last+=c['dir']*par['ad']*av;adds.append((last,base_v*par['af']))
   if act and adds and ((pk-cl) if c['dir']==1 else (cl-tr))>=par['rv']*av: pnl+=sum(c['dir']*(cl-x[0])*x[1] for x in adds);adds=[];act=False;pk=cl;tr=cl;last=cl
   if not any(alive) and not adds:exi=i;break
   exi=i
  else:pnl=0
  ex=float(df.close.iloc[exi]); pnl+=sum(c['dir']*(ex-entry)*base_v*parts[j] for j,a in enumerate(alive) if a)+sum(c['dir']*(ex-x[0])*x[1] for x in adds)
  ev.append(dict(strategy='C',symbol=sym,direction=c['dir'],entry_time=df.datetime.iloc[ei],exit_time=df.datetime.iloc[exi],pnl_frac=pnl/INIT,mae_frac=mae/INIT));blocked=exi
 return ev

def portfolio(events,boost):
 events=sorted(events,key=lambda x:x['entry_time']); eq=INIT; peak=INIT; maxdd=0.; daily=0.; curday=None; day_anchor=INIT; active=[]; rows=[]; boosts=0; boost_pnl=0.
 for e in events:
  active=[a for a in active if a['exit_time']>=e['entry_time']]
  if e['entry_time'].date()!=curday:curday=e['entry_time'].date();day_anchor=eq
  same=[a for a in active if a['symbol']==e['symbol'] and a['direction']==e['direction'] and a['strategy']!=e['strategy']]; opp=[a for a in active if a['symbol']==e['symbol'] and a['direction']!=e['direction']]
  m=1.0
  if boost and not opp:
   if len(same)>=2:m=1.5
   elif len(same)==1:m=1.25
  # DD safety gate: suppress boost if projected adverse DD >5%
  proj=100*(peak-(eq+e['mae_frac']*eq*m))/peak
  if proj>5:m=1.0
  if m>1:boosts+=1
  pnl=e['pnl_frac']*eq*m; base=e['pnl_frac']*eq; mae=e['mae_frac']*eq*m; maxdd=max(maxdd,100*(peak-(eq+mae))/peak); daily=max(daily,100*(day_anchor-(eq+mae))/day_anchor); eq+=pnl; peak=max(peak,eq); boost_pnl+=pnl-base; rows.append({**e,'mult':m,'pnl_dollar':pnl,'equity':eq}); active.append(e)
 a=np.array([r['pnl_dollar'] for r in rows]); gp=a[a>0].sum();gl=-a[a<0].sum(); pf=gp/gl if gl>0 else 9999; wr=(a>0).mean()*100 if len(a) else 0; biz=64; mo=((eq/INIT)**(21/biz)-1)*100
 return dict(N=len(rows),WR=wr,PF=pf,Net=eq-INIT,Final=eq,Month21_pct=mo,MaxDD_float=maxdd,MaxDailyDD=daily,BoostCount=boosts,BoostPnL=boost_pnl),pd.DataFrame(rows)

events=a_events()+b_events()+c_events('XAUUSD')+c_events('EURUSD')
base,db=portfolio(events,False); boost,dx=portfolio(events,True)
pd.DataFrame([{'mode':'OFF',**base},{'mode':'ON',**boost}]).to_csv(OUT/'summary.csv',index=False); db.to_csv(OUT/'trades_off.csv',index=False);dx.to_csv(OUT/'trades_on.csv',index=False)
meta={'verification':'ABC_SHARED_EQUITY_RECONSTRUCTED_PROXY','A':'M5 sqrt+G75 optimized params from A v2','B':'M15 RSI14 touch->EMA9 reconstructed + ATR G75; source exact trigger/period unresolved','C':'M1 C0 balanced + ATR G75 best top2','boost':'same symbol+same direction: 2 engines=1.25x, 3=1.5x; opposite=no boost; projected DD>5 suppresses boost','costs':'none','resample':False,'OFF':base,'ON':boost}
(OUT/'summary.json').write_text(json.dumps(meta,indent=2,default=str));print(json.dumps(meta,indent=2,default=str))