import json, math, runpy
from pathlib import Path
import pandas as pd, numpy as np

OUT=Path('bt_results/abc_math_dd_controller_v1'); OUT.mkdir(parents=True,exist_ok=True)
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

def risk_scale(dd, ddd, p):
    if dd < p['d1']: md=1.0
    elif dd < p['d2']: md=p['m1']
    elif dd < p['d3']: md=p['m2']
    elif dd < p['d4']: md=p['m3']
    else: md=p['m4']
    if ddd < p['q1']: mq=1.0
    elif ddd < p['q2']: mq=p['q_m1']
    elif ddd < p['q3']: mq=p['q_m2']
    else: mq=p['q_m3']
    return min(md,mq)

def portfolio_math(events,p):
    events=sorted(events,key=lambda x:x['entry_time'])
    eq=INIT; peak=INIT; maxdd=0.; daily=0.; curday=None; day_anchor=INIT
    active=[]; rows=[]; boosts=0; boost_pnl=0.; compressed=0; hard_brakes=0
    for e in events:
        active=[a for a in active if a['exit_time']>=e['entry_time']]
        if e['entry_time'].date()!=curday:
            curday=e['entry_time'].date(); day_anchor=eq
        dd=max(0.,100*(peak-eq)/peak); ddd=max(0.,100*(day_anchor-eq)/day_anchor)
        rs=risk_scale(dd,ddd,p)
        if rs<.999: compressed+=1
        if rs<=p['m4']+1e-12: hard_brakes+=1
        same=[a for a in active if a['symbol']==e['symbol'] and a['direction']==e['direction'] and a['strategy']!=e['strategy']]
        opp=[a for a in active if a['symbol']==e['symbol'] and a['direction']!=e['direction']]
        bm=1.0
        if not opp and dd < p['boost_stop_dd'] and ddd < p['boost_stop_daily']:
            if len(same)>=2: bm=1.5
            elif len(same)==1: bm=1.25
        raw_mae=abs(e['mae_frac'])*eq*rs*bm
        cap_dd=max(0.,eq-(peak*(1-p['hard_dd']/100))); cap_daily=max(0.,eq-(day_anchor*(1-p['hard_daily']/100)))
        allowable=min(cap_dd,cap_daily)
        if raw_mae>0 and allowable>=0 and raw_mae>allowable:
            brake=max(p['floor'],allowable/raw_mae); rs*=brake; hard_brakes+=1
        if bm>1: boosts+=1
        pnl=e['pnl_frac']*eq*rs*bm; no_boost=e['pnl_frac']*eq*rs; mae=e['mae_frac']*eq*rs*bm
        maxdd=max(maxdd,100*(peak-(eq+mae))/peak); daily=max(daily,100*(day_anchor-(eq+mae))/day_anchor)
        eq+=pnl; peak=max(peak,eq); boost_pnl+=pnl-no_boost
        rows.append({**e,'risk_mult':rs,'boost_mult':bm,'pnl_dollar':pnl,'equity':eq,'pre_dd':dd,'pre_daily_dd':ddd}); active.append(e)
    a=np.array([r['pnl_dollar'] for r in rows]); gp=a[a>0].sum(); gl=-a[a<0].sum(); pf=gp/gl if gl>0 else 9999; wr=(a>0).mean()*100 if len(a) else 0; biz=64
    mo=((eq/INIT)**(21/biz)-1)*100 if eq>0 else -100
    return dict(N=len(rows),WR=wr,PF=pf,Net=eq-INIT,Final=eq,Month21_pct=mo,MaxDD_float=maxdd,MaxDailyDD=daily,BoostCount=boosts,BoostPnL=boost_pnl,CompressedEvents=compressed,HardBrakes=hard_brakes,AvgRiskMult=float(np.mean([r['risk_mult'] for r in rows]))),pd.DataFrame(rows)

events=a_events()+b_events()+c_events('XAUUSD')+c_events('EURUSD')
params=[]
for d1 in [2.0,2.5,3.0]:
 for d2 in [3.0,3.5]:
  if d2<=d1: continue
  for d3 in [3.75,4.0,4.25]:
   if d3<=d2: continue
   for d4 in [4.25,4.5,4.75]:
    if d4<=d3: continue
    for m1,m2,m3,m4 in [(0.90,0.75,0.55,0.35),(0.85,0.65,0.45,0.25),(0.95,0.80,0.60,0.40)]:
     for boost_stop_dd in [3.0,3.5,4.0]:
      params.append(dict(d1=d1,d2=d2,d3=d3,d4=d4,m1=m1,m2=m2,m3=m3,m4=m4,q1=1.5,q2=2.5,q3=3.0,q_m1=.85,q_m2=.65,q_m3=.40,boost_stop_dd=boost_stop_dd,boost_stop_daily=2.75,hard_dd=4.5,hard_daily=3.5,floor=.20))
rows=[]; trade_cache={}
for k,p in enumerate(params):
 r,t=portfolio_math(events,p); rows.append({'id':k,**p,**r}); trade_cache[k]=t
out=pd.DataFrame(rows); out.to_csv(OUT/'all_candidates.csv',index=False)
pass_df=out[(out.MaxDD_float<=4.5+1e-9)&(out.MaxDailyDD<=3.5+1e-9)].copy().sort_values(['Month21_pct','PF'],ascending=False)
pass_df.head(50).to_csv(OUT/'pass_dd45_daily35.csv',index=False)
if len(pass_df): best=pass_df.iloc[0].to_dict(); bid=int(best['id'])
else: best=out.sort_values(['MaxDD_float','Month21_pct'],ascending=[True,False]).iloc[0].to_dict(); bid=int(best['id'])
trade_cache[bid].to_csv(OUT/'trades_best.csv',index=False)
meta={'verification':'ABC_MATH_DD_CONTROLLER_RECONSTRUCTED_PROXY','controller':'piecewise no-action/reflection + daily-DD limiter + predictive hard boundary brake; preserves N','targets':{'MaxDD_float':4.5,'MaxDailyDD':3.5},'costs':'none','resample':False,'tested':len(out),'passed':len(pass_df),'best':best}
(OUT/'summary.json').write_text(json.dumps(meta,indent=2,default=str)); pd.DataFrame([best]).to_csv(OUT/'best.csv',index=False); print(json.dumps(meta,indent=2,default=str))