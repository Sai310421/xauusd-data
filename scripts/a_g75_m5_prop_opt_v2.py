import json, math
from pathlib import Path
import pandas as pd, numpy as np

D=Path('csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv')
O=Path('bt_results/a_g75_m5_prop_opt_v2'); O.mkdir(parents=True,exist_ok=True)
P=.1
KP=np.array([80,100,110,130,150,160,170,190,210,220,230,250,270.])
KV=np.array([.07,.08,.09,.10,.11,.12,.13,.14,.15,.16,.17,.18,.20])
def K(x): return float(KV[np.argmin(abs(KP-x))])

df=pd.read_csv(D); df.columns=[x.lower() for x in df.columns]; df['datetime']=pd.to_datetime(df.datetime); df=df.sort_values('datetime').reset_index(drop=True)
H=df.high.to_numpy(); L=df.low.to_numpy(); C=df.close.to_numpy(); dates=df.datetime.dt.date.to_numpy(); N=len(df)
s1=np.full(N,np.nan); s2=np.full(N,np.nan); r1=np.full(N,np.nan); r2=np.full(N,np.nan); hs=[]; ls=[]
for i in range(N):
    j=i-3
    if j>=3 and j+3<N:
        if H[j]>max(H[j-3:j]) and H[j]>=max(H[j+1:j+4]): hs=(hs+[H[j]])[-2:]
        if L[j]<min(L[j-3:j]) and L[j]<=min(L[j+1:j+4]): ls=(ls+[L[j]])[-2:]
    if hs: r1[i]=hs[-1]
    if len(hs)>1: r2[i]=hs[-2]
    if ls: s1[i]=ls[-1]
    if len(ls)>1: s2[i]=ls[-2]

# Run at reference lot; structure does not depend on lot, so later rescale exact PnL/MTM to candidate lot.
def run_ref(trigger=.12,add=.025,rev=.20,maxadds=10,g2=45,g3=130,rf=False,lot=.001):
    pos=None; tr=[]; eq=1000.; peak=1000.; max_float_dd=0.; max_daily_dd=0.; adds=0; base=0
    curday=None; day_anchor=1000.; day_peak=1000.
    for i in range(10,N):
        if dates[i]!=curday:
            curday=dates[i]; day_anchor=eq; day_peak=eq
        if pos:
            adv=L[i] if pos['d']==1 else H[i]
            mtm=sum(((adv-e[0]) if pos['d']==1 else (e[0]-adv))*e[1]*100 for e in pos['e'])
            float_eq=eq+mtm
            max_float_dd=max(max_float_dd,100*(peak-float_eq)/peak)
            day_peak=max(day_peak,eq)
            day_ref=max(day_anchor,day_peak)
            max_daily_dd=max(max_daily_dd,100*(day_ref-float_eq)/day_ref)
            avg=sum(x*y for x,y in pos['e'])/sum(y for x,y in pos['e']); x=C[i]; ex=None; reason=''
            if not pos['a']:
                sl=avg-3 if pos['d']==1 else avg+3; tp=avg+6 if pos['d']==1 else avg-6
                slh=L[i]<=sl if pos['d']==1 else H[i]>=sl; tph=H[i]>=tp if pos['d']==1 else L[i]<=tp
                if slh: ex=sl; reason='sl'
                elif tph: ex=tp; reason='tp'
                elif ((x>=avg+trigger) if pos['d']==1 else (x<=avg-trigger)):
                    pos['a']=1; pos['pk']=x; pos['la']=avg+trigger if pos['d']==1 else avg-trigger
            if ex is None and pos['a']:
                if pos['d']==1:
                    pos['pk']=max(pos['pk'],x)
                    while pos['n']<maxadds and x>=pos['la']+add:
                        pos['la']+=add; pos['e'].append((pos['la'],lot)); pos['n']+=1; adds+=1
                    if x<=pos['pk']-rev: ex=x; reason='g75'
                else:
                    pos['pk']=min(pos['pk'],x)
                    while pos['n']<maxadds and x<=pos['la']-add:
                        pos['la']-=add; pos['e'].append((pos['la'],lot)); pos['n']+=1; adds+=1
                    if x>=pos['pk']+rev: ex=x; reason='g75'
            if ex is not None:
                pnl=sum(((ex-p) if pos['d']==1 else (p-ex))*v*100 for p,v in pos['e'])
                eq+=pnl; peak=max(peak,eq); day_peak=max(day_peak,eq); tr.append((pnl,pos['n'],reason,dates[i])); pos=None; continue
        if pos is None and all(np.isfinite([r1[i-1],r2[i-1],s1[i-1],s2[i-1]])) and r1[i-1]>s1[i-1]:
            reg=1 if r1[i-1]>r2[i-1] and s1[i-1]>s2[i-1] else (-1 if r1[i-1]<r2[i-1] and s1[i-1]<s2[i-1] else 0)
            k=K((r1[i-1]-s1[i-1])/P); b=(math.sqrt(r1[i-1])-k)**2; s=(math.sqrt(s1[i-1])+k)**2; ods=[]
            if not rf or reg>=0: ods += [(1,b),(1,b-g2*P),(1,b-(g2+g3)*P)]
            if not rf or reg<=0: ods += [(-1,s),(-1,s+g2*P),(-1,s+(g2+g3)*P)]
            v=[]
            for d,p in ods:
                if d==1 and p<C[i-1] and L[i]<=p<=H[i]: v.append((d,p))
                if d==-1 and p>C[i-1] and L[i]<=p<=H[i]: v.append((d,p))
            ds=set(d for d,p in v)
            if len(ds)==1:
                d=next(iter(ds)); es=[(p,lot) for dd,p in v if dd==d]; base+=len(es)
                pos={'d':d,'e':es,'a':0,'pk':C[i],'la':sum(p for p,v in es)/len(es),'n':0}
    if not tr: return None
    t=pd.DataFrame(tr,columns=['p','n','r','date']); gp=t[t.p>0].p.sum(); gl=-t[t.p<0].p.sum()
    return dict(N=len(t),WR=(t.p>0).mean()*100,PF=gp/gl if gl else 9999,Net=t.p.sum(),BaseEntries=base,G75Adds=adds,AvgAdds=t.n.mean(),MaxDD_float=max_float_dd,MaxDailyDD=max_daily_dd)

# focused structural sweep. rf=False retained because prior test preserved N and higher return.
rows=[]
triggers=[.08,.10,.12,.14,.16]
adds=[.020,.025,.030,.040]
revs=[.15,.20,.25,.30]
maxadds_list=[6,8,10]
g2s=[30,45,60]
g3s=[100,130,160]
lot_scales=[1.00,1.05,1.10,1.15,1.20,1.25,1.30]
biz=len(set(d for d in dates if pd.Timestamp(d).weekday()<5))
for trigger in triggers:
  for add in adds:
   for rev in revs:
    for ma in maxadds_list:
     for g2 in g2s:
      for g3 in g3s:
       rr=run_ref(trigger,add,rev,ma,g2,g3,False,.001)
       if not rr: continue
       # PF/WR/N unchanged by scaling. Re-run DD accurately for scaled lots only when likely promising using proportional prefilter.
       for sc in lot_scales:
        lot=.001*sc
        # exact Net scaling; approximate DD scaling is conservative enough for shortlist, then exact-refinement later.
        net=rr['Net']*sc; final=1000+net
        if final<=0: continue
        mo=((final/1000)**(21/biz)-1)*100
        mdd=rr['MaxDD_float']*sc
        ddd=rr['MaxDailyDD']*sc
        rows.append(dict(trigger=trigger,add=add,reversal=rev,maxadds=ma,g2=g2,g3=g3,lot=lot,rf=False,**{k:v for k,v in rr.items() if k not in ['Net','MaxDD_float','MaxDailyDD']},Net=net,Final=final,MaxDD_float=mdd,MaxDailyDD=ddd,Month21_pct=mo,BusinessDays=biz))
out=pd.DataFrame(rows)
out.to_csv(O/'all_candidates.csv',index=False)
prop=out[(out.MaxDD_float<=5.0)&(out.MaxDailyDD<=3.5)].sort_values(['Month21_pct','PF'],ascending=False)
prop.to_csv(O/'prop_pass.csv',index=False)
target=prop[prop.Month21_pct>=50].copy(); target.to_csv(O/'target_50.csv',index=False)
summary={'rows':N,'start':str(df.datetime.min()),'end':str(df.datetime.max()),'tested':len(out),'prop_pass':len(prop),'target50':len(target),'top':prop.head(20).to_dict('records')}
(O/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
