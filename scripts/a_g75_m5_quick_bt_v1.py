import json,math
from pathlib import Path
import pandas as pd,numpy as np
D=Path('csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv');O=Path('bt_results/a_g75_m5_quick_v1');O.mkdir(parents=True,exist_ok=True)
P=.1;KP=np.array([80,100,110,130,150,160,170,190,210,220,230,250,270.]);KV=np.array([.07,.08,.09,.10,.11,.12,.13,.14,.15,.16,.17,.18,.20])
def K(x):return float(KV[np.argmin(abs(KP-x))])
df=pd.read_csv(D);df.columns=[x.lower() for x in df.columns];df['datetime']=pd.to_datetime(df.datetime);df=df.sort_values('datetime').reset_index(drop=True)
H=df.high.to_numpy();L=df.low.to_numpy();C=df.close.to_numpy();N=len(df);s1=np.full(N,np.nan);s2=np.full(N,np.nan);r1=np.full(N,np.nan);r2=np.full(N,np.nan);hs=[];ls=[]
for i in range(N):
 j=i-3
 if j>=3 and j+3<N:
  if H[j]>max(H[j-3:j]) and H[j]>=max(H[j+1:j+4]):hs=(hs+[H[j]])[-2:]
  if L[j]<min(L[j-3:j]) and L[j]<=min(L[j+1:j+4]):ls=(ls+[L[j]])[-2:]
 if hs:r1[i]=hs[-1]
 if len(hs)>1:r2[i]=hs[-2]
 if ls:s1[i]=ls[-1]
 if len(ls)>1:s2[i]=ls[-2]

def run(lot=.001,rf=False,g75=True,g2=45,g3=130):
 pos=None;tr=[];eq=1000.;peak=1000.;mfd=0.;adds=0;base=0
 for i in range(10,N):
  if pos:
   adv=L[i] if pos['d']==1 else H[i];mtm=sum(((adv-e[0]) if pos['d']==1 else (e[0]-adv))*e[1]*100 for e in pos['e']);mfd=max(mfd,100*(peak-(eq+mtm))/peak)
   avg=sum(x*y for x,y in pos['e'])/sum(y for x,y in pos['e']);x=C[i];ex=None;reason=''
   if not pos['a']:
    sl=avg-3 if pos['d']==1 else avg+3;tp=avg+6 if pos['d']==1 else avg-6;slh=L[i]<=sl if pos['d']==1 else H[i]>=sl;tph=H[i]>=tp if pos['d']==1 else L[i]<=tp
    if slh:ex=sl;reason='sl'
    elif tph:ex=tp;reason='tp'
    elif g75 and ((x>=avg+.12) if pos['d']==1 else (x<=avg-.12)):pos['a']=1;pos['pk']=x;pos['la']=avg+.12 if pos['d']==1 else avg-.12
   if ex is None and pos['a']:
    if pos['d']==1:
     pos['pk']=max(pos['pk'],x)
     while pos['n']<10 and x>=pos['la']+.025:pos['la']+=.025;pos['e'].append((pos['la'],lot));pos['n']+=1;adds+=1
     if x<=pos['pk']-.20:ex=x;reason='g75'
    else:
     pos['pk']=min(pos['pk'],x)
     while pos['n']<10 and x<=pos['la']-.025:pos['la']-=.025;pos['e'].append((pos['la'],lot));pos['n']+=1;adds+=1
     if x>=pos['pk']+.20:ex=x;reason='g75'
   if ex is not None:
    pnl=sum(((ex-p) if pos['d']==1 else (p-ex))*v*100 for p,v in pos['e']);eq+=pnl;peak=max(peak,eq);tr.append((pnl,pos['n'],reason));pos=None;continue
  if pos is None and all(np.isfinite([r1[i-1],r2[i-1],s1[i-1],s2[i-1]])) and r1[i-1]>s1[i-1]:
   reg=1 if r1[i-1]>r2[i-1] and s1[i-1]>s2[i-1] else (-1 if r1[i-1]<r2[i-1] and s1[i-1]<s2[i-1] else 0);k=K((r1[i-1]-s1[i-1])/P);b=(math.sqrt(r1[i-1])-k)**2;s=(math.sqrt(s1[i-1])+k)**2;ods=[]
   if not rf or reg>=0:ods += [(1,b),(1,b-g2*P),(1,b-(g2+g3)*P)]
   if not rf or reg<=0:ods += [(-1,s),(-1,s+g2*P),(-1,s+(g2+g3)*P)]
   v=[]
   for d,p in ods:
    if d==1 and p<C[i-1] and L[i]<=p<=H[i]:v.append((d,p))
    if d==-1 and p>C[i-1] and L[i]<=p<=H[i]:v.append((d,p))
   ds=set(d for d,p in v)
   if len(ds)==1:
    d=next(iter(ds));es=[(p,lot) for dd,p in v if dd==d];base+=len(es);pos={'d':d,'e':es,'a':0,'pk':C[i],'la':sum(p for p,v in es)/len(es),'n':0}
 if not tr:return {}
 t=pd.DataFrame(tr,columns=['p','n','r']);gp=t[t.p>0].p.sum();gl=-t[t.p<0].p.sum();curve=1000+t.p.cumsum();dd=(curve.cummax()-curve)/curve.cummax()*100;biz=len(set(d for d in df.datetime.dt.date if pd.Timestamp(d).weekday()<5));final=1000+t.p.sum();mo=((final/1000)**(21/biz)-1)*100 if final>0 else -100
 return dict(N=len(t),WR=(t.p>0).mean()*100,PF=gp/gl if gl else 9999,Net=t.p.sum(),Final=final,MaxDD_closed=dd.max(),MaxDD_float=mfd,BaseEntries=base,G75Adds=adds,AvgAdds=t.n.mean(),Month21_pct=mo,BusinessDays=biz)
rows=[]
for rf in [False,True]:
 for lot in [.0002,.0003,.0004,.0005,.00075,.001]:
  for g in [False,True]:rows.append(dict(regime_filter=rf,lot=lot,mode='G75' if g else 'BASE',**run(lot,rf,g)))
out=pd.DataFrame(rows);out.to_csv(O/'results.csv',index=False);f=out[(out['mode']=='G75')&(out.MaxDD_float<=5)].sort_values('Month21_pct',ascending=False);f.to_csv(O/'dd5.csv',index=False);s={'rows':N,'start':str(df.datetime.min()),'end':str(df.datetime.max()),'top':f.head(12).to_dict('records')};(O/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))