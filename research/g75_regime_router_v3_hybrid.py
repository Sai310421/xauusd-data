from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
TRIGGER=.12;ADD=.025;REV=.20;MAXL=10;MIN=60_000_000_000;M5=5*MIN;M15=15*MIN

def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def load(c):
 cat=ParquetDataCatalog(c);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');q=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);n=len(q)
 ns=np.fromiter((int(t.ts_event) for t in q),np.int64,count=n);bid=np.fromiter((f(t.bid_price) for t in q),float,count=n);ask=np.fromiter((f(t.ask_price) for t in q),float,count=n);return ns,bid,ask

def mk(ns,bid,ask,m):
 d=pd.DataFrame({'t':pd.to_datetime(ns,unit='ns',utc=True),'x':(bid+ask)/2}).set_index('t');b=d.x.resample(f'{m}min',label='left',closed='left').ohlc().dropna();pc=b.close.shift();tr=pd.concat([b.high-b.low,(b.high-pc).abs(),(b.low-pc).abs()],axis=1).max(axis=1);b['atr']=tr.rolling(14).mean();up=b.high.diff();dn=-b.low.diff();p=np.where((up>dn)&(up>0),up,0.);mm=np.where((dn>up)&(dn>0),dn,0.);a=tr.rolling(14).sum();pdi=100*pd.Series(p,index=b.index).rolling(14).sum()/(a+1e-12);mdi=100*pd.Series(mm,index=b.index).rolling(14).sum()/(a+1e-12);b['adx']=(100*(pdi-mdi).abs()/(pdi+mdi+1e-12)).rolling(14).mean();chg=b.close.diff().abs();b['er']=b.close.diff(10).abs()/(chg.rolling(10).sum()+1e-12);b['sl']=(b.close-b.close.shift(3))/(b.atr+1e-12);b['bbw']=4*b.close.rolling(20).std()/(b.close.abs()+1e-12);b['bmed']=b.bbw.rolling(20).median();return b

def prep(ns,bid,ask):
 a=mk(ns,bid,ask,5);b=mk(ns,bid,ask,15);a['rh']=a.high.shift(1).rolling(20).max();a['rl']=a.low.shift(1).rolling(20).min();a['atr5']=a.atr
 m5={int(t.value):tuple(float(getattr(r,k)) for k in ['adx','er','sl','bbw','bmed','rh','rl','atr5']) for t,r in a.iterrows()};m15={int(t.value):tuple(float(getattr(r,k)) for k in ['adx','er','sl','bbw','bmed']) for t,r in b.iterrows()};return m5,m15

def state(t,m5,m15):
 x=m5.get((t//M5)*M5-M5);y=m15.get((t//M15)*M15-M15)
 if x is None or y is None:return 'TRANSITION',None
 a5,e5,s5,w5,wm5,rh,rl,atr5=x;a15,e15,s15,w15,wm15=y;v=[a5,e5,s5,w5,wm5,rh,rl,atr5,a15,e15,s15,w15,wm15]
 if not all(np.isfinite(z) for z in v):return 'TRANSITION',(rh,rl,atr5)
 expand=w5>1.35*max(wm5,1e-12) and abs(s5)>.20;conflict=(s15*s5<-.035 and abs(s15)>.10 and abs(s5)>.10)
 if expand or conflict:return 'TRANSITION',(rh,rl,atr5)
 if s15>=.08 and (e15>=.18 or a15>=18) and s5>=-.16:return 'UP',(rh,rl,atr5)
 if s15<=-.08 and (e15>=.18 or a15>=18) and s5<=.16:return 'DOWN',(rh,rl,atr5)
 if abs(s15)<.16 and e15<.42 and e5<.50:return 'RANGE',(rh,rl,atr5)
 return 'TRANSITION',(rh,rl,atr5)

def metrics(pnls,peakdd):
 p=np.array(pnls,float);gp=p[p>0].sum() if len(p) else 0.;gl=-p[p<0].sum() if len(p) else 0.;return {'N':len(p),'WR_pct':100*(p>0).sum()/max(1,len(p)),'PF':gp/gl if gl>0 else(999. if gp>0 else 0.),'pnl':float(p.sum()),'gross_win':float(gp),'gross_loss':float(gl),'max_DD_pct':peakdd}

def run_pure(ns,bid,ask):
 bucket=-1;anchor=None;cm=None;started=False;active=False;side=0;entries=[];last=None;ext=None;pn=[];peak=1000.;real=0.;mdd=0.;adds=0
 for i,t0 in enumerate(ns):
  t=int(t0);b=float(bid[i]);a=float(ask[i]);m=(b+a)/2;bk=(t//1_000_000_000)//300
  if bucket<0:bucket=bk;anchor=m
  elif bk!=bucket:
   if active:
    z=b if side>0 else a
    if (z<=ext-REV if side>0 else z>=ext+REV):
     x=sum(((b if side>0 else a)-e)*side for e in entries);pn.append(x);real+=x;active=False;entries=[]
   anchor=cm;bucket=bk;started=False
  cm=m
  if not active and not started and anchor is not None:
   cand=1 if m>=anchor+TRIGGER else(-1 if m<=anchor-TRIGGER else 0)
   if cand:side=cand;entry=a if side>0 else b;active=True;entries=[entry];last=entry;ext=b if side>0 else a;started=True
  if active:
   p=b if side>0 else a;ext=max(ext,p) if side>0 else min(ext,p)
   while len(entries)<MAXL:
    tar=last+side*ADD;cross=p>=tar if side>0 else p<=tar
    if not cross:break
    entries.append(a if side>0 else b);last=tar;adds+=1
   mark=sum((p-e)*side for e in entries);eq=1000+real+mark;peak=max(peak,eq);mdd=max(mdd,max(0.,(peak-eq)/max(peak,1e-9)*100))
 if active:
  x=sum(((float(bid[-1]) if side>0 else float(ask[-1]))-e)*side for e in entries);pn.append(x)
 r=metrics(pn,mdd);r.update({'mode':'PURE','adds':adds});return r

def run_hybrid(ns,bid,ask,m5,m15):
 # One engine at a time: trend=G75 momentum; range=boundary reclaim to midpoint; transition=flat.
 bucket=-1;anchor=None;cm=None;started=False;eng=None;side=0;entries=[];last=None;ext=None;startreg=None;target=None;stop=None
 arm=0;arm_ext=None;pn=[];pn_by={'UP':[],'DOWN':[],'RANGE':[]};real=0.;peak=1000.;mdd=0.;adds=0;blocked=0;starts={k:0 for k in ['UP','DOWN','RANGE','TRANSITION']}
 def close(b,a):
  nonlocal eng,side,entries,last,ext,startreg,target,stop,real
  if eng is None:return
  p=b if side>0 else a;x=sum((p-e)*side for e in entries);pn.append(x);pn_by[startreg].append(x);real+=x;eng=None;side=0;entries=[];last=None;ext=None;startreg=None;target=stop=None
 for i,t0 in enumerate(ns):
  t=int(t0);b=float(bid[i]);a=float(ask[i]);m=(b+a)/2;reg,rr=state(t,m5,m15);bk=(t//1_000_000_000)//300
  if bucket<0:bucket=bk;anchor=m
  elif bk!=bucket:
   if eng=='TREND':
    z=b if side>0 else a
    if (z<=ext-REV if side>0 else z>=ext+REV):close(b,a)
   anchor=cm;bucket=bk;started=False
  cm=m
  if eng=='RANGE':
   p=b if side>0 else a
   if reg!='RANGE': close(b,a)
   elif (side>0 and (p>=target or p<=stop)) or (side<0 and (p<=target or p>=stop)): close(b,a)
  elif eng=='TREND':
   p=b if side>0 else a;ext=max(ext,p) if side>0 else min(ext,p)
   while len(entries)<10:
    tar=last+side*ADD;cross=p>=tar if side>0 else p<=tar
    if not cross:break
    entries.append(a if side>0 else b);last=tar;adds+=1
  if eng is None and rr is not None:
   rh,rl,atr5=rr;w=rh-rl
   if reg=='RANGE' and np.isfinite(w) and w>max(.8,2*atr5):
    q=(m-rl)/w
    if q<=.22:
     if arm!=1:arm=1;arm_ext=m
     arm_ext=min(arm_ext,m)
     if m>=arm_ext+max(.08,.10*atr5):
      side=1;eng='RANGE';entries=[a];target=(rh+rl)/2;stop=rl-max(.12,.08*w);startreg='RANGE';starts['RANGE']+=1;arm=0
    elif q>=.78:
     if arm!=-1:arm=-1;arm_ext=m
     arm_ext=max(arm_ext,m)
     if m<=arm_ext-max(.08,.10*atr5):
      side=-1;eng='RANGE';entries=[b];target=(rh+rl)/2;stop=rh+max(.12,.08*w);startreg='RANGE';starts['RANGE']+=1;arm=0
    else: arm=0;arm_ext=None
   elif reg in ('UP','DOWN') and not started and anchor is not None:
    cand=1 if m>=anchor+TRIGGER else(-1 if m<=anchor-TRIGGER else 0);want=1 if reg=='UP' else -1
    if cand==want:
     side=cand;entry=a if side>0 else b;eng='TREND';entries=[entry];last=entry;ext=b if side>0 else a;startreg=reg;starts[reg]+=1;started=True
    elif cand:blocked+=1
   elif reg=='TRANSITION':arm=0;arm_ext=None
  if eng is not None:
   p=b if side>0 else a;mark=sum((p-e)*side for e in entries);eq=1000+real+mark;peak=max(peak,eq);mdd=max(mdd,max(0.,(peak-eq)/max(peak,1e-9)*100))
 if eng is not None:close(float(bid[-1]),float(ask[-1]))
 r=metrics(pn,mdd);r.update({'mode':'HYBRID_V3','adds':adds,'blocked_trend_starts':blocked,'starts_by_regime':starts,'by_engine':{k:metrics(v,0.) for k,v in pn_by.items()}});return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ns,bid,ask=load(a.catalog);m5,m15=prep(ns,bid,ask);rows=[run_pure(ns,bid,ask),run_hybrid(ns,bid,ask,m5,m15)]
 for r in rows:r['raw_ticks']=len(ns);print(json.dumps(r))
 (out/'summary.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
if __name__=='__main__':main()
