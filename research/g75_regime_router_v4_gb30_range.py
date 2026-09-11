from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
TR=.12;ADD=.025;REV=.20;MAXL=10;NS=1_000_000_000;MIN=60*NS;M5=5*MIN;M15=15*MIN

def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def load(c):
 cat=ParquetDataCatalog(c);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');q=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);n=len(q);t=np.fromiter((int(z.ts_event) for z in q),np.int64,count=n);b=np.fromiter((f(z.bid_price) for z in q),float,count=n);a=np.fromiter((f(z.ask_price) for z in q),float,count=n);return t,b,a

def to_ns(s):return np.array([pd.Timestamp(x).value for x in pd.to_datetime(s,utc=True)],np.int64)
def bias30(t,path):
 d=pd.read_csv(path);en=to_ns(d.entry_time);ex=to_ns(d.exit_time);side=d.side.to_numpy(np.int8);out=np.zeros(len(t),np.int8)
 for s,e,x in zip(side,en,ex):out[np.searchsorted(t,e,'left'):np.searchsorted(t,max(e,x)+30*MIN,'right')]=s
 return out

def mk(t,b,a,m):
 d=pd.DataFrame({'t':pd.to_datetime(t,unit='ns',utc=True),'x':(b+a)/2}).set_index('t');z=d.x.resample(f'{m}min',label='left',closed='left').ohlc().dropna();pc=z.close.shift();tr=pd.concat([z.high-z.low,(z.high-pc).abs(),(z.low-pc).abs()],axis=1).max(axis=1);z['atr']=tr.rolling(14).mean();up=z.high.diff();dn=-z.low.diff();p=np.where((up>dn)&(up>0),up,0.);mm=np.where((dn>up)&(dn>0),dn,0.);aa=tr.rolling(14).sum();pdi=100*pd.Series(p,index=z.index).rolling(14).sum()/(aa+1e-12);mdi=100*pd.Series(mm,index=z.index).rolling(14).sum()/(aa+1e-12);z['adx']=(100*(pdi-mdi).abs()/(pdi+mdi+1e-12)).rolling(14).mean();chg=z.close.diff().abs();z['er']=z.close.diff(10).abs()/(chg.rolling(10).sum()+1e-12);z['sl']=(z.close-z.close.shift(3))/(z.atr+1e-12);z['bbw']=4*z.close.rolling(20).std()/(z.close.abs()+1e-12);z['bmed']=z.bbw.rolling(20).median();return z
def prep(t,b,a):
 x=mk(t,b,a,5);y=mk(t,b,a,15);x['rh']=x.high.shift(1).rolling(20).max();x['rl']=x.low.shift(1).rolling(20).min();m5={int(k.value):tuple(float(getattr(r,n)) for n in ['adx','er','sl','bbw','bmed','rh','rl','atr']) for k,r in x.iterrows()};m15={int(k.value):tuple(float(getattr(r,n)) for n in ['adx','er','sl','bbw','bmed']) for k,r in y.iterrows()};return m5,m15
def state(t,m5,m15):
 x=m5.get((t//M5)*M5-M5);y=m15.get((t//M15)*M15-M15)
 if x is None or y is None:return 'TRANSITION',None
 a5,e5,s5,w5,wm5,rh,rl,atr=x;a15,e15,s15,w15,wm15=y;v=[a5,e5,s5,w5,wm5,rh,rl,atr,a15,e15,s15,w15,wm15]
 if not all(np.isfinite(q) for q in v):return 'TRANSITION',(rh,rl,atr)
 if (w5>1.35*max(wm5,1e-12) and abs(s5)>.20) or (s15*s5<-.035 and abs(s15)>.10 and abs(s5)>.10):return 'TRANSITION',(rh,rl,atr)
 if s15>=.08 and (e15>=.18 or a15>=18) and s5>=-.16:return 'UP',(rh,rl,atr)
 if s15<=-.08 and (e15>=.18 or a15>=18) and s5<=.16:return 'DOWN',(rh,rl,atr)
 if abs(s15)<.16 and e15<.42 and e5<.50:return 'RANGE',(rh,rl,atr)
 return 'TRANSITION',(rh,rl,atr)
def met(p,mdd):
 p=np.array(p,float);gp=p[p>0].sum() if len(p) else 0.;gl=-p[p<0].sum() if len(p) else 0.;return {'N':len(p),'WR_pct':100*(p>0).sum()/max(1,len(p)),'PF':gp/gl if gl else(999. if gp else 0.),'pnl':float(p.sum()),'gross_win':float(gp),'gross_loss':float(gl),'max_DD_pct':mdd}
def run(t,bid,ask,bias,m5,m15,variant):
 bucket=-1;anchor=None;cm=None;started=False;eng=None;side=0;entries=[];last=None;ext=None;startreg=None;target=stop=None;arm=0;armext=None;pn=[];by={'UP':[],'DOWN':[],'RANGE':[]};real=0.;peak=1000.;mdd=0.;adds=0;blocked=0;starts={k:0 for k in ['UP','DOWN','RANGE','TRANSITION']}
 def close(b,a):
  nonlocal eng,side,entries,last,ext,startreg,target,stop,real
  if eng is None:return
  z=b if side>0 else a;p=sum((z-e)*side for e in entries);pn.append(p);by[startreg].append(p);real+=p;eng=None;side=0;entries=[];last=None;ext=None;startreg=None;target=stop=None
 for i,t0 in enumerate(t):
  tt=int(t0);b=float(bid[i]);a=float(ask[i]);mid=(b+a)/2;reg,rr=state(tt,m5,m15);bk=(tt//NS)//300
  if bucket<0:bucket=bk;anchor=mid
  elif bk!=bucket:
   if eng=='TREND':
    z=b if side>0 else a
    if (z<=ext-REV if side>0 else z>=ext+REV):close(b,a)
   anchor=cm;bucket=bk;started=False
  cm=mid
  if eng=='RANGE':
   z=b if side>0 else a
   if reg!='RANGE' or (side>0 and (z>=target or z<=stop)) or (side<0 and (z<=target or z>=stop)):close(b,a)
  elif eng=='TREND':
   z=b if side>0 else a;ext=max(ext,z) if side>0 else min(ext,z)
   while len(entries)<MAXL:
    tar=last+side*ADD;cross=z>=tar if side>0 else z<=tar
    if not cross:break
    entries.append(a if side>0 else b);last=tar;adds+=1
  if eng is None and rr is not None:
   rh,rl,atr=rr;w=rh-rl
   if reg=='RANGE' and np.isfinite(w) and w>max(.8,2*atr):
    q=(mid-rl)/w
    if q<=.22:
     if arm!=1:arm=1;armext=mid
     armext=min(armext,mid)
     if mid>=armext+max(.08,.10*atr):side=1;eng='RANGE';entries=[a];target=rl+.38*w;stop=rl-max(.12,.06*w);startreg='RANGE';starts['RANGE']+=1;arm=0
    elif q>=.78:
     if arm!=-1:arm=-1;armext=mid
     armext=max(armext,mid)
     if mid<=armext-max(.08,.10*atr):side=-1;eng='RANGE';entries=[b];target=rh-.38*w;stop=rh+max(.12,.06*w);startreg='RANGE';starts['RANGE']+=1;arm=0
    else:arm=0;armext=None
   elif reg in ('UP','DOWN') and not started and anchor is not None:
    cand=1 if mid>=anchor+TR else(-1 if mid<=anchor-TR else 0)
    allow=cand!=0 and cand==int(bias[i])
    if variant=='MATCH_STATE' and cand:allow=allow and cand==(1 if reg=='UP' else -1)
    if allow:side=cand;eng='TREND';entry=a if side>0 else b;entries=[entry];last=entry;ext=b if side>0 else a;startreg=reg;starts[reg]+=1;started=True
    elif cand:blocked+=1
   else:
    if reg=='TRANSITION':arm=0;armext=None
  if eng is not None:
   z=b if side>0 else a;mark=sum((z-e)*side for e in entries);eq=1000+real+mark;peak=max(peak,eq);mdd=max(mdd,max(0.,(peak-eq)/max(peak,1e-9)*100))
 if eng is not None:close(float(bid[-1]),float(ask[-1]))
 r=met(pn,mdd);r.update({'variant':variant,'adds':adds,'blocked':blocked,'starts':starts,'by_engine':{k:met(v,0.) for k,v in by.items()}});return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--events',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);t,bid,ask=load(a.catalog);bi=bias30(t,a.events);m5,m15=prep(t,bid,ask);rows=[]
 for v in ['GB30_TREND','MATCH_STATE']:
  r=run(t,bid,ask,bi,m5,m15,v);r['raw_ticks']=len(t);rows.append(r);print(json.dumps(r))
 (o/'summary.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(o/'summary.csv',index=False)
if __name__=='__main__':main()
