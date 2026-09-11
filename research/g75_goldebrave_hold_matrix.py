from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
TR=.12;ADD=.025;REV=.20;MAXL=10;TF=300;NS=1_000_000_000;MIN=60*NS

def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def load(c):
 cat=ParquetDataCatalog(c);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');q=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);n=len(q)
 t=np.fromiter((int(z.ts_event) for z in q),np.int64,count=n);b=np.fromiter((f(z.bid_price) for z in q),float,count=n);a=np.fromiter((f(z.ask_price) for z in q),float,count=n);return t,b,a

def events(path):
 d=pd.read_csv(path);d['en']=pd.to_datetime(d.entry_time,utc=True).astype('int64');d['ex']=pd.to_datetime(d.exit_time,utc=True).astype('int64');return d.sort_values('en')

def make_bias(t,d,mode):
 out=np.zeros(len(t),np.int8);en=d.en.to_numpy(np.int64);ex=d.ex.to_numpy(np.int64);side=d.side.to_numpy(np.int8)
 if mode=='UNTIL_OPP':
  j=0;cur=0
  for i,x in enumerate(t):
   while j<len(en) and en[j]<=x:cur=int(side[j]);j+=1
   out[i]=cur
  return out
 hold=int(mode)*MIN
 # active interval plus hold after exit; latest event wins
 for s,e,x in zip(side,en,ex):
  lo=np.searchsorted(t,e,'left');hi=np.searchsorted(t,max(x,e)+hold,'right');out[lo:hi]=s
 return out

def run(t,bid,ask,bias):
 bucket=-1;anchor=None;cm=None;started=False;active=False;side=0;entries=[];last=None;ext=None;pn=[];real=0.;peak=1000.;mdd=0.;adds=0;blocked=0
 for i,x0 in enumerate(t):
  x=int(x0);b=float(bid[i]);a=float(ask[i]);m=(b+a)/2;bk=(x//NS)//TF
  if bucket<0:bucket=bk;anchor=m
  elif bk!=bucket:
   if active:
    z=b if side>0 else a
    if (z<=ext-REV if side>0 else z>=ext+REV):
     p=sum((z-e)*side for e in entries);pn.append(p);real+=p;active=False;entries=[]
   anchor=cm;bucket=bk;started=False
  cm=m
  if not active and not started and anchor is not None:
   cand=1 if m>=anchor+TR else(-1 if m<=anchor-TR else 0)
   if cand:
    if bias[i]==cand:
     side=cand;entry=a if side>0 else b;active=True;entries=[entry];last=entry;ext=b if side>0 else a;started=True
    else:blocked+=1
  if active:
   p=b if side>0 else a;ext=max(ext,p) if side>0 else min(ext,p)
   while len(entries)<MAXL:
    tar=last+side*ADD;cross=p>=tar if side>0 else p<=tar
    if not cross:break
    entries.append(a if side>0 else b);last=tar;adds+=1
   mark=sum((p-e)*side for e in entries);eq=1000+real+mark;peak=max(peak,eq);mdd=max(mdd,max(0.,(peak-eq)/max(peak,1e-9)*100))
 if active:
  z=float(bid[-1]) if side>0 else float(ask[-1]);p=sum((z-e)*side for e in entries);pn.append(p)
 p=np.array(pn,float);gp=p[p>0].sum() if len(p) else 0.;gl=-p[p<0].sum() if len(p) else 0.
 return {'N':len(p),'WR_pct':100*(p>0).sum()/max(1,len(p)),'PF':gp/gl if gl else(999. if gp else 0.),'pnl':float(p.sum()),'max_DD_pct':mdd,'adds':adds,'blocked':blocked,'bias_tick_pct':100*np.count_nonzero(bias)/len(bias)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--events',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t,bid,ask=load(a.catalog);d=events(a.events);rows=[]
 for mode in ['0','30','60','120','UNTIL_OPP']:
  bias=make_bias(t,d,mode);r=run(t,bid,ask,bias);r['hold']=mode;r['raw_ticks']=len(t);rows.append(r);print(json.dumps(r))
 pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);(out/'summary.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
