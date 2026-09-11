from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
TRIGGER=.12;ADD=.025;REV=.20;MAXL=10;MIN=60_000_000_000;M5=5*MIN;M15=15*MIN

def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def load(c):
 cat=ParquetDataCatalog(c);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');q=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
 ns=np.fromiter((int(t.ts_event) for t in q),np.int64,count=len(q));bid=np.fromiter((f(t.bid_price) for t in q),float,count=len(q));ask=np.fromiter((f(t.ask_price) for t in q),float,count=len(q));return ns,bid,ask

def mk(ns,bid,ask,m):
 d=pd.DataFrame({'t':pd.to_datetime(ns,unit='ns',utc=True),'x':(bid+ask)/2}).set_index('t');b=d.x.resample(f'{m}min',label='left',closed='left').ohlc().dropna();pc=b.close.shift();tr=pd.concat([b.high-b.low,(b.high-pc).abs(),(b.low-pc).abs()],axis=1).max(axis=1);b['atr']=tr.rolling(14).mean();up=b.high.diff();dn=-b.low.diff();p=np.where((up>dn)&(up>0),up,0.);mm=np.where((dn>up)&(dn>0),dn,0.);a=tr.rolling(14).sum();pdi=100*pd.Series(p,index=b.index).rolling(14).sum()/(a+1e-12);mdi=100*pd.Series(mm,index=b.index).rolling(14).sum()/(a+1e-12);b['adx']=(100*(pdi-mdi).abs()/(pdi+mdi+1e-12)).rolling(14).mean();chg=b.close.diff().abs();b['er']=b.close.diff(10).abs()/(chg.rolling(10).sum()+1e-12);b['sl']=(b.close-b.close.shift(3))/(b.atr+1e-12);b['bbw']=4*b.close.rolling(20).std()/(b.close.abs()+1e-12);b['bmed']=b.bbw.rolling(20).median();return b

def prep(ns,bid,ask):
 a=mk(ns,bid,ask,5);b=mk(ns,bid,ask,15);a['rh']=a.high.shift(1).rolling(20).max();a['rl']=a.low.shift(1).rolling(20).min();
 m5={int(t.value):tuple(float(getattr(r,k)) for k in ['adx','er','sl','bbw','bmed','rh','rl']) for t,r in a.iterrows()};m15={int(t.value):tuple(float(getattr(r,k)) for k in ['adx','er','sl','bbw','bmed']) for t,r in b.iterrows()};return m5,m15

def state(t,m5,m15):
 x=m5.get((t//M5)*M5-M5);y=m15.get((t//M15)*M15-M15)
 if x is None or y is None:return 'TRANSITION',None
 a5,e5,s5,w5,wm5,rh,rl=x;a15,e15,s15,w15,wm15=y;v=[a5,e5,s5,w5,wm5,rh,rl,a15,e15,s15,w15,wm15]
 if not all(np.isfinite(z) for z in v):return 'TRANSITION',(rh,rl)
 expand=w5>1.35*max(wm5,1e-12) and abs(s5)>.20;conflict=(s15*s5<-.035 and abs(s15)>.10 and abs(s5)>.10)
 if expand or conflict:return 'TRANSITION',(rh,rl)
 if s15>=.08 and (e15>=.18 or a15>=18) and s5>=-.16:return 'UP',(rh,rl)
 if s15<=-.08 and (e15>=.18 or a15>=18) and s5<=.16:return 'DOWN',(rh,rl)
 if abs(s15)<.16 and e15<.42 and e5<.50:return 'RANGE',(rh,rl)
 return 'TRANSITION',(rh,rl)

def run(ns,bid,ask,mode,m5,m15):
 bucket=-1;anchor=None;cm=None;started=False;active=False;side=0;entries=[];last=None;ext=None;startreg=None
 real=0.;peak=1000.;mdd=0.;cyc=win=adds=maxl=0;gw=gl=0.;blocked=0;cnt={k:0 for k in ['UP','DOWN','RANGE','TRANSITION']};starts={k:0 for k in cnt};rp={k:0. for k in cnt};rn={k:0 for k in cnt};rw={k:0 for k in cnt}
 def mark(b,a):
  if not active:return 0.
  p=b if side>0 else a;return sum((p-e)*side for e in entries)
 def dd(b,a):
  nonlocal peak,mdd
  eq=1000+real+mark(b,a);peak=max(peak,eq);mdd=max(mdd,max(0.,(peak-eq)/max(peak,1e-9)*100))
 def close(b,a):
  nonlocal active,side,entries,last,ext,real,cyc,win,gw,gl,startreg
  if not active:return
  p=(b if side>0 else a);x=sum((p-e)*side for e in entries);real+=x;cyc+=1;rn[startreg]+=1;rp[startreg]+=x
  if x>0:win+=1;gw+=x;rw[startreg]+=1
  elif x<0:gl+=abs(x)
  active=False;side=0;entries=[];last=None;ext=None;startreg=None
 for i,t0 in enumerate(ns):
  t=int(t0);b=float(bid[i]);a=float(ask[i]);m=(b+a)/2;reg,rr=state(t,m5,m15);cnt[reg]+=1;bk=(t//1_000_000_000)//300
  if bucket<0:bucket=bk;anchor=m
  elif bk!=bucket:
   if active:
    z=b if side>0 else a
    if (z<=ext-REV if side>0 else z>=ext+REV):close(b,a)
   anchor=cm;bucket=bk;started=False
  cm=m
  if not active and not started and anchor is not None:
   cand=1 if m>=anchor+TRIGGER else(-1 if m<=anchor-TRIGGER else 0)
   if cand:
    allow=True
    if mode=='ROUTER':
     allow=False
     if reg=='UP':allow=cand==1
     elif reg=='DOWN':allow=cand==-1
     elif reg=='RANGE' and rr:
      rh,rl=rr;w=rh-rl
      if np.isfinite(w) and w>0:
       q=(m-rl)/w
       if q<=.30:allow=cand==1
       elif q>=.70:allow=cand==-1
     if not allow:blocked+=1
    if allow:
     side=cand;entry=a if side>0 else b;active=True;entries=[entry];last=entry;ext=b if side>0 else a;started=True;startreg=reg;starts[reg]+=1;maxl=max(maxl,1)
  if active:
   p=b if side>0 else a;ext=max(ext,p) if side>0 else min(ext,p);cap=MAXL if mode=='PURE' else(10 if reg in ('UP','DOWN') else(4 if reg=='RANGE' else 2))
   while len(entries)<cap:
    tar=last+side*ADD;cross=p>=tar if side>0 else p<=tar
    if not cross:break
    entries.append(a if side>0 else b);last=tar;adds+=1;maxl=max(maxl,len(entries))
   dd(b,a)
 if active:close(float(bid[-1]),float(ask[-1]))
 pf=gw/gl if gl else(999. if gw else 0.);diag={k:{'N':int(rn[k]),'WR':100*rw[k]/max(1,rn[k]),'pnl':rp[k]} for k in cnt}
 return {'mode':mode,'cycles':cyc,'WR_pct':100*win/max(1,cyc),'PF':pf,'realized_usd_0p01lot_equiv':real,'return_pct_on_1000':real/10,'max_DD_pct':mdd,'adds':adds,'max_layers':maxl,'gross_win':gw,'gross_loss':gl,'blocked_starts':blocked,'regime_tick_counts':cnt,'starts_by_regime':starts,'pnl_by_start_regime':diag,'core':{'trigger':TRIGGER,'add':ADD,'reversal':REV,'max_layers':MAXL}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ns,bid,ask=load(a.catalog);m5,m15=prep(ns,bid,ask);rows=[]
 for mode in ['PURE','ROUTER']:
  r=run(ns,bid,ask,mode,m5,m15);r['raw_ticks']=len(ns);rows.append(r);print(json.dumps(r))
 (out/'summary.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
if __name__=='__main__':main()
