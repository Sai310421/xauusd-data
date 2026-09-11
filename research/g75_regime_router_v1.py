from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

TRIGGER=.12; ADD=.025; REVERSAL=.20; MAX_LAYERS=10
M1_NS=60_000_000_000; M5_NS=5*M1_NS; M15_NS=15*M1_NS

def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def load_ticks(catalog):
    cat=ParquetDataCatalog(catalog); inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    q=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    ns=np.fromiter((int(t.ts_event) for t in q),dtype=np.int64,count=len(q)); bid=np.fromiter((f(t.bid_price) for t in q),float,count=len(q)); ask=np.fromiter((f(t.ask_price) for t in q),float,count=len(q))
    return ns,bid,ask

def bars(ns,bid,ask,mins):
    d=pd.DataFrame({'t':pd.to_datetime(ns,unit='ns',utc=True),'m':(bid+ask)/2}).set_index('t')
    b=d.m.resample(f'{mins}min',label='left',closed='left').ohlc().dropna(); return b

def adx(b,n=14):
    up=b.high.diff(); dn=-b.low.diff(); plus=np.where((up>dn)&(up>0),up,0.); minus=np.where((dn>up)&(dn>0),dn,0.)
    tr=pd.concat([b.high-b.low,(b.high-b.close.shift()).abs(),(b.low-b.close.shift()).abs()],axis=1).max(axis=1); a=tr.rolling(n).sum()
    p=100*pd.Series(plus,index=b.index).rolling(n).sum()/(a+1e-12); m=100*pd.Series(minus,index=b.index).rolling(n).sum()/(a+1e-12)
    return (100*(p-m).abs()/(p+m+1e-12)).rolling(n).mean()

def features(ns,bid,ask):
    m5=bars(ns,bid,ask,5); m15=bars(ns,bid,ask,15)
    for b in (m5,m15):
        b['tr']=pd.concat([b.high-b.low,(b.high-b.close.shift()).abs(),(b.low-b.close.shift()).abs()],axis=1).max(axis=1); b['atr']=b.tr.rolling(14).mean(); b['adx']=adx(b,14)
        ch=b.close.diff().abs(); b['er']=b.close.diff(10).abs()/(ch.rolling(10).sum()+1e-12); b['slope']=(b.close-b.close.shift(3))/(b.atr+1e-12)
        b['bbw']=4*b.close.rolling(20).std()/(b.close.abs()+1e-12); b['bbw_med']=b.bbw.rolling(20).median()
    m5['rh']=m5.high.shift(1).rolling(20).max(); m5['rl']=m5.low.shift(1).rolling(20).min()
    return m5,m15

def make_maps(m5,m15):
    a={int(t.value):(float(r.adx),float(r.er),float(r.slope),float(r.bbw),float(r.bbw_med),float(r.rh),float(r.rl),float(r.close)) for t,r in m5.iterrows()}
    b={int(t.value):(float(r.adx),float(r.er),float(r.slope),float(r.bbw),float(r.bbw_med),float(r.close)) for t,r in m15.iterrows()}; return a,b

def classify(t,m5map,m15map):
    k5=(t//M5_NS)*M5_NS-M5_NS; k15=(t//M15_NS)*M15_NS-M15_NS
    x=m5map.get(k5); y=m15map.get(k15)
    if x is None or y is None:return 'TRANSITION',None
    a5,e5,s5,w5,wm5,rh,rl,c5=x; a15,e15,s15,w15,wm15,c15=y
    vals=[a5,e5,s5,w5,wm5,rh,rl,a15,e15,s15,w15,wm15]
    if not all(np.isfinite(z) for z in vals):return 'TRANSITION',(rh,rl)
    up=(a15>=22 and e15>=.34 and s15>=.18 and s5>=-.10)
    dn=(a15>=22 and e15>=.34 and s15<=-.18 and s5<=.10)
    rng=(a15<=20 and e15<=.32 and a5<=22 and e5<=.38)
    expand=(w5>1.28*max(wm5,1e-12) and a5>a15)
    if expand:return 'TRANSITION',(rh,rl)
    if up:return 'UP',(rh,rl)
    if dn:return 'DOWN',(rh,rl)
    if rng:return 'RANGE',(rh,rl)
    return 'TRANSITION',(rh,rl)

def run(ns,bid,ask,mode):
    bucket=-1;anchor=None;close_mid=None;started=False;active=False;side=0;entries=[];last_add=None;extreme=None
    realized=0.;peak=1000.;maxdd=0.;cycles=wins=adds=maxlayers=0;gw=gl=0.;blocked=0;regcnt={k:0 for k in ['UP','DOWN','RANGE','TRANSITION']}; starts={k:0 for k in regcnt}
    m5,m15=features(ns,bid,ask); m5map,m15map=make_maps(m5,m15)
    def mark(b,a):
        if not active:return 0.
        px=b if side>0 else a; return sum((px-e)*side for e in entries)
    def dd(b,a):
        nonlocal peak,maxdd
        eq=1000+realized+mark(b,a);peak=max(peak,eq);d=max(0.,(peak-eq)/max(peak,1e-9)*100);maxdd=max(maxdd,d)
    def close(b,a):
        nonlocal active,side,entries,last_add,extreme,realized,cycles,wins,gw,gl
        if not active:return
        px=b if side>0 else a;p=sum((px-e)*side for e in entries);realized+=p;cycles+=1
        if p>0:wins+=1;gw+=p
        elif p<0:gl+=abs(p)
        active=False;side=0;entries=[];last_add=None;extreme=None
    for i,t0 in enumerate(ns):
        t=int(t0);b=float(bid[i]);a=float(ask[i]);m=(b+a)/2;reg,rr=classify(t,m5map,m15map);regcnt[reg]+=1
        buck=(t//1_000_000_000)//300
        if bucket<0:bucket=buck;anchor=m
        elif buck!=bucket:
            if active:
                cm=b if side>0 else a; rev=(cm<=extreme-REVERSAL) if side>0 else (cm>=extreme+REVERSAL)
                if rev:close(b,a)
            anchor=close_mid;bucket=buck;started=False
        close_mid=m
        if not active and not started and anchor is not None:
            cand=1 if m>=anchor+TRIGGER else (-1 if m<=anchor-TRIGGER else 0)
            if cand:
                allow=True; use=cand
                if mode=='ROUTER':
                    allow=False
                    if reg=='UP': allow=(cand==1)
                    elif reg=='DOWN': allow=(cand==-1)
                    elif reg=='RANGE' and rr is not None:
                        rh,rl=rr; width=rh-rl
                        if np.isfinite(width) and width>0:
                            q=(m-rl)/width
                            if q<=.20: use=1;allow=(cand==1)
                            elif q>=.80: use=-1;allow=(cand==-1)
                    if not allow: blocked+=1
                if allow:
                    side=use;entry=a if side>0 else b;active=True;entries=[entry];last_add=entry;extreme=b if side>0 else a;started=True;maxlayers=max(maxlayers,1);starts[reg]+=1
        if active:
            px=b if side>0 else a;extreme=max(extreme,px) if side>0 else min(extreme,px)
            cap=MAX_LAYERS
            if mode=='ROUTER': cap=10 if reg in ('UP','DOWN') else (4 if reg=='RANGE' else 2)
            while len(entries)<cap:
                target=last_add+side*ADD;cross=px>=target if side>0 else px<=target
                if not cross:break
                fill=a if side>0 else b;entries.append(fill);last_add=target;adds+=1;maxlayers=max(maxlayers,len(entries))
            dd(b,a)
    if active:close(float(bid[-1]),float(ask[-1]))
    pf=gw/gl if gl else (999. if gw else 0.)
    return {'mode':mode,'cycles':cycles,'WR_pct':100*wins/max(1,cycles),'PF':pf,'realized_usd_0p01lot_equiv':realized,'return_pct_on_1000':realized/10,'max_DD_pct':maxdd,'adds':adds,'max_layers':maxlayers,'gross_win':gw,'gross_loss':gl,'blocked_starts':blocked,'regime_tick_counts':regcnt,'starts_by_regime':starts,'core':{'trigger':TRIGGER,'add':ADD,'reversal':REVERSAL,'max_layers':MAX_LAYERS}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ns,bid,ask=load_ticks(a.catalog);rows=[]
    for mode in ['PURE','ROUTER']:
        r=run(ns,bid,ask,mode);r['raw_ticks']=len(ns);rows.append(r);print(json.dumps(r))
    (out/'summary.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
if __name__=='__main__':main()
