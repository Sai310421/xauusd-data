from __future__ import annotations
import argparse,json,math
from pathlib import Path
from collections import deque
import pandas as pd, numpy as np
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

TRIGGER=.12; ADD=.025; REVERSAL=.20; MAX_LAYERS=10
TF_SEC={'M1':60,'M5':300,'M15':900}

def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)

def load_ticks(catalog):
    cat=ParquetDataCatalog(catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    ns=np.fromiter((int(t.ts_event) for t in ticks),dtype=np.int64,count=len(ticks))
    bid=np.fromiter((f(t.bid_price) for t in ticks),dtype=float,count=len(ticks))
    ask=np.fromiter((f(t.ask_price) for t in ticks),dtype=float,count=len(ticks))
    return ns,bid,ask

def load_supervisor(path):
    d=pd.read_csv(path); events=[]
    for r in d.itertuples(index=False):
        s=int(r.side); a=int(pd.Timestamp(r.entry_time).value); b=int(pd.Timestamp(r.exit_time).value)
        events.append((a,1,s)); events.append((b,-1,s))
    events.sort(); return events

def run(ns,bid,ask,tf_sec,mode,events):
    bucket=-1; anchor=None; close_mid=None; started=False
    active=False; side=0; entries=[]; last_add=None; extreme=None
    realized=0.; peak=1000.; maxdd=0.; cycles=wins=losses=adds=maxlayers=0; gw=gl=0.
    last_bid=last_ask=None; ei=0; long_n=short_n=0; starts_allowed=starts_rejected=0
    hour_key=-1; hour_hi=hour_lo=None; ranges=deque(maxlen=20); regime='NEUTRAL'
    regime_counts={'TREND':0,'NEUTRAL':0,'RANGE':0}; sup_ticks=0
    def sup_side():
        if long_n>0 and short_n==0:return 1
        if short_n>0 and long_n==0:return -1
        return 0
    def mark(b,a):
        if not active:return 0.
        px=b if side>0 else a
        return sum((px-e)*side for e in entries)
    def dd(b,a):
        nonlocal peak,maxdd
        eq=1000.+realized+mark(b,a); peak=max(peak,eq)
        x=max(0.,(peak-eq)/max(peak,1e-9)*100); maxdd=max(maxdd,x)
    def close(b,a):
        nonlocal active,side,entries,last_add,extreme,realized,cycles,wins,losses,gw,gl
        if not active:return
        px=b if side>0 else a; p=sum((px-e)*side for e in entries); realized+=p; cycles+=1
        if p>0:wins+=1;gw+=p
        elif p<0:losses+=1;gl+=abs(p)
        active=False;side=0;entries=[];last_add=None;extreme=None
    for i in range(len(ns)):
        t=int(ns[i]); b=float(bid[i]); a=float(ask[i]); m=(b+a)/2; last_bid=b;last_ask=a
        while ei<len(events) and events[ei][0]<=t:
            _,typ,s=events[ei]
            if s>0: long_n += typ
            else: short_n += typ
            ei+=1
        ss=sup_side()
        if ss:sup_ticks+=1
        hk=(t//1_000_000_000)//3600
        if hk!=hour_key:
            if hour_key>=0 and hour_hi is not None:
                r=hour_hi-hour_lo
                if len(ranges)>=5:
                    med=float(np.median(np.asarray(ranges)))
                    regime='TREND' if r>1.25*med else ('RANGE' if r<.75*med else 'NEUTRAL')
                ranges.append(r)
            hour_key=hk;hour_hi=m;hour_lo=m
        else:
            hour_hi=max(hour_hi,m);hour_lo=min(hour_lo,m)
        buck=(t//1_000_000_000)//tf_sec
        if bucket<0:
            bucket=buck;anchor=m
        elif buck!=bucket:
            if active:
                cm=b if side>0 else a
                rev=(cm<=extreme-REVERSAL) if side>0 else (cm>=extreme+REVERSAL)
                if rev: close(b,a)
            anchor=close_mid; bucket=buck; started=False
        close_mid=m
        if not active and not started and anchor is not None:
            cand=1 if m>=anchor+TRIGGER else (-1 if m<=anchor-TRIGGER else 0)
            if cand:
                allow=True
                if mode!='PURE':
                    allow=(ss==cand)
                    if allow: starts_allowed+=1
                    else: starts_rejected+=1
                if allow:
                    side=cand; entry=a if side>0 else b
                    active=True; entries=[entry]; last_add=entry; extreme=b if side>0 else a; started=True; maxlayers=max(maxlayers,1)
        if active:
            px=b if side>0 else a; extreme=max(extreme,px) if side>0 else min(extreme,px)
            cap=MAX_LAYERS
            if mode=='DIR_REGIME': cap=10 if regime=='TREND' else (5 if regime=='RANGE' else 7)
            while len(entries)<cap:
                target=last_add+side*ADD; cross=(px>=target) if side>0 else (px<=target)
                if not cross: break
                fill=a if side>0 else b; entries.append(fill); last_add=target; adds+=1; maxlayers=max(maxlayers,len(entries))
            dd(b,a)
        regime_counts[regime]+=1
    if active and last_bid is not None: close(last_bid,last_ask)
    pf=gw/gl if gl else (999. if gw else 0.)
    return {'mode':mode,'tf_sec':tf_sec,'cycles':cycles,'WR_pct':100*wins/max(1,cycles),'PF':pf,'realized_usd_0p01lot_equiv':realized,'return_pct_on_1000':realized/10,'max_DD_pct':maxdd,'adds':adds,'max_layers':maxlayers,'gross_win':gw,'gross_loss':gl,'starts_allowed':starts_allowed,'starts_rejected':starts_rejected,'supervisor_tick_coverage_pct':100*sup_ticks/max(1,len(ns)),'regime_tick_counts':regime_counts}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--supervisor',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ns,bid,ask=load_ticks(a.catalog); events=load_supervisor(a.supervisor); rows=[]
    for tf,sec in TF_SEC.items():
        for mode in ['PURE','DIR','DIR_REGIME']:
            r=run(ns,bid,ask,sec,mode,events);r['tf']=tf;r['raw_ticks']=len(ns);rows.append(r);print(json.dumps(r))
    (out/'summary.json').write_text(json.dumps(rows,indent=2)); pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
if __name__=='__main__': main()
