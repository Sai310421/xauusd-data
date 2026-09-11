from __future__ import annotations
import argparse, json, math, bisect
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

EPS=1e-12

@dataclass
class Bar:
    start_ns:int; end_ns:int; o:float; h:float; l:float; c:float; first_idx:int; last_idx:int


def load_ticks(catalog,max_ticks=0):
    cat=ParquetDataCatalog(catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    if not ticks: raise SystemExit('no XAUUSD QuoteTicks')
    if max_ticks and len(ticks)>max_ticks: ticks=ticks[:max_ticks]
    return ticks


def bars15(ticks):
    out=[]; bucket=None; cur=None
    for idx,q in enumerate(ticks):
        ns=int(q.ts_event); sec=ns//1_000_000_000; b=sec//15
        p=float(q.bid_price)
        if bucket is None or b!=bucket:
            if cur: out.append(cur)
            bucket=b; start=b*15*1_000_000_000
            cur=Bar(start,start+15_000_000_000,p,p,p,p,idx,idx)
        else:
            cur.h=max(cur.h,p); cur.l=min(cur.l,p); cur.c=p; cur.last_idx=idx
    if cur: out.append(cur)
    return out


def rng(b): return max(b.h-b.l,EPS)
def body_ratio(b): return abs(b.c-b.o)/rng(b)
def close_loc(b): return (b.c-b.l)/rng(b)


def cisd_dir(hist,cur,lookback=12):
    lb=next((b for b in reversed(hist[-lookback:]) if b.c<b.o),None)
    lu=next((b for b in reversed(hist[-lookback:]) if b.c>b.o),None)
    up=lb is not None and cur.c>lb.o
    dn=lu is not None and cur.c<lu.o
    if up and not dn:return 1
    if dn and not up:return -1
    return 0


def classify_amd(bars,i,variant):
    # i is candidate D quarter; use A=i-2, M=i-1, D=i. All are 15s states.
    if i<14:return None
    A,M,D=bars[i-2],bars[i-1],bars[i]
    recent=bars[i-12:i-2]
    medr=median([rng(x) for x in recent]) if recent else rng(A)
    accumulation = rng(A) <= 0.80*medr or body_ratio(A)<=0.35

    # Manipulation is a sweep of the accumulation range with close reclaim.
    bull_manip = M.l < A.l and M.c > A.l
    bear_manip = M.h > A.h and M.c < A.h

    # Distribution must leave the manipulation in the opposite direction.
    bull_dist = D.c > M.o and D.c>D.o and (rng(D)>=0.80*medr or body_ratio(D)>=0.45)
    bear_dist = D.c < M.o and D.c<D.o and (rng(D)>=0.80*medr or body_ratio(D)>=0.45)

    side=0
    if bull_manip and bull_dist: side=1
    elif bear_manip and bear_dist: side=-1
    if side==0:return None

    if variant in ('AMD_A','AMD_CISD','AMD_TICK') and not accumulation:return None
    if variant in ('AMD_CISD','AMD_TICK'):
        cd=cisd_dir(bars[max(0,i-12):i],D)
        if cd!=side:return None
    return side


def first_passage(ticks,start_idx,side,entry,seconds,dist):
    start_ns=int(ticks[start_idx].ts_event); end_ns=start_ns+int(seconds*1e9)
    up=entry+dist; dn=entry-dist
    max_fav=0.0; max_adv=0.0
    for j in range(start_idx+1,len(ticks)):
        q=ticks[j]; ns=int(q.ts_event)
        if ns>end_ns:break
        p=float(q.bid_price)
        fav=(p-entry)*side; adv=-(p-entry)*side
        max_fav=max(max_fav,fav); max_adv=max(max_adv,adv)
        if fav>=dist:return 1,max_fav,max_adv
        if adv>=dist:return -1,max_fav,max_adv
    return 0,max_fav,max_adv


def trigger_index_tick(ticks,D,side,M):
    # Event-driven trigger inside D: first raw tick that reclaims M open in distribution direction.
    level=M.o
    for j in range(D.first_idx,D.last_idx+1):
        p=float(ticks[j].bid_price)
        if (side>0 and p>level) or (side<0 and p<level): return j
    return D.last_idx


def run(catalog,variant,max_ticks=0):
    ticks=load_ticks(catalog,max_ticks); bs=bars15(ticks)
    horizons=[1,3,5,10,15]; dists=[0.10,0.20,0.30]
    stats={(h,d):{'w':0,'l':0,'none':0,'mfe':0.0,'mae':0.0} for h in horizons for d in dists}
    signals=0; long_n=0; short_n=0
    quarter_counts=[0,0,0,0]
    for i in range(14,len(bs)-2):
        if variant=='BASE15':
            D=bs[i]; side=1 if D.c>D.o else -1 if D.c<D.o else 0
            if side==0:continue
        else:
            side=classify_amd(bs,i,variant)
            if side is None:continue
            D=bs[i]
        M=bs[i-1]
        if variant=='AMD_TICK': idx=trigger_index_tick(ticks,D,side,M)
        else: idx=D.last_idx
        if idx>=len(ticks)-2:continue
        entry=float(ticks[idx].bid_price)
        signals+=1; long_n+=side>0; short_n+=side<0
        q=(D.start_ns//1_000_000_000//15)%4; quarter_counts[q]+=1
        for h in horizons:
            for d in dists:
                r,mfe,mae=first_passage(ticks,idx,side,entry,h,d)
                s=stats[(h,d)]
                if r>0:s['w']+=1
                elif r<0:s['l']+=1
                else:s['none']+=1
                s['mfe']+=mfe; s['mae']+=mae
    rows=[]
    for h in horizons:
        for d in dists:
            s=stats[(h,d)]; decided=s['w']+s['l']; n=decided+s['none']
            rows.append({'horizon_s':h,'distance':d,'wins_first':s['w'],'losses_first':s['l'],'none':s['none'],
                         'first_passage_wr':s['w']/decided if decided else None,'coverage':decided/n if n else 0.0,
                         'avg_mfe':s['mfe']/n if n else 0.0,'avg_mae':s['mae']/n if n else 0.0})
    return {'variant':variant,'raw_bidask':True,'ticks':len(ticks),'bars15':len(bs),'signals':signals,'longs':long_n,'shorts':short_n,
            'distribution_quarter_counts_q1_q4':quarter_counts,
            'definition':{'A':'15s compression: range<=0.8 recent median OR body ratio<=0.35','M':'sweep of A high/low with close reclaim','D':'opposite distribution: reclaim M open + directional expansion','CISD':'reclaim latest opposing delivery open','AMD_TICK':'raw-tick trigger inside D at first M-open reclaim'},
            'first_passage':rows,'status':'DISCOVERY_DIAGNOSTIC_NOT_BROKER_PNL'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--variant',choices=['BASE15','AMD_MD','AMD_A','AMD_CISD','AMD_TICK'],required=True); ap.add_argument('--max-ticks',type=int,default=3000000)
    a=ap.parse_args(); res=run(a.catalog,a.variant,a.max_ticks)
    d=Path('results/dexg-amd15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/f'{a.variant}.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
