from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import median
from dataclasses import dataclass
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
    if max_ticks and len(ticks)>max_ticks: ticks=ticks[:max_ticks]
    return ticks

def bars15(ticks):
    out=[]; bucket=None; cur=None
    for idx,q in enumerate(ticks):
        ns=int(q.ts_event); b=(ns//1_000_000_000)//15; p=float(q.bid_price)
        if bucket is None or b!=bucket:
            if cur: out.append(cur)
            bucket=b; s=b*15*1_000_000_000
            cur=Bar(s,s+15_000_000_000,p,p,p,p,idx,idx)
        else:
            cur.h=max(cur.h,p); cur.l=min(cur.l,p); cur.c=p; cur.last_idx=idx
    if cur: out.append(cur)
    return out

def rng(b): return max(b.h-b.l,EPS)
def body_ratio(b): return abs(b.c-b.o)/rng(b)

def setup_side(bs,i):
    if i<14:return 0
    A,M=bs[i-2],bs[i-1]
    recent=bs[i-12:i-2]
    medr=median([rng(x) for x in recent]) if recent else rng(A)
    accumulation = rng(A)<=0.80*medr or body_ratio(A)<=0.35
    if not accumulation:return 0
    bull_manip = M.l<A.l and M.c>A.l
    bear_manip = M.h>A.h and M.c<A.h
    if bull_manip and not bear_manip:return 1
    if bear_manip and not bull_manip:return -1
    return 0

def trigger_a(ticks,D,M,side,confirm_ticks=3):
    level=M.o
    for j in range(max(D.first_idx+confirm_ticks,D.first_idx),D.last_idx+1):
        p=float(ticks[j].bid_price)
        reclaimed=(side>0 and p>level) or (side<0 and p<level)
        if not reclaimed: continue
        ps=[float(ticks[k].bid_price) for k in range(j-confirm_ticks,j+1)]
        ds=[ps[k+1]-ps[k] for k in range(len(ps)-1)]
        score=sum(1 if d*side>0 else -1 if d*side<0 else 0 for d in ds)
        if score>=1:return j,side
    return None,None

def trigger_b(ticks,D,M,side):
    idx,s=trigger_a(ticks,D,M,side)
    return (idx,-s) if idx is not None else (None,None)

def trigger_c(ticks,D,M,side,confirm_ticks=3,lookback=8):
    # C: setup side from A/M, but do NOT enter on first M-open reclaim.
    # Require causal sequence inside D:
    # 1) M-open reclaim in setup direction
    # 2) pullback/retest toward level
    # 3) micro-CISD style break of local opposing delivery open
    # 4) positive velocity/acceleration in setup direction
    level=M.o
    reclaimed_idx=None
    retest_idx=None
    for j in range(max(D.first_idx+confirm_ticks,D.first_idx),D.last_idx+1):
        p=float(ticks[j].bid_price)
        if reclaimed_idx is None:
            if (side>0 and p>level) or (side<0 and p<level):
                reclaimed_idx=j
            continue
        # retest: price revisits within 0.08 of M open without decisively crossing manipulation side
        if retest_idx is None:
            if abs(p-level)<=0.08:
                retest_idx=j
            continue
        if j<max(retest_idx+3,D.first_idx+3):
            continue
        start=max(D.first_idx,j-lookback)
        ps=[float(ticks[k].bid_price) for k in range(start,j+1)]
        if len(ps)<5: continue
        # micro opposing-delivery-open proxy: prior local 4-tick extreme body transition
        prior=ps[-5:-1]
        ref=max(prior) if side>0 else min(prior)
        cisd=(side>0 and p>ref) or (side<0 and p<ref)
        if not cisd: continue
        d1=ps[-1]-ps[-2]; d2=ps[-2]-ps[-3]; d3=ps[-3]-ps[-4]
        v=(d1+d2+d3)/3.0
        a=d1-d2
        if v*side>0 and a*side>=0:
            return j,side
    return None,None

def first_passage(ticks,start_idx,side,entry,seconds,dist):
    end=int(ticks[start_idx].ts_event)+int(seconds*1e9); mfe=mae=0.0
    for j in range(start_idx+1,len(ticks)):
        if int(ticks[j].ts_event)>end:break
        p=float(ticks[j].bid_price); fav=(p-entry)*side; adv=-(p-entry)*side
        mfe=max(mfe,fav); mae=max(mae,adv)
        if fav>=dist:return 1,mfe,mae
        if adv>=dist:return -1,mfe,mae
    return 0,mfe,mae

def run_variant(ticks,bs,variant):
    horizons=[1,3,5,10,15]; dists=[0.10,0.20,0.30]
    stats={(h,d):{'w':0,'l':0,'none':0,'mfe':0.0,'mae':0.0} for h in horizons for d in dists}
    sig=longs=shorts=0
    for i in range(14,len(bs)-2):
        setup=setup_side(bs,i)
        if not setup: continue
        D=bs[i]; M=bs[i-1]
        if variant=='A_RECLAIM': idx,side=trigger_a(ticks,D,M,setup)
        elif variant=='B_REVERSE': idx,side=trigger_b(ticks,D,M,setup)
        else: idx,side=trigger_c(ticks,D,M,setup)
        if idx is None or idx>=len(ticks)-2: continue
        entry=float(ticks[idx].bid_price); sig+=1; longs+=side>0; shorts+=side<0
        for h in horizons:
            for d in dists:
                r,mfe,mae=first_passage(ticks,idx,side,entry,h,d); s=stats[(h,d)]
                if r>0:s['w']+=1
                elif r<0:s['l']+=1
                else:s['none']+=1
                s['mfe']+=mfe; s['mae']+=mae
    rows=[]
    for h in horizons:
        for d in dists:
            s=stats[(h,d)]; dec=s['w']+s['l']; n=dec+s['none']
            rows.append({'horizon_s':h,'distance':d,'wins_first':s['w'],'losses_first':s['l'],'none':s['none'],
                         'first_passage_wr':s['w']/dec if dec else None,'coverage':dec/n if n else 0.0,
                         'avg_mfe':s['mfe']/n if n else 0.0,'avg_mae':s['mae']/n if n else 0.0})
    return {'variant':variant,'signals':sig,'longs':longs,'shorts':shorts,'first_passage':rows}

def run(catalog,max_ticks=3000000):
    ticks=load_ticks(catalog,max_ticks); bs=bars15(ticks)
    variants=['A_RECLAIM','B_REVERSE','C_MICRO_CISD']
    return {'raw_bidask':True,'ticks':len(ticks),'bars15':len(bs),
            'causality':'Only completed A/M + current/past D raw ticks. No completed-D filter and no backward entry search.',
            'definitions':{
                'A_RECLAIM':'v2 setup direction; first causal M-open reclaim + 3-tick confirmation',
                'B_REVERSE':'same trigger timing as A but trade opposite direction',
                'C_MICRO_CISD':'reclaim -> retest -> causal micro-CISD/local break + velocity/acceleration confirmation'},
            'variants':[run_variant(ticks,bs,v) for v in variants],
            'status':'CAUSAL_ABLATION_DIAGNOSTIC_NOT_BROKER_PNL'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--max-ticks',type=int,default=3000000); a=ap.parse_args()
    res=run(a.catalog,a.max_ticks); d=Path('results/dexg-amd15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/'AMD_CAUSAL_AB_V3.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
