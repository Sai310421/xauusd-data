from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import median
from dataclasses import dataclass
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

EPS = 1e-12

@dataclass
class Bar:
    start_ns:int; end_ns:int; o:float; h:float; l:float; c:float; first_idx:int; last_idx:int

def load_ticks(catalog,max_ticks=0):
    cat=ParquetDataCatalog(catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    if max_ticks and len(ticks)>max_ticks: ticks=ticks[:max_ticks]
    return ticks

def bars_n(ticks,seconds):
    out=[]; bucket=None; cur=None; span_ns=int(seconds*1e9)
    for idx,q in enumerate(ticks):
        ns=int(q.ts_event); b=ns//span_ns; p=float(q.bid_price)
        if bucket is None or b!=bucket:
            if cur: out.append(cur)
            bucket=b; s=b*span_ns
            cur=Bar(s,s+span_ns,p,p,p,p,idx,idx)
        else:
            cur.h=max(cur.h,p); cur.l=min(cur.l,p); cur.c=p; cur.last_idx=idx
    if cur: out.append(cur)
    return out

def rng(b): return max(b.h-b.l,EPS)
def body_ratio(b): return abs(b.c-b.o)/rng(b)
def close_loc(b): return (b.c-b.l)/rng(b)

def completed_before(bs,ts_ns):
    lo,hi=0,len(bs)-1; ans=None
    while lo<=hi:
        m=(lo+hi)//2
        if bs[m].end_ns<=ts_ns: ans=bs[m]; lo=m+1
        else: hi=m-1
    return ans

def accumulation_ok(bs,i):
    # v5 sequence uses A=i-3, InitialSweep=i-2, OppositeHarvest=i-1, D=i.
    if i<15:return False
    A=bs[i-3]; recent=bs[max(0,i-13):i-3]
    medr=median([rng(x) for x in recent]) if recent else rng(A)
    return rng(A)<=0.90*medr or body_ratio(A)<=0.40

def initial_sweep_side(A,S):
    # Direction anchor = first liquidity sweep direction.
    up = S.h>A.h and S.c<A.h
    dn = S.l<A.l and S.c>A.l
    if up and not dn:return 1
    if dn and not up:return -1
    return 0

def opposite_harvest_meta(A,S,H,anchor_side):
    # H is completed before D. It must travel opposite the initial sweep direction.
    # strict_sweep: takes the opposite side of A (external liquidity).
    # broad_harvest: also accepts internal/local liquidity collection without an A-extreme sweep.
    a_range=rng(A)
    if anchor_side>0:
        strict = H.l<A.l
        opposite_travel = H.l < min(S.o,S.c) - max(0.05,0.15*a_range)
        internal_take = H.l < min(A.o,A.c)
        wick_probe = (min(H.o,H.c)-H.l) >= 0.25*rng(H)
        return {'strict_sweep':bool(strict),'broad_harvest':bool(strict or (opposite_travel and (internal_take or wick_probe))),
                'harvest_level':H.l,'return_level':max(A.h,S.o)}
    strict = H.h>A.h
    opposite_travel = H.h > max(S.o,S.c) + max(0.05,0.15*a_range)
    internal_take = H.h > max(A.o,A.c)
    wick_probe = (H.h-max(H.o,H.c)) >= 0.25*rng(H)
    return {'strict_sweep':bool(strict),'broad_harvest':bool(strict or (opposite_travel and (internal_take or wick_probe))),
            'harvest_level':H.h,'return_level':min(A.l,S.o)}

def htf_context(ts_ns,p,side,b1,b5,b60):
    score=0.0
    for series,w in ((b1,0.30),(b5,0.30),(b60,0.40)):
        b=completed_before(series,ts_ns)
        if not b:continue
        orient=(b.c-b.o)*side
        if orient>0: score+=w*0.60
        cl=close_loc(b)
        if (side>0 and cl>=0.60) or (side<0 and cl<=0.40): score+=w*0.25
        lo=min(b.o,b.c); hi=max(b.o,b.c)
        if lo<=p<=hi: score+=w*0.15
    return min(1.0,score)

def return_cisd_displacement(ticks,D,anchor_side,return_level,require_cisd=True):
    # Event-driven D start. All checks use only current/past ticks inside D.
    reclaimed=False
    for j in range(D.first_idx+4,D.last_idx+1):
        p=float(ticks[j].bid_price)
        if not reclaimed:
            reclaimed=(anchor_side>0 and p>return_level) or (anchor_side<0 and p<return_level)
            if not reclaimed: continue
        if j<8:continue
        ps=[float(ticks[k].bid_price) for k in range(j-8,j+1)]
        prior=ps[-6:-1]
        ref=max(prior) if anchor_side>0 else min(prior)
        cisd=(p>ref) if anchor_side>0 else (p<ref)
        d1=ps[-1]-ps[-2]; d2=ps[-2]-ps[-3]; d3=ps[-3]-ps[-4]
        vel=((d1+d2+d3)/3.0)*anchor_side
        acc=(d1-d2)*anchor_side
        displacement=vel>0 and acc>=0 and abs(ps[-1]-ps[-4])>=0.03
        if displacement and ((not require_cisd) or cisd):
            return j,{'cisd':bool(cisd),'velocity':vel,'acceleration':acc,'reclaimed':True}
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

def run_variant(ticks,b15,b1,b5,b60,variant):
    horizons=[1,3,5,10,15]; dists=[0.10,0.20,0.30]
    stats={(h,d):{'w':0,'l':0,'none':0,'mfe':0.0,'mae':0.0} for h in horizons for d in dists}
    sig=longs=shorts=0
    feat={'initial_up':0,'initial_down':0,'strict_opp_sweep':0,'broad_opp_harvest':0,'htf_positive':0}
    for i in range(15,len(b15)-1):
        if not accumulation_ok(b15,i):continue
        A,S,H,D=b15[i-3],b15[i-2],b15[i-1],b15[i]
        side=initial_sweep_side(A,S)
        if not side:continue
        hm=opposite_harvest_meta(A,S,H,side)
        if variant=='V5_STRICT_SWEEP_CISD' and not hm['strict_sweep']:continue
        if variant in ('V5_BROAD_HARVEST_CISD','V5_BROAD_HARVEST_DISP') and not hm['broad_harvest']:continue
        require_cisd=variant!='V5_BROAD_HARVEST_DISP'
        idx,meta=return_cisd_displacement(ticks,D,side,hm['return_level'],require_cisd=require_cisd)
        if idx is None or idx>=len(ticks)-2:continue
        p=float(ticks[idx].bid_price); ts=int(ticks[idx].ts_event)
        htf=htf_context(ts,p,side,b1,b5,b60)
        sig+=1; longs+=side>0; shorts+=side<0
        feat['initial_up']+=side>0; feat['initial_down']+=side<0
        feat['strict_opp_sweep']+=hm['strict_sweep']; feat['broad_opp_harvest']+=hm['broad_harvest']; feat['htf_positive']+=htf>0
        for h in horizons:
            for d in dists:
                r,mfe,mae=first_passage(ticks,idx,side,p,h,d); s=stats[(h,d)]
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
    return {'variant':variant,'signals':sig,'longs':longs,'shorts':shorts,'feature_counts':feat,'first_passage':rows}

def run(catalog,max_ticks=3000000):
    ticks=load_ticks(catalog,max_ticks)
    b15=bars_n(ticks,15); b1=bars_n(ticks,60); b5=bars_n(ticks,300); b60=bars_n(ticks,3600)
    variants=['V5_STRICT_SWEEP_CISD','V5_BROAD_HARVEST_CISD','V5_BROAD_HARVEST_DISP']
    return {
      'raw_bidask':True,'ticks':len(ticks),'bars15':len(b15),'bars1m':len(b1),'bars5m':len(b5),'bars1h':len(b60),
      'causality':'A, initial sweep, and opposite-harvest bars are completed before D. D entry is current/past raw ticks only. No completed-D filter and no backward entry search.',
      'hard_rule':'Final trade direction equals InitialSweepDirection.',
      'sequence':'Accumulation -> InitialSweep(direction anchor) -> OppositeLiquidityHarvest -> Return/Reclaim -> CISD(optional ablation) -> RawTick displacement -> Entry in InitialSweepDirection',
      'harvest_definition':'Strict external opposite sweep OR broad internal/local-liquidity take / wick probe with sufficient opposite travel. Broad mode is intended to capture LAN-like no-external-sweep cases.',
      'variants':[run_variant(ticks,b15,b1,b5,b60,v) for v in variants],
      'status':'CAUSAL_V5_LIQUIDITY_CYCLE_DIAGNOSTIC_NOT_BROKER_PNL'
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--max-ticks',type=int,default=3000000); a=ap.parse_args()
    res=run(a.catalog,a.max_ticks); d=Path('results/dexg-amd15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/'AMD_LIQUIDITY_CYCLE_V5.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
