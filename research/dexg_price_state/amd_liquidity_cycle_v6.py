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
    if max_ticks and len(ticks)>max_ticks:ticks=ticks[:max_ticks]
    return ticks

def bid(q):return float(q.bid_price)
def ask(q):return float(q.ask_price)
def spread(q):return max(0.0,ask(q)-bid(q))
def mid(q):return 0.5*(bid(q)+ask(q))

def bars_n(ticks,seconds):
    out=[]; bucket=None; cur=None; span_ns=int(seconds*1e9)
    for idx,q in enumerate(ticks):
        ns=int(q.ts_event); b=ns//span_ns; p=bid(q)
        if bucket is None or b!=bucket:
            if cur:out.append(cur)
            bucket=b; s=b*span_ns
            cur=Bar(s,s+span_ns,p,p,p,p,idx,idx)
        else:
            cur.h=max(cur.h,p);cur.l=min(cur.l,p);cur.c=p;cur.last_idx=idx
    if cur:out.append(cur)
    return out

def rng(b):return max(b.h-b.l,EPS)
def body_ratio(b):return abs(b.c-b.o)/rng(b)

def accumulation_ok(bs,i):
    if i<15:return False
    A=bs[i-3];recent=bs[max(0,i-13):i-3]
    medr=median([rng(x) for x in recent]) if recent else rng(A)
    return rng(A)<=0.90*medr or body_ratio(A)<=0.40

def initial_sweep_side(A,S):
    up=S.h>A.h and S.c<A.h;dn=S.l<A.l and S.c>A.l
    if up and not dn:return 1
    if dn and not up:return -1
    return 0

def opposite_harvest_meta(A,S,H,side):
    ar=rng(A)
    if side>0:
        strict=H.l<A.l
        travel=H.l<min(S.o,S.c)-max(0.05,0.15*ar)
        internal=H.l<min(A.o,A.c)
        wick=(min(H.o,H.c)-H.l)>=0.25*rng(H)
        return {'strict':bool(strict),'broad':bool(strict or (travel and (internal or wick))),
                'harvest_extreme':H.l,'return_level':max(A.h,S.o)}
    strict=H.h>A.h
    travel=H.h>max(S.o,S.c)+max(0.05,0.15*ar)
    internal=H.h>max(A.o,A.c)
    wick=(H.h-max(H.o,H.c))>=0.25*rng(H)
    return {'strict':bool(strict),'broad':bool(strict or (travel and (internal or wick))),
            'harvest_extreme':H.h,'return_level':min(A.l,S.o)}

def micro_state(ticks,j,side,lookback=8):
    if j<lookback:return False,False,0.0,0.0
    ps=[mid(ticks[k]) for k in range(j-lookback,j+1)]
    p=ps[-1];prior=ps[-6:-1]
    ref=max(prior) if side>0 else min(prior)
    cisd=(p>ref) if side>0 else (p<ref)
    d1=ps[-1]-ps[-2];d2=ps[-2]-ps[-3];d3=ps[-3]-ps[-4]
    vel=((d1+d2+d3)/3.0)*side;acc=(d1-d2)*side
    disp=vel>0 and acc>=0 and abs(ps[-1]-ps[-4])>=0.03
    return bool(cisd),bool(disp),vel,acc

def find_arm(ticks,D,side,return_level):
    reclaimed=False
    for j in range(D.first_idx+4,D.last_idx+1):
        p=mid(ticks[j])
        if not reclaimed:
            reclaimed=(side>0 and p>return_level) or (side<0 and p<return_level)
            if not reclaimed:continue
        _,disp,vel,acc=micro_state(ticks,j,side)
        if disp:return j,{'velocity':vel,'acceleration':acc}
    return None,None

def executable_entry(q,side):return ask(q) if side>0 else bid(q)
def executable_mark(q,side):return bid(q) if side>0 else ask(q)

def first_passage_exec(ticks,start_idx,side,entry,seconds,dist):
    end=int(ticks[start_idx].ts_event)+int(seconds*1e9);mfe=mae=0.0
    for j in range(start_idx+1,len(ticks)):
        if int(ticks[j].ts_event)>end:break
        p=executable_mark(ticks[j],side);fav=(p-entry)*side;adv=-(p-entry)*side
        mfe=max(mfe,fav);mae=max(mae,adv)
        if fav>=dist:return 1,mfe,mae
        if adv>=dist:return -1,mfe,mae
    return 0,mfe,mae

def seek_second_entry(ticks,arm_idx,side,return_level,harvest_extreme,ae_req,final_retest=False,max_life_s=15.0):
    arm_mid=mid(ticks[arm_idx]);arm_ts=int(ticks[arm_idx].ts_event);end=arm_ts+int(max_life_s*1e9)
    adverse_seen=False;recovered=False;retest_seen=False;retest_held=False
    max_adverse=0.0; recovery_level=return_level
    structural_buffer=0.05
    for j in range(arm_idx+1,len(ticks)):
        ts=int(ticks[j].ts_event)
        if ts>end:break
        p=mid(ticks[j]);adv=(arm_mid-p)*side
        max_adverse=max(max_adverse,adv)
        # Structural invalidation beyond completed opposite-harvest extreme.
        if side>0 and p<harvest_extreme-structural_buffer:return None,{'reason':'invalidated','max_adverse':max_adverse}
        if side<0 and p>harvest_extreme+structural_buffer:return None,{'reason':'invalidated','max_adverse':max_adverse}
        if not adverse_seen:
            if adv>=ae_req:adverse_seen=True
            else:continue
        if not recovered:
            if (side>0 and p>=recovery_level) or (side<0 and p<=recovery_level):recovered=True
            else:continue
        if final_retest and not retest_held:
            # Require a pullback after recovery, but it must hold on the anchor side of return_level.
            if not retest_seen:
                pullback=(recovery_level-p)*side
                if pullback>=0.02:retest_seen=True
                else:continue
            if retest_seen:
                if (side>0 and p<recovery_level-0.02) or (side<0 and p>recovery_level+0.02):
                    return None,{'reason':'retest_failed','max_adverse':max_adverse}
                if (side>0 and p>=recovery_level+0.02) or (side<0 and p<=recovery_level-0.02):retest_held=True
                else:continue
        cisd,disp,vel,acc=micro_state(ticks,j,side)
        if cisd and disp:
            return j,{'reason':'entry','max_adverse':max_adverse,'velocity':vel,'acceleration':acc,
                      'retest_held':bool(retest_held),'latency_s':(ts-arm_ts)/1e9}
    return None,{'reason':'timeout','max_adverse':max_adverse}

def summarize_latency(xs):
    if not xs:return {'mean':None,'median':None,'p90':None}
    ys=sorted(xs);n=len(ys)
    return {'mean':sum(ys)/n,'median':ys[n//2],'p90':ys[min(n-1,int(0.90*(n-1)))]}

def run(catalog,max_ticks=3000000):
    ticks=load_ticks(catalog,max_ticks);b15=bars_n(ticks,15)
    variants=['V6_ARM_BASELINE','V6_AE05_SECOND','V6_AE10_SECOND','V6_FINAL_RETEST','V6_INVERSE_ARM']
    horizons=[1,3,5,10,15];dists=[0.10,0.20,0.30]
    out={v:{'arms':0,'entries':0,'longs':0,'shorts':0,'strict':0,'broad_only':0,'latencies':[],'adverse':[],
            'arm_spread':0.0,'entry_spread':0.0,
            'stats':{(h,d):{'w':0,'l':0,'none':0,'mfe':0.0,'mae':0.0} for h in horizons for d in dists}} for v in variants}
    base_arms=0
    for i in range(15,len(b15)-1):
        if not accumulation_ok(b15,i):continue
        A,S,H,D=b15[i-3],b15[i-2],b15[i-1],b15[i]
        side=initial_sweep_side(A,S)
        if not side:continue
        hm=opposite_harvest_meta(A,S,H,side)
        if not hm['broad']:continue
        arm_idx,_=find_arm(ticks,D,side,hm['return_level'])
        if arm_idx is None or arm_idx>=len(ticks)-2:continue
        base_arms+=1
        for v in variants:
            o=out[v];o['arms']+=1;o['arm_spread']+=spread(ticks[arm_idx])
            entry_idx=None;trade_side=side;meta={'max_adverse':0.0,'latency_s':0.0}
            if v=='V6_ARM_BASELINE':entry_idx=arm_idx
            elif v=='V6_INVERSE_ARM':entry_idx=arm_idx;trade_side=-side
            elif v=='V6_AE05_SECOND':entry_idx,meta=seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.05,False)
            elif v=='V6_AE10_SECOND':entry_idx,meta=seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.10,False)
            elif v=='V6_FINAL_RETEST':entry_idx,meta=seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.05,True)
            if entry_idx is None:continue
            o['entries']+=1;o['longs']+=trade_side>0;o['shorts']+=trade_side<0
            o['strict']+=hm['strict'];o['broad_only']+=hm['broad'] and not hm['strict']
            o['latencies'].append(meta.get('latency_s',0.0));o['adverse'].append(meta.get('max_adverse',0.0))
            o['entry_spread']+=spread(ticks[entry_idx])
            ep=executable_entry(ticks[entry_idx],trade_side)
            for h in horizons:
                for d in dists:
                    r,mfe,mae=first_passage_exec(ticks,entry_idx,trade_side,ep,h,d);s=o['stats'][(h,d)]
                    if r>0:s['w']+=1
                    elif r<0:s['l']+=1
                    else:s['none']+=1
                    s['mfe']+=mfe;s['mae']+=mae
    variants_out=[]
    for v in variants:
        o=out[v];rows=[]
        for h in horizons:
            for d in dists:
                s=o['stats'][(h,d)];dec=s['w']+s['l'];n=dec+s['none']
                rows.append({'horizon_s':h,'distance':d,'wins_first':s['w'],'losses_first':s['l'],'none':s['none'],
                             'first_passage_wr':s['w']/dec if dec else None,'coverage':dec/n if n else 0.0,
                             'avg_mfe':s['mfe']/n if n else 0.0,'avg_mae':s['mae']/n if n else 0.0})
        adv=o['adverse']
        variants_out.append({'variant':v,'arms':o['arms'],'entries':o['entries'],
                             'arm_to_entry_conversion':o['entries']/o['arms'] if o['arms'] else 0.0,
                             'longs':o['longs'],'shorts':o['shorts'],'strict_external':o['strict'],'broad_only':o['broad_only'],
                             'avg_arm_spread':o['arm_spread']/o['arms'] if o['arms'] else 0.0,
                             'avg_entry_spread':o['entry_spread']/o['entries'] if o['entries'] else 0.0,
                             'latency_s':summarize_latency(o['latencies']),
                             'adverse_before_entry':{'mean':sum(adv)/len(adv) if adv else None,'median':sorted(adv)[len(adv)//2] if adv else None},
                             'first_passage':rows})
    return {'raw_bidask':True,'ticks':len(ticks),'bars15':len(b15),
            'causality':'Completed A/S/H before ARM; after ARM only current/past QuoteTicks. No completed future D filter and no backward entry search.',
            'execution':'Side-correct executable entry: long at Ask, short at Bid; favorable/adverse marks: long Bid, short Ask.',
            'sequence':'Accumulation -> InitialSweep anchor -> OppositeHarvest -> ARM -> adverse excursion -> recovery/final retest -> second micro-CISD + displacement -> entry',
            'direction_hypothesis':'FinalDirection=InitialSweepDirection remains unproven and is tested against inverse ARM control.',
            'variants':variants_out,'status':'CAUSAL_V6_FINAL_RETEST_DIAGNOSTIC_NOT_BROKER_PNL'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--max-ticks',type=int,default=3000000);a=ap.parse_args()
    res=run(a.catalog,a.max_ticks);d=Path('results/dexg-amd15')/a.experiment_id;d.mkdir(parents=True,exist_ok=True)
    p=d/'AMD_LIQUIDITY_CYCLE_V6.json';p.write_text(json.dumps(res,indent=2),encoding='utf-8');print(json.dumps(res,indent=2))
if __name__=='__main__':main()
