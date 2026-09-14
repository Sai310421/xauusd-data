from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_liquidity_cycle_v6 as v6

# v6.4: split the liquidity cycle into two separately measured legs.
# PROBE = trade opposite the initial-sweep anchor immediately at ARM.
# DISTRIBUTION = after the adverse probe/recovery path, trade the original anchor direction.
# Structural Mid/Mid only: establish timing edge before broker-cost optimization.

HORIZONS = [1, 3, 5, 10, 15, 30]
DISTS = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]


def first_passage(ticks, start_idx, side, seconds, dist):
    entry = v6.mid(ticks[start_idx])
    end = int(ticks[start_idx].ts_event) + int(seconds * 1e9)
    mfe = 0.0; mae = 0.0
    for j in range(start_idx + 1, len(ticks)):
        if int(ticks[j].ts_event) > end: break
        move = (v6.mid(ticks[j]) - entry) * side
        mfe = max(mfe, move); mae = max(mae, -move)
        if move >= dist: return 1, mfe, mae
        if -move >= dist: return -1, mfe, mae
    return 0, mfe, mae


def zstats():
    return {(h,d): {'w':0,'l':0,'none':0,'mfe':0.0,'mae':0.0} for h in HORIZONS for d in DISTS}


def add(stats, ticks, idx, side):
    for h in HORIZONS:
        for d in DISTS:
            r,mfe,mae = first_passage(ticks, idx, side, h, d)
            s=stats[(h,d)]
            if r>0:s['w']+=1
            elif r<0:s['l']+=1
            else:s['none']+=1
            s['mfe']+=mfe; s['mae']+=mae


def summary(stats):
    out=[]
    for h in HORIZONS:
        for d in DISTS:
            s=stats[(h,d)]; dec=s['w']+s['l']; n=dec+s['none']
            out.append({'horizon_s':h,'distance':d,'wins_first':s['w'],'losses_first':s['l'],'none':s['none'],
                        'first_passage_wr':s['w']/dec if dec else None,'coverage':dec/n if n else 0.0,
                        'avg_mfe':s['mfe']/n if n else 0.0,'avg_mae':s['mae']/n if n else 0.0})
    return out


def run(catalog,max_ticks=3000000):
    ticks=v6.load_ticks(catalog,max_ticks); b15=v6.bars_n(ticks,15)
    lanes={k:{'arms':0,'entries':0,'stats':zstats(),'latencies':[]} for k in [
        'PROBE_ARM_INVERSE','DIST_AE05','DIST_AE10','DIST_FINAL_RETEST']}
    paired={'arms':0,'probe_entries':0,'dist_ae05_entries':0,'dist_ae10_entries':0}

    for i in range(15,len(b15)-1):
        if not v6.accumulation_ok(b15,i): continue
        A,S,H,D=b15[i-3],b15[i-2],b15[i-1],b15[i]
        anchor=v6.initial_sweep_side(A,S)
        if not anchor: continue
        hm=v6.opposite_harvest_meta(A,S,H,anchor)
        if not hm['broad']: continue
        arm_idx,_=v6.find_arm(ticks,D,anchor,hm['return_level'])
        if arm_idx is None or arm_idx>=len(ticks)-2: continue
        paired['arms']+=1

        # Leg 1: harvest/probe continuation, opposite to final anchor.
        p=lanes['PROBE_ARM_INVERSE']; p['arms']+=1; p['entries']+=1; paired['probe_entries']+=1
        add(p['stats'],ticks,arm_idx,-anchor)

        # Leg 2: original anchor direction only after adverse probe + recovery confirmation.
        for name,ae,retest in [('DIST_AE05',0.05,False),('DIST_AE10',0.10,False),('DIST_FINAL_RETEST',0.05,True)]:
            o=lanes[name]; o['arms']+=1
            idx,meta=v6.seek_second_entry(ticks,arm_idx,anchor,hm['return_level'],hm['harvest_extreme'],ae,retest)
            if idx is None: continue
            o['entries']+=1; o['latencies'].append(meta.get('latency_s',0.0)); add(o['stats'],ticks,idx,anchor)
            if name=='DIST_AE05': paired['dist_ae05_entries']+=1
            if name=='DIST_AE10': paired['dist_ae10_entries']+=1

    results=[]
    for name,o in lanes.items():
        results.append({'lane':name,'arms':o['arms'],'entries':o['entries'],'conversion':o['entries']/o['arms'] if o['arms'] else 0.0,
                        'latency_s':v6.summarize_latency(o['latencies']),'structural_first_passage':summary(o['stats'])})
    return {'status':'CAUSAL_V64_PROBE_DISTRIBUTION_STRUCTURAL_DIAGNOSTIC_NOT_BROKER_PNL',
            'raw_source':'Nautilus Raw Bid/Ask catalog; state and entry decisions are causal. First-passage is Mid/Mid structural diagnostic.',
            'ticks':len(ticks),'bars15':len(b15),
            'sequence':'Accumulation -> InitialSweep(anchor) -> OppositeHarvest -> ARM -> PROBE(-anchor); separately ARM -> adverse excursion -> recovery/confirmation -> DISTRIBUTION(anchor)',
            'hypothesis':'ARM is a probe-leg opportunity in the opposite direction; the original initial-sweep direction is evaluated only as a separate later distribution leg.',
            'paired_counts':paired,'lanes':results}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--max-ticks',type=int,default=3000000); a=ap.parse_args()
    res=run(a.catalog,a.max_ticks); d=Path('results/dexg-amd15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/'AMD_PROBE_DISTRIBUTION_V64.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
