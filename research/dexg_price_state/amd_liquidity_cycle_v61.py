from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import median
from research.dexg_price_state import amd_liquidity_cycle_v6 as v6


def first_passage_struct(ticks,start_idx,side,seconds,dist):
    entry=v6.mid(ticks[start_idx]); end=int(ticks[start_idx].ts_event)+int(seconds*1e9); mfe=mae=0.0
    for j in range(start_idx+1,len(ticks)):
        if int(ticks[j].ts_event)>end: break
        p=v6.mid(ticks[j]); fav=(p-entry)*side; adv=-(p-entry)*side
        mfe=max(mfe,fav); mae=max(mae,adv)
        if fav>=dist: return 1,mfe,mae
        if adv>=dist: return -1,mfe,mae
    return 0,mfe,mae


def first_passage_exec(ticks,start_idx,side,seconds,dist):
    entry=v6.executable_entry(ticks[start_idx],side); end=int(ticks[start_idx].ts_event)+int(seconds*1e9); mfe=mae=0.0
    for j in range(start_idx+1,len(ticks)):
        if int(ticks[j].ts_event)>end: break
        p=v6.executable_mark(ticks[j],side); fav=(p-entry)*side; adv=-(p-entry)*side
        mfe=max(mfe,fav); mae=max(mae,adv)
        if fav>=dist: return 1,mfe,mae
        if adv>=dist: return -1,mfe,mae
    return 0,mfe,mae


def summarize(stats,horizons,dists):
    rows=[]
    for h in horizons:
        for d in dists:
            s=stats[(h,d)]; dec=s['w']+s['l']; n=dec+s['none']
            rows.append({'horizon_s':h,'distance':d,'wins_first':s['w'],'losses_first':s['l'],'none':s['none'],
                         'first_passage_wr':s['w']/dec if dec else None,'coverage':dec/n if n else 0.0,
                         'avg_mfe':s['mfe']/n if n else 0.0,'avg_mae':s['mae']/n if n else 0.0})
    return rows


def run(catalog,max_ticks=3000000):
    ticks=v6.load_ticks(catalog,max_ticks); b15=v6.bars_n(ticks,15)
    variants=['V61_ARM_BASELINE','V61_AE05_SECOND','V61_AE10_SECOND','V61_FINAL_RETEST','V61_INVERSE_ARM']
    horizons=[1,3,5,10,15]; struct_dists=[0.10,0.20,0.30]; exec_dists=[0.75,1.00,1.50]
    def zstats(ds): return {(h,d):{'w':0,'l':0,'none':0,'mfe':0.0,'mae':0.0} for h in horizons for d in ds}
    out={x:{'arms':0,'entries':0,'longs':0,'shorts':0,'strict':0,'broad_only':0,'latencies':[],'adverse':[],
            'arm_spread':0.0,'entry_spread':0.0,'struct':zstats(struct_dists),'exec':zstats(exec_dists)} for x in variants}
    for i in range(15,len(b15)-1):
        if not v6.accumulation_ok(b15,i): continue
        A,S,H,D=b15[i-3],b15[i-2],b15[i-1],b15[i]
        side=v6.initial_sweep_side(A,S)
        if not side: continue
        hm=v6.opposite_harvest_meta(A,S,H,side)
        if not hm['broad']: continue
        arm_idx,_=v6.find_arm(ticks,D,side,hm['return_level'])
        if arm_idx is None or arm_idx>=len(ticks)-2: continue
        for name in variants:
            o=out[name]; o['arms']+=1; o['arm_spread']+=v6.spread(ticks[arm_idx])
            entry_idx=None; trade_side=side; meta={'max_adverse':0.0,'latency_s':0.0}
            if name=='V61_ARM_BASELINE': entry_idx=arm_idx
            elif name=='V61_INVERSE_ARM': entry_idx=arm_idx; trade_side=-side
            elif name=='V61_AE05_SECOND': entry_idx,meta=v6.seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.05,False)
            elif name=='V61_AE10_SECOND': entry_idx,meta=v6.seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.10,False)
            elif name=='V61_FINAL_RETEST': entry_idx,meta=v6.seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.05,True)
            if entry_idx is None: continue
            o['entries']+=1; o['longs']+=trade_side>0; o['shorts']+=trade_side<0
            o['strict']+=hm['strict']; o['broad_only']+=hm['broad'] and not hm['strict']
            o['latencies'].append(meta.get('latency_s',0.0)); o['adverse'].append(meta.get('max_adverse',0.0)); o['entry_spread']+=v6.spread(ticks[entry_idx])
            for h in horizons:
                for d in struct_dists:
                    r,mfe,mae=first_passage_struct(ticks,entry_idx,trade_side,h,d); s=o['struct'][(h,d)]
                    if r>0:s['w']+=1
                    elif r<0:s['l']+=1
                    else:s['none']+=1
                    s['mfe']+=mfe; s['mae']+=mae
                for d in exec_dists:
                    r,mfe,mae=first_passage_exec(ticks,entry_idx,trade_side,h,d); s=o['exec'][(h,d)]
                    if r>0:s['w']+=1
                    elif r<0:s['l']+=1
                    else:s['none']+=1
                    s['mfe']+=mfe; s['mae']+=mae
    vo=[]
    for name in variants:
        o=out[name]; adv=o['adverse']; es=o['entry_spread']/o['entries'] if o['entries'] else 0.0
        vo.append({'variant':name,'arms':o['arms'],'entries':o['entries'],'arm_to_entry_conversion':o['entries']/o['arms'] if o['arms'] else 0.0,
                   'longs':o['longs'],'shorts':o['shorts'],'strict_external':o['strict'],'broad_only':o['broad_only'],
                   'avg_arm_spread':o['arm_spread']/o['arms'] if o['arms'] else 0.0,'avg_entry_spread':es,
                   'entry_spread_to_structural_010_ratio':es/0.10 if o['entries'] else None,
                   'latency_s':v6.summarize_latency(o['latencies']),
                   'adverse_before_entry':{'mean':sum(adv)/len(adv) if adv else None,'median':sorted(adv)[len(adv)//2] if adv else None},
                   'structural_first_passage':summarize(o['struct'],horizons,struct_dists),
                   'executable_first_passage':summarize(o['exec'],horizons,exec_dists)})
    spreads=[v6.spread(q) for q in ticks]
    return {'raw_bidask':True,'ticks':len(ticks),'bars15':len(b15),
            'causality':'Same causal v6 state machine. Completed A/S/H before ARM; after ARM only current/past QuoteTicks.',
            'metric_split':{'structural':'Mid-to-Mid directional first passage, distances $0.10/$0.20/$0.30.',
                            'executable':'Long Ask/short Bid entry, long Bid/short Ask mark, distances $0.75/$1.00/$1.50.'},
            'spread_all_ticks':{'mean':sum(spreads)/len(spreads),'median':median(spreads),'p90':sorted(spreads)[int(0.90*(len(spreads)-1))]},
            'sequence':'Accumulation -> InitialSweep anchor -> OppositeHarvest -> ARM -> adverse excursion -> recovery/final retest -> second micro-CISD + displacement -> entry',
            'variants':vo,'status':'CAUSAL_V61_STRUCTURAL_VS_EXECUTABLE_DIAGNOSTIC_NOT_BROKER_PNL'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--max-ticks',type=int,default=3000000); a=ap.parse_args()
    res=run(a.catalog,a.max_ticks); d=Path('results/dexg-amd15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/'AMD_LIQUIDITY_CYCLE_V61.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
