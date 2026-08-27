#!/usr/bin/env python3
import argparse, json, lzma, struct, bisect
from pathlib import Path
from collections import deque

REC=struct.Struct('>3i2f')
SCALE=1000.0
HORIZONS=(50,75,100,125,150,200,250,300)


def load_ticks(root):
    out=[]; hour=0
    files=sorted(Path(root).rglob('*h_ticks.bi5'))
    for f in files:
        try:
            dec=lzma.decompress(f.read_bytes())
        except Exception:
            hour+=1; continue
        for i in range(0,len(dec)-REC.size+1,REC.size):
            ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i)
            out.append((hour*3600000+ms,bid_i/SCALE,ask_i/SCALE))
        hour+=1
    return out,files


def candidate_signal(asks,bids):
    # exact Candidate A requested by user
    ask_slope=(-2*asks[0]-asks[1]+asks[3]+2*asks[4])/10.0
    bid_slope=(-2*bids[0]-bids[1]+bids[3]+2*bids[4])/10.0
    a_cond=-ask_slope
    b_cond=bid_slope
    if b_cond>0 and a_cond<0: return 1
    if a_cond>0 and b_cond<0: return -1
    return 0


def evaluate(ticks,start,end,horizon_ms):
    ts=[x[0] for x in ticks]
    asks=deque(maxlen=5); bids=deque(maxlen=5)
    wins=losses=ties=signals=buy=sell=0
    for idx in range(start,end):
        t,bid,ask=ticks[idx]
        asks.append(ask); bids.append(bid)
        if len(asks)<5: continue
        side=candidate_signal(asks,bids)
        if side==0: continue
        j=bisect.bisect_left(ts,t+horizon_ms,idx+1,end)
        if j>=end: continue
        mid0=(bid+ask)/2.0
        _,b2,a2=ticks[j]
        mid1=(b2+a2)/2.0
        ret=mid1-mid0
        signals+=1
        if side>0: buy+=1
        else: sell+=1
        z=side*ret
        if z>0: wins+=1
        elif z<0: losses+=1
        else: ties+=1
    decisive=wins+losses
    return {
        'signals':signals,'BUY':buy,'SELL':sell,'wins':wins,'losses':losses,'ties':ties,
        'WR_decisive_pct':100.0*wins/decisive if decisive else None,
        'WR_all_pct':100.0*wins/signals if signals else None,
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('-o','--out',default='candidate_a_ols_wr.json'); a=ap.parse_args()
    ticks,files=load_ticks(a.root)
    if len(ticks)<100: raise SystemExit('insufficient ticks')
    cut=len(ticks)//2
    out={'candidate':'5-point OLS: a=-ask_slope, b=+bid_slope','target_WR_pct':72.71,'ticks':len(ticks),'bi5_files':len(files),'discovery_ticks':cut,'validation_ticks':len(ticks)-cut,'horizons':{}}
    for h in HORIZONS:
        d=evaluate(ticks,0,cut,h); v=evaluate(ticks,cut,len(ticks),h)
        vals=[x for x in (d['WR_decisive_pct'],v['WR_decisive_pct']) if x is not None]
        worst_error=max(abs(x-72.71) for x in vals) if vals else None
        out['horizons'][str(h)]={'discovery':d,'validation':v,'worst_abs_error_to_72_71_pctpt':worst_error}
    ranked=sorted(out['horizons'].items(),key=lambda kv: kv[1]['worst_abs_error_to_72_71_pctpt'] if kv[1]['worst_abs_error_to_72_71_pctpt'] is not None else 1e99)
    out['best_by_worst_split']= {'horizon_ms':int(ranked[0][0]), **ranked[0][1]}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
