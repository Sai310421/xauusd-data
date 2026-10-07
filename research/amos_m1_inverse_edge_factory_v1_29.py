#!/usr/bin/env python3
"""AMOS M1 Inverse EDGE Factory v1.29.

Middle architecture: shared core + parallel lanes + re-merge.
- Real lane: Dukascopy Raw Bid/Ask, authoritative promotion KPI.
- Random lane: synthetic M1 -> synthetic QuoteTick stress, fragility only.
- Same candidate definitions and inverse KPI scorer on both lanes.
- M1 only. No M5/M15/M30/G75. No fixed RR. No HTF target.

Target EDGE (Classic A SELL):
N >= 120, WR >= 45%, PF >= 1.20.
"""
from __future__ import annotations
import argparse, json, importlib.util, sys, math, random
from pathlib import Path
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v128", HERE/"amos_m1_classic_a_line_invalidation_v1_28.py")
v128=importlib.util.module_from_spec(spec);sys.modules["v128"]=v128;spec.loader.exec_module(v128)
v127=v128.v127; v125=v128.v125; v16=v128.v16

TARGET_N=120
TARGET_WR=45.0
TARGET_PF=1.20

CANDIDATES=[]
for fm in ["NECK","GEOM_NECK","LOOSE_NECK"]:
    for sk in ["ANCHOR_CLOSE","SIGNAL_OPEN","MIN_ANCHOR_WICK"]:
        for b in [0.00,0.03,0.05,0.10]:
            CANDIDATES.append((f"{fm}_{sk}_B{int(round(b*100)):02d}",fm,sk,b))

def inverse_gap(m):
    n=int(m.get("N") or 0);wr=float(m.get("WR_pct") or 0.0);pf=float(m.get("PF") or 0.0)
    gn=max(0.0,TARGET_N-n)/TARGET_N
    gw=max(0.0,TARGET_WR-wr)/TARGET_WR
    gp=max(0.0,TARGET_PF-pf)/TARGET_PF
    score=3.0*gp+2.0*gw+1.0*gn
    return {"gap_N":gn,"gap_WR":gw,"gap_PF":gp,"inverse_score":score,
            "passes_real_gate":bool(n>=TARGET_N and wr>=TARGET_WR and pf>=TARGET_PF)}

def eval_quotes(raw):
    z=v16.v14.feature(v16.v14.bars(raw,1))
    ctx,counts=v125.v22.owned_mssplus_context(z)
    base,stats,audit=v125.path_extreme_entries(ctx,z,raw)
    res={}
    for name,fm,sk,b in CANDIDATES:
        ent=v128.build(base,z,fm,sk,b)
        tr,mp=v128.pack(ent,raw)
        a=mp["CLASSIC_A_SELL"]
        res[name]={"filter":fm,"stop_kind":sk,"buffer_atr":b,
                   "selected_A":int((ent.pattern=="CLASSIC_A_SELL").sum()),
                   **mp,**inverse_gap(a)}
    rank=sorted(
        [{"name":k,**v} for k,v in res.items()],
        key=lambda x:(x["inverse_score"], -(x["CLASSIC_A_SELL"].get("PF") or 0.0), -(x["CLASSIC_A_SELL"].get("N") or 0))
    )
    return z,counts,stats,res,rank

# Synthetic generator adapted from the user's RandomChart_Nautilus_v1.
def make_random_quotes(seed:int,vol_mult:float,days:int=7,ticks_per_minute:int=20,
                       initial_price:float=2000.0,point:float=0.01,spread_points:float=20.0):
    from nautilus_trader.model.data import QuoteTick
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Price, Quantity
    rng=random.Random(seed or 1)
    def normal():
        u1=max(rng.random(),1e-12);u2=rng.random()
        return math.sqrt(-2.0*math.log(u1))*math.cos(2.0*math.pi*u2)
    # Same fallback calibration intent as uploaded generator: 8 points at price 2000.
    sigma=((8.0*point)/max(initial_price,1.0))*max(vol_mult,0.0)
    range_ratio=2.0*((8.0*point)/max(initial_price,1.0))*max(vol_mult,0.0)
    end=datetime(2026,10,8,0,0,tzinfo=timezone.utc)
    start=end-timedelta(days=days)
    iid=InstrumentId.from_str("XAUUSD.SIM")
    q=[];prev=initial_price
    minute_count=int((end-start).total_seconds()//60)
    spread=max(spread_points,0.0)*point
    n=max(4,int(ticks_per_minute))
    for mi in range(minute_count):
        ts0=start+timedelta(minutes=mi)
        o=prev
        c=max(100.0,o*math.exp(normal()*sigma-0.5*sigma*sigma))
        body_ratio=abs(c-o)/max(o,1.0)
        target_range=max(range_ratio*0.35,body_ratio)
        extra=abs(normal())*target_range*0.65*0.75
        h=max(o,c)+(target_range+extra)*o*0.25
        l=max(100.0,min(o,c)-(target_range+extra)*o*0.25)
        anchors=[o,h,l,c] if rng.random()<0.5 else [o,l,h,c]
        lens=[max(1,n//3),max(1,n//3)];lens.append(max(1,n-sum(lens)))
        mids=[]
        for ix,m in enumerate(lens):
            a,b=anchors[ix],anchors[ix+1]
            for j in range(m):
                frac=(j+1)/m
                base=a+(b-a)*frac
                noise=normal()*max(abs(h-l),1e-9)*0.015*(1-abs(2*frac-1))
                mids.append(min(h,max(l,base+noise)))
        mids=mids[:n]
        if mids:mids[-1]=c
        for i,mid in enumerate(mids):
            ms=int(i*60000/n)
            ts=ts0+timedelta(milliseconds=ms)
            bid=round((mid-spread/2)/point)*point
            ask=round(max(mid+spread/2,bid+point)/point)*point
            ns=int(ts.timestamp()*1_000_000_000)
            q.append(QuoteTick(instrument_id=iid,
                 bid_price=Price(Decimal(str(round(bid,2))),precision=2),
                 ask_price=Price(Decimal(str(round(ask,2))),precision=2),
                 bid_size=Quantity.from_str("1"),ask_size=Quantity.from_str("1"),
                 ts_event=ns,ts_init=ns))
        prev=c
    return q

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lane",choices=["real","random"],required=True)
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",required=True)
    ap.add_argument("--seed",type=int,default=20261005)
    ap.add_argument("--vol-mult",type=float,default=1.0)
    ap.add_argument("--days",type=int,default=7)
    ap.add_argument("--ticks-per-minute",type=int,default=20)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.lane=="real":
        raw=v127.v126.load_raw(args.catalog)
        lane_meta={"lane":"real_raw","authoritative":True,"raw_ticks":len(raw)}
    else:
        raw=make_random_quotes(args.seed,args.vol_mult,args.days,args.ticks_per_minute)
        lane_meta={"lane":"random_stress","authoritative":False,"synthetic_quotes":len(raw),
                   "seed":args.seed,"vol_mult":args.vol_mult,"days":args.days,
                   "ticks_per_minute":args.ticks_per_minute,
                   "note":"Synthetic stress only; never final KPI."}
    z,counts,stats,res,rank=eval_quotes(raw)
    result={"version":"v1.29","architecture":"SHARED_CORE_PARALLEL_LANES_REMERGE",
      "lane_meta":lane_meta,
      "target_edge":{"Classic_A_SELL":{"N_min":TARGET_N,"WR_min_pct":TARGET_WR,"PF_min":TARGET_PF}},
      "constraints":["M1 only","video semantics primary","Classic V BUY untouched",
                     "no M5/M15/M30/G75","no fixed RR","no fixed multiple TP","no HTF target"],
      "m1_bars":len(z),"mssplus_counts":counts,"path_stats":dict(stats),
      "top_inverse":rank[:15],"candidates":res}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
