#!/usr/bin/env python3
"""AMOS pure-M1 destination-touch precision v1.26.

Precision correction derived from the user's core rule:
  "MSS+ が行って向かう POI がエントリー場所"

Therefore v1.26 removes the late-entry mistake from v1.25:
- a V/A/QM/OCL POI must already be causally defined by the SAME M1 setup
  before/at MSS+ activation;
- after MSS+, the active price path is followed;
- the first owned POI actually reached by that path is the destination;
- entry is at the first raw Bid/Ask touch of that POI, NOT after a later
  reversal-confirmation signal.

Pure M1 only:
- no M5/M15/M30/G75;
- no higher-timeframe target;
- no fixed-multiple / synthetic TP.

Target:
same-setup causal opposite Close pivot which already exists before entry.
No target => no official trade.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

spec=importlib.util.spec_from_file_location("v24",Path(__file__).with_name("amos_m1_pure_video_target_audit_v1_24.py"))
v24=importlib.util.module_from_spec(spec);sys.modules["v24"]=v24;spec.loader.exec_module(v24)
v22=v24.v22;v16=v24.v16

def pre_mss_owned_models(z,cx):
    """Only same-setup line models causally known no later than MSS+."""
    m=v22.context_models(z,cx)
    if m.empty:return m
    t0=pd.Timestamp(cx.mssplus_time)
    # context_models signal_time is bar open-time of confirming bar; require its
    # close to be known by MSS+ activation.
    known=m[(m.signal_time+pd.Timedelta(minutes=1))<=t0].copy()
    return known

def first_raw_touch(raw,start_ns,end_ns,poi,di,tol):
    """First executable raw quote touching the destination POI."""
    ns=raw.ns.to_numpy()
    j0=int(np.searchsorted(ns,int(start_ns),side="left"))
    j1=int(np.searchsorted(ns,int(end_ns),side="right"))
    for j in range(j0,min(j1,len(raw))):
        # BUY destination is approached downward: ask is executable buy price.
        # SELL destination is approached upward: bid is executable sell price.
        px=float(raw.ask.iat[j] if di>0 else raw.bid.iat[j])
        if (px<=poi+tol) if di>0 else (px>=poi-tol):
            return j,px
    return None,None

def causal_structure_target_before_entry(e,z):
    """Pure-M1 target known before entry and born inside same setup."""
    di=int(e.entry_dir);ep=float(e.entry);et=pd.Timestamp(e.entry_time)
    start=int(e.context_sweep_i)
    end=int(np.searchsorted(z.datetime.to_numpy(),np.datetime64(et),side="right")-1)
    want="H" if di>0 else "L";c=[]
    for p in v24.causal_pivots(z,start,end):
        if p["type"]!=want or pd.Timestamp(p["confirmed_time"])>et:continue
        px=float(p["price"]);dist=(px-ep) if di>0 else (ep-px)
        if dist>0:c.append((dist,p))
    if not c:return None
    c.sort(key=lambda x:(x[0],x[1]["idx"]))
    _,p=c[0]
    return {"tp":float(p["price"]),"target_kind":"CLOSE_PIVOT_"+want,
            "target_ref_i":int(p["idx"]),"target_ref_time":pd.Timestamp(p["confirmed_time"])}

def destination_touch_entries(ctx,z,raw):
    stats=Counter();rows=[];audit=[]
    raw_ns=raw.ns.to_numpy()
    for cid,cx in ctx.sort_values("mssplus_time").reset_index(drop=True).iterrows():
        stats["framework_armed"]+=1
        t0=pd.Timestamp(cx.mssplus_time);t1=t0+pd.Timedelta(minutes=60)
        travel=int(cx.travel_dir);entry_dir=-travel;origin=float(cx.mssplus_close)

        models=pre_mss_owned_models(z,cx)
        if models.empty:
            stats["no_pre_mss_owned_poi"]+=1;continue

        # Destination side is fixed by the active delivery direction.
        cand=models[models.dir==entry_dir].copy()
        if cand.empty:
            stats["no_correct_direction_owned_poi"]+=1;continue
        if travel<0:
            cand=cand[cand.poi<origin]
        else:
            cand=cand[cand.poi>origin]
        if cand.empty:
            stats["no_owned_poi_ahead"]+=1;continue

        # No nearest/freshness/ranking guess. Let the real post-MSS+ path decide:
        # whichever owned POI is actually touched first is the destination.
        hits=[]
        start_ns=int(t0.value);end_ns=int(t1.value)
        for _,s in cand.iterrows():
            tol=max(0.0,0.10*float(s.atr))
            j,px=first_raw_touch(raw,start_ns,end_ns,float(s.poi),entry_dir,tol)
            if j is None:continue
            hits.append((int(raw.ns.iat[j]),j,s,px))
        if not hits:
            stats["no_owned_poi_reached_by_path"]+=1;continue
        hits.sort(key=lambda x:x[0])
        hit_ns,j,s,ep=hits[0]
        stats["destination_poi_reached"]+=1
        stats["reached_"+str(s.pattern)]+=1

        # Structural invalidation must not precede the actual touch.
        # Check M1 bars from MSS+ to touch against the model's own invalidation.
        hit_time=pd.Timestamp(raw.datetime.iat[j])
        q=z[(z.datetime>=t0)&(z.datetime<hit_time)]
        if len(q):
            invalid=bool((q.low<=float(s.sl)).any()) if entry_dir>0 else bool((q.high>=float(s.sl)).any())
            if invalid:
                stats["reject_invalid_before_touch"]+=1;continue

        risk=(float(ep)-float(s.sl))*entry_dir
        if risk<=0:
            stats["bad_risk"]+=1;continue

        base={"tf":1,"context_id":int(cid),"context_sweep_i":int(cx.sweep_i),
              "context_mss_i":int(cx.mss_i),"context_disp_i":int(cx.disp_i),
              "sweep_time":cx.sweep_time,"mssplus_time":t0,
              "travel_dir":travel,"entry_dir":entry_dir,"mssplus_close":origin,
              "pattern":str(s.pattern),"poi":float(s.poi),"anchor_i":int(s.anchor_i),
              "poi_signal_time":pd.Timestamp(s.signal_time),
              "entry_ns":hit_ns,"entry_time":hit_time,"entry":float(ep),
              "sl":float(s.sl),"risk":float(risk),"mssplus_poi_type":str(cx.mssplus_poi_type)}
        audit.append(base.copy())

        tgt=causal_structure_target_before_entry(pd.Series(base),z)
        if tgt is None:
            stats["reject_no_m1_structure_target"]+=1;continue
        base.update(tgt);rows.append(base)
        stats["entries"]+=1;stats["entry_"+str(s.pattern)]+=1

    return pd.DataFrame(rows),stats,pd.DataFrame(audit)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cat=ParquetDataCatalog(a.catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks.sort(key=lambda x:int(x.ts_event))
    ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
    bid=np.fromiter((v16.v14.fpx(x.bid_price) for x in ticks),float,count=len(ticks))
    ask=np.fromiter((v16.v14.fpx(x.ask_price) for x in ticks),float,count=len(ticks))
    raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"ns":ts,"bid":bid,"ask":ask})
    z=v16.v14.feature(v16.v14.bars(raw,1))
    ctx,counts=v22.owned_mssplus_context(z)
    ent,stats,audit=destination_touch_entries(ctx,z,raw)
    tr=v16.simulate(ent,raw,1) if len(ent) else pd.DataFrame()
    ctx.to_csv(out/"framework_contexts.csv",index=False)
    audit.to_csv(out/"destination_touches_before_target.csv",index=False)
    ent.to_csv(out/"entries.csv",index=False)
    tr.to_csv(out/"trades.csv",index=False)
    result={
      "version":"v1.26","verification":"PURE_M1_VIDEO_DESTINATION_POI_RAW_TOUCH_ENTRY",
      "timeframe_min":1,"m5_used":False,"m15_used":False,"m30_used":False,"g75_used":False,
      "raw_ticks":len(raw),"m1_bars":len(z),"mssplus_counts":counts,
      "path_stats":dict(stats),
      "architecture":"same-setup POI known before MSS+ -> MSS+ delivery path -> first owned POI actually touched -> raw Bid/Ask entry at POI -> same-setup M1 structure target",
      "key_fix_from_v125":"Entry moved from post-reversal confirmation to the destination POI touch itself. The path chooses the POI by actual first reach; no nearest/freshness ranking.",
      "target_policy":"PURE M1 causal same-setup opposite Close pivot known before entry; NO fixed multiple and NO fallback",
      "cost_assumption":{"commission_rt_per_lot":v16.COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":v16.CASHBACK_RT_PER_LOT},
      **v16.metrics(tr),
      "pattern_metrics":{k:v16.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}
    }
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
