#!/usr/bin/env python3
"""AMOS pure-M1 path-extreme precision v1.25.

Goal: improve M1 precision by fixing entry ownership, not by optimizing filters.

Source of truth:
- video sequence is primary;
- line chart is the M1 structure representation;
- POI is the concrete entry location reached by that SAME setup;
- no M5/M15/M30/G75;
- no fixed RR / no synthetic TP.

Key precision change vs v1.24:
Each active setup carries ONE live Close-line delivery path after MSS+.
A terminal V/A/QM/OCL model is eligible only when its POI anchor is the
CURRENT EXTREME reached by that setup's own delivery:
  bearish delivery -> running minimum Close -> BUY terminal POI
  bullish delivery -> running maximum Close -> SELL terminal POI
This prevents unrelated line models inside the same time window from owning
the setup.

Target:
Pure-M1 structure target only (the stronger v1.24 branch): nearest causally
confirmed opposite Close pivot ahead, born inside the same setup.
No target => no official trade. No RR fallback.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys,math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

spec=importlib.util.spec_from_file_location("v24",Path(__file__).with_name("amos_m1_pure_video_target_audit_v1_24.py"))
v24=importlib.util.module_from_spec(spec);sys.modules["v24"]=v24;spec.loader.exec_module(v24)
v22=v24.v22; v16=v24.v16

def path_extreme_entries(ctx,z,raw):
    stats=Counter(); rows=[]; audit=[]
    raw_ns=raw.ns.to_numpy()
    for cid,cx in ctx.sort_values("mssplus_time").reset_index(drop=True).iterrows():
        stats["framework_armed"]+=1
        t0=pd.Timestamp(cx.mssplus_time); t1=t0+pd.Timedelta(minutes=60)
        travel=int(cx.travel_dir); entry_dir=-travel
        start_i=int(np.searchsorted(z.datetime.to_numpy(),np.datetime64(t0),side="left"))
        end_i=min(len(z)-1,start_i+60)
        if start_i>=len(z): continue

        # All pattern anchors are still generated only from this setup.
        models=v22.context_models(z,cx)
        if models.empty:
            stats["no_owned_line_model"]+=1;continue

        # One live Close-line delivery path.
        running_extreme=float(cx.mssplus_close)
        extreme_i=start_i-1
        chosen=None
        for i in range(start_i,end_i+1):
            c=float(z.close.iloc[i])
            advanced=(c<running_extreme) if travel<0 else (c>running_extreme)
            if advanced:
                running_extreme=c; extreme_i=i; stats["path_extreme_updates"]+=1

            # Only models confirmed by now, correct reversal direction.
            now=pd.Timestamp(z.datetime.iloc[i])
            cand=models[(models.dir==entry_dir)&(models.signal_time<=now)&(models.signal_time>=t0)]
            if cand.empty: continue
            # terminal model must be anchored at the live delivery extreme.
            for _,s in cand.sort_values("signal_time").iterrows():
                ai=int(s.anchor_i)
                if ai!=extreme_i:
                    stats["reject_not_live_path_extreme"]+=1
                    continue
                # POI itself must agree with the live extreme close (line-chart semantics).
                tol=max(1e-9,0.10*float(s.atr))
                if abs(float(s.poi)-running_extreme)>tol:
                    stats["reject_poi_not_extreme_price"]+=1
                    continue
                chosen=s; break
            if chosen is not None: break

        if chosen is None:
            stats["no_terminal_model_on_live_path"]+=1;continue

        s=chosen; stats["destination_poi_resolved"]+=1; stats["resolved_"+str(s.pattern)]+=1
        confirm_end=pd.Timestamp(s.signal_time)+pd.Timedelta(minutes=1)
        j=int(np.searchsorted(raw_ns,int(confirm_end.value),side="left"))
        if j>=len(raw):stats["no_raw_quote"]+=1;continue
        first=raw.iloc[j]; ep=float(first.ask if entry_dir>0 else first.bid)
        risk=(ep-float(s.sl))*entry_dir
        if risk<=0:stats["bad_risk"]+=1;continue

        entry_bar_i=int(np.searchsorted(z.datetime.to_numpy(),np.datetime64(pd.Timestamp(s.signal_time)),side="right")-1)
        base={"tf":1,"context_id":int(cid),"context_sweep_i":int(cx.sweep_i),
              "context_mss_i":int(cx.mss_i),"context_disp_i":int(cx.disp_i),
              "sweep_time":cx.sweep_time,"mssplus_time":t0,
              "travel_dir":travel,"entry_dir":entry_dir,"mssplus_close":float(cx.mssplus_close),
              "pattern":str(s.pattern),"poi":float(s.poi),"anchor_i":int(s.anchor_i),
              "path_extreme_i":int(extreme_i),"path_extreme_close":float(running_extreme),
              "poi_signal_time":pd.Timestamp(s.signal_time),"entry_bar_i":entry_bar_i,
              "entry_ns":int(first.ns),"entry_time":first.datetime,"entry":ep,
              "sl":float(s.sl),"risk":risk,"mssplus_poi_type":str(cx.mssplus_poi_type)}
        audit.append(base.copy())

        # Pure M1 structure target, same setup only.
        tgt=v24.resolve_structure_target(pd.Series(base),z)
        if tgt is None:
            stats["reject_no_m1_structure_target"]+=1
            continue
        base.update(tgt)
        rows.append(base); stats["entries"]+=1; stats["entry_"+str(s.pattern)]+=1

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
    ent,stats,audit=path_extreme_entries(ctx,z,raw)
    tr=v16.simulate(ent,raw,1) if len(ent) else pd.DataFrame()
    ctx.to_csv(out/"framework_contexts.csv",index=False)
    audit.to_csv(out/"path_extreme_resolutions_before_target.csv",index=False)
    ent.to_csv(out/"entries.csv",index=False)
    tr.to_csv(out/"trades.csv",index=False)
    result={
      "version":"v1.25","verification":"PURE_M1_VIDEO_LIVE_CLOSE_PATH_EXTREME_POI_RAW_BIDASK",
      "timeframe_min":1,"m5_used":False,"m15_used":False,"m30_used":False,"g75_used":False,
      "raw_ticks":len(raw),"m1_bars":len(z),"mssplus_counts":counts,
      "path_stats":dict(stats),
      "architecture":"M1 setup -> MSS+ -> one live Close delivery path -> running path extreme -> terminal V/A/QM/OCL anchored exactly at that extreme -> M1 same-setup structure target",
      "precision_change":"Reject any same-window model whose anchor is not the live extreme of the active setup path.",
      "target_policy":"PURE M1 same-setup opposite Close pivot; NO RR and NO fallback",
      "cost_assumption":{"commission_rt_per_lot":v16.COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":v16.CASHBACK_RT_PER_LOT},
      **v16.metrics(tr),
      "pattern_metrics":{k:v16.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}
    }
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
