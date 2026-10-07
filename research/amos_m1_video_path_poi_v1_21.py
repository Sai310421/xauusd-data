#!/usr/bin/env python3
"""AMOS M1 video-path -> destination line-POI decoder v1.21.

This revision fixes ONE conceptual error in v1.20:
the line POI is not selected first and then attached to MSS+.
The active video-framework path owns the search. Price delivery from MSS+ is
followed until it TERMINATES at the first causally confirmed V/A/QM/OCL entry
model lying in that delivery path. That terminal POI is the entry location.

Video framework source preserved from video_gate_sequence_v23:
  Sweep -> Delivery Context -> MSS/CISD -> Entry Model -> Clear Target -> Entry.
For this standalone M1 diagnostic the already-audited M1 MSS+ detector supplies
Sweep/CISD/MSS/Displacement context; no M5/M15/G75 gate is introduced.

Key geometry:
- travel_dir = direction of delivery TOWARD the destination POI.
- entry_dir  = opposite travel_dir because the supplied V/A/QM/OCL examples
  are reversal-entry POIs at the terminal end of that delivery.
- bearish delivery -> lower BUY POI -> BUY
- bullish delivery -> upper SELL POI -> SELL
- one active context owns at most one destination POI.
- no global POI-first matching, ranking, freshness scoring or rescue search.
"""
from __future__ import annotations
import argparse, json, importlib.util, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

spec=importlib.util.spec_from_file_location("v16", Path(__file__).with_name("amos_mssplus_destination_poi_v1_16.py"))
v16=importlib.util.module_from_spec(spec); sys.modules["v16"]=v16; spec.loader.exec_module(v16)

def path_entries(ctx, st, z, raw):
    stats=Counter(); out=[]
    if ctx.empty or st.empty:return pd.DataFrame(),stats
    raw_ns=raw.ns.to_numpy(); wait=pd.Timedelta(minutes=60)

    for cid,cx in ctx.sort_values("mssplus_time").reset_index(drop=True).iterrows():
        stats["framework_armed"]+=1
        t0=pd.Timestamp(cx.mssplus_time); t1=t0+wait
        travel=int(cx.travel_dir); entry_dir=-travel
        origin=float(cx.mssplus_close)

        # PATH-FIRST decoder. The context is already active. We only observe line
        # models which become causally confirmed while that path is active.
        formed=st[(st.dir==entry_dir)&(st.signal_time>=t0)&(st.signal_time<=t1)].sort_values("signal_time")
        if formed.empty:
            stats["no_terminal_entry_model"]+=1; continue

        resolved=None
        for _,s in formed.iterrows():
            # POI must be physically AHEAD in the active delivery direction.
            ahead=(float(s.poi)<origin) if travel<0 else (float(s.poi)>origin)
            if not ahead:
                stats["model_not_in_delivery_path"]+=1
                continue
            # First model in the path is the terminal decoder. Do not rank/scan
            # alternatives after claiming one.
            resolved=s; break

        if resolved is None:
            stats["no_poi_in_delivery_path"]+=1; continue
        s=resolved
        stats["destination_poi_resolved"]+=1
        stats["resolved_"+str(s.pattern)]+=1

        # The line model itself is causal confirmation of the terminal reversal.
        # Entry on first raw quote after its signal close; no artificial POI
        # revisit is demanded because the delivery has already ARRIVED at this POI.
        sig=pd.Timestamp(s.signal_time)
        confirm_end=sig+pd.Timedelta(minutes=1)
        j=int(np.searchsorted(raw_ns,int(confirm_end.value),side="left"))
        if j>=len(raw):
            stats["no_raw_quote"]+=1; continue
        first=raw.iloc[j]
        ep=float(first.ask if entry_dir>0 else first.bid)
        risk=(ep-float(s.sl))*entry_dir
        if risk<=0:
            stats["bad_risk"]+=1; continue
        tp=float(s.target)
        if not np.isfinite(tp) or (tp-ep)*entry_dir<=0:
            tp=ep+entry_dir*2*risk
        out.append({
          "tf":1,"context_id":int(cid),"mssplus_time":t0,
          "travel_dir":travel,"entry_dir":entry_dir,
          "mssplus_close":origin,"mssplus_poi_type":str(cx.mssplus_poi_type),
          "pattern":str(s.pattern),"destination_poi":float(s.poi),
          "poi_signal_time":sig,"confirm_time":confirm_end,
          "entry_ns":int(first.ns),"entry_time":first.datetime,
          "entry":ep,"sl":float(s.sl),"tp":tp,"risk":risk,
        })
        stats["entries"]+=1;stats["entry_"+str(s.pattern)]+=1
    return pd.DataFrame(out),stats

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
    ctx,counts=v16.mssplus_context(z,1)
    st=v16.add_targets(v16.line_setups(z),z,1)
    ent,stats=path_entries(ctx,st,z,raw)
    tr=v16.simulate(ent.rename(columns={"destination_poi":"poi"}),raw,1) if len(ent) else pd.DataFrame()

    ctx.to_csv(out/"framework_contexts.csv",index=False)
    st.to_csv(out/"passive_line_models.csv",index=False)
    ent.to_csv(out/"path_destination_entries.csv",index=False)
    tr.to_csv(out/"trades.csv",index=False)
    result={
      "version":"v1.21",
      "verification":"M1_VIDEO_PATH_TO_DESTINATION_LINE_POI_RAW_BIDASK",
      "timeframe_min":1,"m5_used":False,"m15_used":False,"g75_used":False,
      "raw_ticks":len(raw),"m1_bars":len(z),"mssplus_counts":counts,
      "passive_line_models":len(st),"path_stats":dict(stats),
      "video_framework":"Sweep -> Delivery Context -> MSS/CISD -> Entry Model -> Clear Target -> Entry",
      "current_executable_context":"M1 Sweep -> CISD -> MSS -> Displacement/FVG (existing audited proxy; no new proprietary gate invented)",
      "decoder":"active delivery path -> first opposite-direction V/A/QM/OCL terminal model physically ahead -> its POI is entry location",
      "direction_rule":"bearish delivery -> lower BUY POI -> BUY; bullish delivery -> upper SELL POI -> SELL",
      "entry_timing":"first raw Bid/Ask quote after causal terminal line-model confirmation; no second POI revisit",
      "ownership_rule":"one framework context -> max one destination POI; no POI-first search and no rescue scan after claim",
      "cost_assumption":{"commission_rt_per_lot":v16.COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":v16.CASHBACK_RT_PER_LOT},
      **v16.metrics(tr),
      "pattern_metrics":{k:v16.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}
    }
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
