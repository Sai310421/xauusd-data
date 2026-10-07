#!/usr/bin/env python3
"""AMOS M1 ordered state-machine -> line-chart POI entry decoder v1.20.

Purpose
-------
Restore the ordered M1 logic first, then use the supplied line-chart POI models
only to resolve the concrete entry location which was missing when N=0.

Order (M1 only):
  Sweep -> CISD -> MSS -> Displacement -> MSS+
  -> arm ONE entry-location state
  -> first same-direction V/A/QM/OCL model formed after MSS+
  -> that model's POI becomes this context's destination entry location
  -> POI revisit -> favorable M1 Close reaction
  -> first subsequent raw Bid/Ask quote entry -> target/SL.

Important:
- The global V/A/QM/OCL detector is retained as a passive line-chart decoder.
- It is NOT the trade-search origin.
- No freshness ranking, candidate scoring, or pattern cherry-picking.
- One MSS+ context can own at most one POI. If that first entry model is on the
  wrong geometric side or invalidates before entry, the context fails; it does
  not search for another POI.
- Bullish MSS+ -> BUY model -> lower/equal POI retrace -> BUY.
- Bearish MSS+ -> SELL model -> upper/equal POI retrace -> SELL.
- M5/G75 and M15 are not used or modified.
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

def ordered_entries(ctx, st, z, raw):
    stats=Counter(); out=[]
    if ctx.empty or st.empty:
        return pd.DataFrame(), stats
    raw_ns=raw.ns.to_numpy()
    wait=pd.Timedelta(minutes=60)

    # Chronological state-machine ownership. We do not loop through alternatives.
    for cid,cx in ctx.sort_values("mssplus_time").reset_index(drop=True).iterrows():
        stats["mssplus_armed"] += 1
        t0=pd.Timestamp(cx.mssplus_time); t1=t0+wait
        di=int(cx.travel_dir)  # same-direction entry intent

        # Entry-location decoder: first line-chart entry model that FORMS after MSS+.
        formed=st[(st.dir==di) & (st.signal_time>=t0) & (st.signal_time<=t1)].sort_values("signal_time")
        if formed.empty:
            stats["no_entry_model_after_mssplus"] += 1
            continue

        s=formed.iloc[0]
        stats["entry_model_claimed"] += 1
        stats["claimed_"+str(s.pattern)] += 1

        # The claimed model either belongs to this sequence or the sequence fails.
        # Do not scan later POIs to rescue it.
        correct_side=(float(s.poi)<=float(cx.mssplus_close)) if di>0 else (float(s.poi)>=float(cx.mssplus_close))
        if not correct_side:
            stats["claimed_poi_wrong_side"] += 1
            continue
        stats["destination_poi_resolved"] += 1

        start=pd.Timestamp(s.signal_time)
        expiry=min(t1, start+pd.Timedelta(minutes=180))
        q=z[(z.datetime>=start)&(z.datetime<=expiry)]
        touched=False; touch_time=None
        for k in range(1,len(q)):
            b=q.iloc[k]; prev=q.iloc[k-1]; tol=.10*float(s.atr)
            invalid=(float(b.low)<=float(s.sl)) if di>0 else (float(b.high)>=float(s.sl))
            if invalid:
                stats["invalid_before_entry"] += 1
                break
            if not touched:
                touch=(float(b.low)<=float(s.poi)+tol) if di>0 else (float(b.high)>=float(s.poi)-tol)
                if not touch:
                    continue
                touched=True; touch_time=pd.Timestamp(b.datetime); stats["poi_revisit"] += 1

            confirm=(float(b.close)>float(s.poi) and float(b.close)>float(prev.close)) if di>0 else (float(b.close)<float(s.poi) and float(b.close)<float(prev.close))
            if not confirm:
                continue

            confirm_end=pd.Timestamp(b.datetime)+pd.Timedelta(minutes=1)
            j=int(np.searchsorted(raw_ns,int(confirm_end.value),side="left"))
            if j>=len(raw):
                stats["no_raw_quote"] += 1
                break
            first=raw.iloc[j]
            ep=float(first.ask if di>0 else first.bid)
            risk=(ep-float(s.sl))*di
            if risk<=0:
                stats["bad_risk"] += 1
                break
            tp=float(s.target)
            if not np.isfinite(tp) or (tp-ep)*di<=0:
                tp=ep+di*2*risk
            out.append({
                "tf":1,"context_id":int(cid),"mssplus_time":t0,"mssplus_dir":di,
                "mssplus_close":float(cx.mssplus_close),"mssplus_poi_type":str(cx.mssplus_poi_type),
                "pattern":str(s.pattern),"poi":float(s.poi),"poi_signal_time":pd.Timestamp(s.signal_time),
                "poi_revisit_time":touch_time,"confirm_time":confirm_end,
                "entry_dir":di,"entry_ns":int(first.ns),"entry_time":first.datetime,
                "entry":ep,"sl":float(s.sl),"tp":tp,"risk":risk,
            })
            stats["entries"] += 1
            stats["entry_"+str(s.pattern)] += 1
            break
        else:
            stats["no_revisit_or_confirm_before_expiry"] += 1

    return pd.DataFrame(out),stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    cat=ParquetDataCatalog(a.catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]); ticks.sort(key=lambda x:int(x.ts_event))
    ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
    bid=np.fromiter((v16.v14.fpx(x.bid_price) for x in ticks),float,count=len(ticks))
    ask=np.fromiter((v16.v14.fpx(x.ask_price) for x in ticks),float,count=len(ticks))
    raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"ns":ts,"bid":bid,"ask":ask})

    z=v16.v14.feature(v16.v14.bars(raw,1))
    ctx,counts=v16.mssplus_context(z,1)
    st=v16.add_targets(v16.line_setups(z),z,1)
    ent,stats=ordered_entries(ctx,st,z,raw)
    tr=v16.simulate(ent,raw,1)

    ctx.to_csv(out/"mssplus_contexts.csv",index=False)
    st.to_csv(out/"passive_line_models.csv",index=False)
    ent.to_csv(out/"ordered_entries.csv",index=False)
    tr.to_csv(out/"trades.csv",index=False)

    result={
      "version":"v1.20",
      "verification":"M1_ORDERED_STATE_MACHINE_LINE_POI_ENTRY_DECODER_RAW_BIDASK",
      "timeframe_min":1,
      "m5_used":False,"m15_used":False,"g75_used":False,
      "raw_ticks":len(raw),"m1_bars":len(z),
      "mssplus_counts":counts,
      "passive_line_models":len(st),
      "ordered_stats":dict(stats),
      "architecture":"M1 Sweep -> CISD -> MSS -> Displacement -> MSS+ -> arm entry-location state -> first same-direction line model -> its POI -> revisit -> favorable Close -> raw Bid/Ask entry",
      "ownership_rule":"one MSS+ context owns at most one line-chart POI; no alternative-POI search after claim",
      "direction_rule":"bullish MSS+ -> BUY POI -> BUY; bearish MSS+ -> SELL POI -> SELL",
      "entry_rule":"line-chart V/A/QM/OCL is the missing entry-location decoder, not an independent trade-search engine",
      "cost_assumption":{"commission_rt_per_lot":v16.COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":v16.CASHBACK_RT_PER_LOT},
      **v16.metrics(tr),
      "pattern_metrics":{k:v16.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}
    }
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
