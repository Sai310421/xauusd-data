#!/usr/bin/env python3
"""AMOS M1 setup-owned POI + video Clear-Target decoder v1.23.

v1.22 fixed entry-POI ownership. v1.23 fixes EXIT target semantics.

NO RR / NO 2R fallback.
The trade is valid for performance scoring only when a causal Clear Target
visible before entry can be resolved from the video-style FVG target map.

Target map:
- build 15M and 30M bars causally from raw BID quotes;
- detect FVG zones which already existed before the entry;
- reject zones already touched/mitigated before the entry;
- BUY: nearest still-clear FVG zone ABOVE entry;
- SELL: nearest still-clear FVG zone BELOW entry;
- TP is first-touch boundary of that target zone (near edge), because the
  video's target is the zone itself, not an arbitrary reward multiple.

This uses 15M/30M ONLY as target-reference maps. It does not introduce the
separate M15 trading/AMD engine and does not alter M5/G75.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys,math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

spec=importlib.util.spec_from_file_location("v22",Path(__file__).with_name("amos_m1_setup_owned_linepoi_v1_22.py"))
v22=importlib.util.module_from_spec(spec);sys.modules["v22"]=v22;spec.loader.exec_module(v22)
v16=v22.v16

def fvg_targets(z,tf):
    """Causal FVG target zones. signal_time is when the third bar closes."""
    rows=[]
    for i in range(2,len(z)):
        r=z.iloc[i]
        # Bullish FVG: current low > high two bars back.
        if float(r.low)>float(z.high.iloc[i-2]):
            lo=float(z.high.iloc[i-2]);hi=float(r.low)
            rows.append({"tf":tf,"formed_i":i,"formed_time":pd.Timestamp(r.datetime)+pd.Timedelta(minutes=tf),
                         "lo":min(lo,hi),"hi":max(lo,hi),"fvg_kind":"BULL_FVG"})
        # Bearish FVG: current high < low two bars back.
        if float(r.high)<float(z.low.iloc[i-2]):
            lo=float(r.high);hi=float(z.low.iloc[i-2])
            rows.append({"tf":tf,"formed_i":i,"formed_time":pd.Timestamp(r.datetime)+pd.Timedelta(minutes=tf),
                         "lo":min(lo,hi),"hi":max(lo,hi),"fvg_kind":"BEAR_FVG"})
    return pd.DataFrame(rows)

def is_clear_before_entry(z,zone,entry_time):
    """Zone must not have been touched after it formed and before entry."""
    ft=pd.Timestamp(zone.formed_time)
    q=z[(z.datetime>=ft)&(z.datetime<entry_time)]
    if q.empty:return True
    lo=float(zone.lo);hi=float(zone.hi)
    # any bar overlapping the zone means it was already mitigated/touched
    return not bool(((q.high>=lo)&(q.low<=hi)).any())

def resolve_clear_target(entry_time,entry,di,z15,z30,t15,t30):
    c=[]
    for tf,z,tmap in ((15,z15,t15),(30,z30,t30)):
        if tmap.empty:continue
        prior=tmap[tmap.formed_time<=entry_time]
        for _,g in prior.iterrows():
            lo=float(g.lo);hi=float(g.hi)
            if di>0:
                if lo<=entry:continue
                target=lo  # first touch of target zone
                dist=target-entry
            else:
                if hi>=entry:continue
                target=hi  # first touch of target zone
                dist=entry-target
            if dist<=0:continue
            if not is_clear_before_entry(z,g,entry_time):continue
            c.append({"target":target,"tf":tf,"kind":str(g.fvg_kind),
                      "zone_lo":lo,"zone_hi":hi,"formed_time":pd.Timestamp(g.formed_time),
                      "distance":dist})
    if not c:return None
    # "Clear Target": first still-open target in the actual delivery direction.
    c.sort(key=lambda x:(x["distance"],x["tf"],x["formed_time"]))
    return c[0]

def attach_video_targets(ent,raw,z15,z30,t15,t30):
    stats=Counter();rows=[]
    if ent.empty:return pd.DataFrame(),stats
    for _,e in ent.iterrows():
        di=int(e.entry_dir);ep=float(e.entry);et=pd.Timestamp(e.entry_time)
        stats["entry_candidates"]+=1
        tgt=resolve_clear_target(et,ep,di,z15,z30,t15,t30)
        if tgt is None:
            stats["reject_no_clear_fvg_target"]+=1
            continue
        r=e.to_dict()
        r["tp"]=float(tgt["target"])
        r["target_tf_min"]=int(tgt["tf"])
        r["target_kind"]=tgt["kind"]
        r["target_zone_lo"]=float(tgt["zone_lo"])
        r["target_zone_hi"]=float(tgt["zone_hi"])
        r["target_formed_time"]=tgt["formed_time"]
        rows.append(r)
        stats["clear_target_resolved"]+=1
        stats["target_"+str(tgt["tf"])+"m"]+=1
    return pd.DataFrame(rows),stats

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

    z1=v16.v14.feature(v16.v14.bars(raw,1))
    ctx,counts=v22.owned_mssplus_context(z1)
    ent0,entry_stats,owned=v22.owned_entries(ctx,z1,raw)

    z15=v16.v14.feature(v16.v14.bars(raw,15))
    z30=v16.v14.feature(v16.v14.bars(raw,30))
    t15=fvg_targets(z15,15);t30=fvg_targets(z30,30)
    ent,target_stats=attach_video_targets(ent0,raw,z15,z30,t15,t30)
    tr=v16.simulate(ent,raw,1) if len(ent) else pd.DataFrame()

    ctx.to_csv(out/"framework_contexts.csv",index=False)
    owned.to_csv(out/"owned_line_models.csv",index=False)
    ent0.to_csv(out/"owned_entry_candidates_before_target.csv",index=False)
    pd.concat([t15,t30],ignore_index=True).to_csv(out/"causal_fvg_target_map.csv",index=False)
    ent.to_csv(out/"entries_with_clear_target.csv",index=False)
    tr.to_csv(out/"trades.csv",index=False)

    result={
      "version":"v1.23",
      "verification":"M1_VIDEO_SETUP_OWNED_POI_PLUS_CAUSAL_CLEAR_FVG_TARGET_RAW_BIDASK",
      "timeframe_min":1,"m5_trade_logic_used":False,"m15_trade_logic_used":False,"g75_used":False,
      "target_reference_timeframes":[15,30],
      "raw_ticks":len(raw),"m1_bars":len(z1),
      "mssplus_counts":counts,"owned_line_models":len(owned),
      "entry_ownership_stats":dict(entry_stats),
      "target_stats":dict(target_stats),
      "architecture":"video setup-owned M1 entry POI -> causal pre-existing unmitigated 15M/30M FVG Clear Target -> raw Bid/Ask execution",
      "tp_rule":"NO_RR. Nearest still-clear FVG zone ahead; TP at first-touch boundary of that zone.",
      "no_target_rule":"If no causal Clear Target exists, reject from official performance sample. No synthetic/fixed-multiple fallback.",
      "target_reference_note":"15M/30M are target maps only; separate M15 AMD/trading logic is not used.",
      "cost_assumption":{"commission_rt_per_lot":v16.COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":v16.CASHBACK_RT_PER_LOT},
      **v16.metrics(tr),
      "pattern_metrics":{k:v16.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}
    }
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
# workflow trigger: v1.23 clear-target validation
