#!/usr/bin/env python3
"""AMOS pure-M1 video path v1.24.

M1 ONLY.
No M5 trade logic. No M15/30 target map. No G75.

Fixes two remaining leaks:
1) destination POI must be the endpoint/extreme of THIS active M1 delivery path,
   not merely any owned pattern inside the setup;
2) TP is resolved from a causal, still-clear M1 Close-liquidity target only.
   No fixed-multiple fallback and no higher-timeframe target substitution.

Flow:
  M1 Sweep -> CISD -> MSS -> Displacement/MSS+
  -> same setup owns line structure
  -> destination POI must sit at active delivery endpoint
  -> raw Bid/Ask entry
  -> M1 Clear Target (unswept confirmed Close pivot ahead)
  -> raw Bid/Ask exit.

This is a diagnostic reconstruction of the supplied video sequence. It does not
invent HTF-PDA/Macro/Volume gates which are not uniquely specified.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys,math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

spec=importlib.util.spec_from_file_location("v22",Path(__file__).with_name("amos_m1_setup_owned_linepoi_v1_22.py"))
v22=importlib.util.module_from_spec(spec);sys.modules["v22"]=v22;spec.loader.exec_module(v22)
v16=v22.v16

def causal_close_pivots(z):
    """2-left/2-right Close pivots, available only when right bars have closed."""
    rows=[]
    for confirm_i in range(4,len(z)):
        i=confirm_i-2;c=float(z.close.iloc[i])
        left=z.close.iloc[i-2:i];right=z.close.iloc[i+1:i+3]
        if c<float(left.min()) and c<=float(right.min()):
            rows.append({"type":"L","pivot_i":i,"confirm_i":confirm_i,"price":c,
                         "confirm_time":pd.Timestamp(z.datetime.iloc[confirm_i])+pd.Timedelta(minutes=1)})
        elif c>float(left.max()) and c>=float(right.max()):
            rows.append({"type":"H","pivot_i":i,"confirm_i":confirm_i,"price":c,
                         "confirm_time":pd.Timestamp(z.datetime.iloc[confirm_i])+pd.Timedelta(minutes=1)})
    return pd.DataFrame(rows)

def path_endpoint_filter(ent,z):
    stats=Counter();rows=[]
    if ent.empty:return pd.DataFrame(),stats
    for _,e in ent.iterrows():
        stats["owned_entry_candidate"]+=1
        t0=pd.Timestamp(e.mssplus_time)
        t1=pd.Timestamp(e.poi_signal_time)
        seg=z[(z.datetime>=t0)&(z.datetime<=t1)]
        if seg.empty:
            stats["reject_empty_delivery_path"]+=1;continue
        di=int(e.entry_dir);poi=float(e.poi)
        # M1 line chart uses Close structure. Require POI to be at the terminal
        # Close extreme reached by the active delivery, within a small M1 ATR tolerance.
        a=float(seg.atr.iloc[-1]) if np.isfinite(seg.atr.iloc[-1]) else 0.0
        tol=max(0.0,0.10*a)
        if di>0: # bearish delivery ended at BUY POI
            endpoint=float(seg.close.min())
            ok=poi<=endpoint+tol
        else:    # bullish delivery ended at SELL POI
            endpoint=float(seg.close.max())
            ok=poi>=endpoint-tol
        if not ok:
            stats["reject_not_delivery_endpoint"]+=1;continue
        r=e.to_dict();r["delivery_endpoint_close"]=endpoint;r["endpoint_tolerance"]=tol
        rows.append(r);stats["endpoint_poi_resolved"]+=1;stats["endpoint_"+str(e.pattern)]+=1
    return pd.DataFrame(rows),stats

def pivot_still_clear(z,p,entry_time,di,entry):
    price=float(p.price)
    # target must be ahead in the entry direction
    if di>0 and price<=entry:return False
    if di<0 and price>=entry:return False
    ct=pd.Timestamp(p.confirm_time)
    if ct>entry_time:return False
    q=z[(z.datetime>=ct)&(z.datetime<entry_time)]
    if q.empty:return True
    # "Clear" means the target liquidity has not already been taken.
    if di>0:return not bool((q.high>=price).any())
    return not bool((q.low<=price).any())

def attach_m1_clear_target(ent,z,piv):
    stats=Counter();rows=[]
    if ent.empty or piv.empty:return pd.DataFrame(),stats
    for _,e in ent.iterrows():
        stats["endpoint_entry_candidate"]+=1
        et=pd.Timestamp(e.entry_time);ep=float(e.entry);di=int(e.entry_dir)
        typ="H" if di>0 else "L"
        pp=piv[(piv.type==typ)&(piv.confirm_time<=et)]
        c=[]
        for _,p in pp.iterrows():
            if pivot_still_clear(z,p,et,di,ep):
                dist=(float(p.price)-ep)*di
                if dist>0:c.append((dist,p))
        if not c:
            stats["reject_no_m1_clear_target"]+=1;continue
        c.sort(key=lambda x:(x[0],int(x[1].pivot_i)))
        _,p=c[0]
        r=e.to_dict();r["tp"]=float(p.price)
        r["target_type"]="M1_CLEAR_CLOSE_LIQUIDITY"
        r["target_pivot_i"]=int(p.pivot_i);r["target_confirm_time"]=pd.Timestamp(p.confirm_time)
        rows.append(r);stats["m1_clear_target_resolved"]+=1
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

    z=v16.v14.feature(v16.v14.bars(raw,1))
    ctx,counts=v22.owned_mssplus_context(z)
    ent0,ownership_stats,owned=v22.owned_entries(ctx,z,raw)
    ent1,endpoint_stats=path_endpoint_filter(ent0,z)
    piv=causal_close_pivots(z)
    ent,target_stats=attach_m1_clear_target(ent1,z,piv)
    tr=v16.simulate(ent,raw,1) if len(ent) else pd.DataFrame()

    ctx.to_csv(out/"m1_framework_contexts.csv",index=False)
    owned.to_csv(out/"m1_setup_owned_line_models.csv",index=False)
    ent0.to_csv(out/"m1_owned_entry_candidates.csv",index=False)
    ent1.to_csv(out/"m1_path_endpoint_entries.csv",index=False)
    piv.to_csv(out/"m1_causal_close_targets.csv",index=False)
    ent.to_csv(out/"m1_entries_with_clear_target.csv",index=False)
    tr.to_csv(out/"trades.csv",index=False)

    result={
      "version":"v1.24",
      "verification":"PURE_M1_VIDEO_PATH_ENDPOINT_POI_AND_M1_CLEAR_TARGET_RAW_BIDASK",
      "timeframe_min":1,
      "m5_used":False,"m15_used":False,"m30_used":False,"g75_used":False,
      "raw_ticks":len(raw),"m1_bars":len(z),"mssplus_counts":counts,
      "setup_owned_line_models":len(owned),
      "ownership_stats":dict(ownership_stats),
      "endpoint_stats":dict(endpoint_stats),
      "target_stats":dict(target_stats),
      "architecture":"M1 setup state -> setup-owned line structure -> POI must be active delivery Close-endpoint -> raw entry -> still-clear M1 Close liquidity target",
      "tp_rule":"M1 clear structural target only. No fixed-multiple target and no higher-timeframe target map.",
      "no_target_rule":"No M1 clear target -> excluded from official performance sample.",
      "cost_assumption":{"commission_rt_per_lot":v16.COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":v16.CASHBACK_RT_PER_LOT},
      **v16.metrics(tr),
      "pattern_metrics":{k:v16.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}
    }
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
