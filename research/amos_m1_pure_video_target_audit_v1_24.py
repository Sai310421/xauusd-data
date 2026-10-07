#!/usr/bin/env python3
"""AMOS pure-M1 video entry/target audit v1.24.

Scope is M1 ONLY.
- No M5.
- No M15/30 reference map.
- No G75.
- No fixed-multiple / synthetic TP fallback.

Entry side:
  Uses v1.22 setup-owned video path / line-chart POI decoder.

Target side:
  The supplied videos/checklist show Clear Target / FVG Target, but do not
  uniquely prove one proprietary formula. Therefore v1.24 does NOT silently
  choose one. It evaluates two pure-M1, causal target decoders separately:

  A) M1_FVG_TARGET:
     nearest still-clear M1 FVG zone ahead which already existed before entry.

  B) M1_STRUCTURE_TARGET:
     nearest opposite Close-line pivot ahead, born inside the SAME setup,
     and causally confirmed before entry.

There is NO RR fallback. If the selected M1 target is absent, the trade is
excluded from that variant's official KPI.

The purpose is to identify which M1 target interpretation matches the video
behavior before freezing the exit logic.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

spec=importlib.util.spec_from_file_location("v22",Path(__file__).with_name("amos_m1_setup_owned_linepoi_v1_22.py"))
v22=importlib.util.module_from_spec(spec);sys.modules["v22"]=v22;spec.loader.exec_module(v22)
v16=v22.v16

def causal_pivots(z,start_i,end_i):
    out=[]
    for i in range(max(start_i+4,4),min(end_i+1,len(z))):
        idx=i-2
        if idx-2<start_i:continue
        c=float(z.close.iloc[idx])
        if c<float(z.close.iloc[idx-2:idx].min()) and c<=float(z.close.iloc[idx+1:idx+3].min()):
            out.append({"type":"L","idx":idx,"confirmed_i":i,"price":c,
                        "confirmed_time":pd.Timestamp(z.datetime.iloc[i])+pd.Timedelta(minutes=1)})
        elif c>float(z.close.iloc[idx-2:idx].max()) and c>=float(z.close.iloc[idx+1:idx+3].max()):
            out.append({"type":"H","idx":idx,"confirmed_i":i,"price":c,
                        "confirmed_time":pd.Timestamp(z.datetime.iloc[i])+pd.Timedelta(minutes=1)})
    return out

def m1_fvgs(z,start_i,end_i):
    rows=[]
    for i in range(max(start_i+2,2),min(end_i+1,len(z))):
        r=z.iloc[i]
        formed=pd.Timestamp(r.datetime)+pd.Timedelta(minutes=1)
        if float(r.low)>float(z.high.iloc[i-2]):
            lo=float(z.high.iloc[i-2]);hi=float(r.low)
            rows.append({"formed_i":i,"formed_time":formed,"lo":min(lo,hi),"hi":max(lo,hi),"kind":"BULL_FVG"})
        if float(r.high)<float(z.low.iloc[i-2]):
            lo=float(r.high);hi=float(z.low.iloc[i-2])
            rows.append({"formed_i":i,"formed_time":formed,"lo":min(lo,hi),"hi":max(lo,hi),"kind":"BEAR_FVG"})
    return rows

def fvg_clear_until(z,g,entry_time):
    q=z[(z.datetime>=pd.Timestamp(g["formed_time"]))&(z.datetime<entry_time)]
    if q.empty:return True
    lo=float(g["lo"]);hi=float(g["hi"])
    return not bool(((q.high>=lo)&(q.low<=hi)).any())

def resolve_fvg_target(e,z):
    di=int(e.entry_dir);ep=float(e.entry);et=pd.Timestamp(e.entry_time)
    start=int(e.context_sweep_i);end=int(e.entry_bar_i)
    c=[]
    for g in m1_fvgs(z,start,end):
        if pd.Timestamp(g["formed_time"])>et:continue
        lo=float(g["lo"]);hi=float(g["hi"])
        if di>0:
            if lo<=ep:continue
            target=lo;dist=target-ep
        else:
            if hi>=ep:continue
            target=hi;dist=ep-target
        if dist<=0 or not fvg_clear_until(z,g,et):continue
        c.append((dist,target,g))
    if not c:return None
    c.sort(key=lambda x:(x[0],x[2]["formed_i"]))
    _,target,g=c[0]
    return {"tp":float(target),"target_kind":g["kind"],"target_ref_i":int(g["formed_i"]),
            "target_ref_time":pd.Timestamp(g["formed_time"])}

def resolve_structure_target(e,z):
    di=int(e.entry_dir);ep=float(e.entry);et=pd.Timestamp(e.entry_time)
    start=int(e.context_sweep_i);end=int(e.entry_bar_i)
    want="H" if di>0 else "L";c=[]
    for p in causal_pivots(z,start,end):
        if p["type"]!=want or pd.Timestamp(p["confirmed_time"])>et:continue
        px=float(p["price"])
        dist=(px-ep) if di>0 else (ep-px)
        if dist>0:c.append((dist,p))
    if not c:return None
    c.sort(key=lambda x:(x[0],x[1]["idx"]))
    _,p=c[0]
    return {"tp":float(p["price"]),"target_kind":"CLOSE_PIVOT_"+want,
            "target_ref_i":int(p["idx"]),"target_ref_time":pd.Timestamp(p["confirmed_time"])}

def build_entry_candidates(ctx,z,raw):
    """Same setup-owned entry logic as v1.22, but emit M1 context indices."""
    stats=Counter();out=[];owned_all=[];raw_ns=raw.ns.to_numpy()
    for cid,cx in ctx.sort_values("mssplus_time").reset_index(drop=True).iterrows():
        stats["framework_armed"]+=1
        models=v22.context_models(z,cx)
        if models.empty:
            stats["no_owned_line_model"]+=1;continue
        models=models.copy();models["context_id"]=cid;owned_all.append(models)
        t0=pd.Timestamp(cx.mssplus_time);t1=t0+pd.Timedelta(minutes=60)
        travel=int(cx.travel_dir);entry_dir=-travel;origin=float(cx.mssplus_close)
        formed=models[(models.dir==entry_dir)&(models.signal_time>=t0)&(models.signal_time<=t1)].sort_values("signal_time")
        if formed.empty:
            stats["no_owned_terminal_model_after_mssplus"]+=1;continue
        chosen=None
        for _,s in formed.iterrows():
            ahead=(float(s.poi)<origin) if travel<0 else (float(s.poi)>origin)
            if not ahead:
                stats["owned_model_not_in_delivery_path"]+=1;continue
            chosen=s;break
        if chosen is None:
            stats["no_owned_poi_in_delivery_path"]+=1;continue
        s=chosen;stats["destination_poi_resolved"]+=1;stats["resolved_"+str(s.pattern)]+=1
        confirm_end=pd.Timestamp(s.signal_time)+pd.Timedelta(minutes=1)
        j=int(np.searchsorted(raw_ns,int(confirm_end.value),side="left"))
        if j>=len(raw):stats["no_raw_quote"]+=1;continue
        first=raw.iloc[j];ep=float(first.ask if entry_dir>0 else first.bid)
        risk=(ep-float(s.sl))*entry_dir
        if risk<=0:stats["bad_risk"]+=1;continue
        entry_bar_i=int(np.searchsorted(z.datetime.to_numpy(),np.datetime64(pd.Timestamp(s.signal_time)),side="right")-1)
        out.append({"tf":1,"context_id":int(cid),"context_sweep_i":int(cx.sweep_i),
                    "context_mss_i":int(cx.mss_i),"context_disp_i":int(cx.disp_i),
                    "sweep_time":cx.sweep_time,"mssplus_time":t0,
                    "travel_dir":travel,"entry_dir":entry_dir,"mssplus_close":origin,
                    "pattern":str(s.pattern),"poi":float(s.poi),"anchor_i":int(s.anchor_i),
                    "poi_signal_time":pd.Timestamp(s.signal_time),"entry_bar_i":entry_bar_i,
                    "entry_ns":int(first.ns),"entry_time":first.datetime,
                    "entry":ep,"sl":float(s.sl),"risk":risk,
                    "mssplus_poi_type":str(cx.mssplus_poi_type)})
        stats["entries"]+=1;stats["entry_"+str(s.pattern)]+=1
    return pd.DataFrame(out),stats,(pd.concat(owned_all,ignore_index=True) if owned_all else pd.DataFrame())

def attach_targets(entries,z,resolver,name):
    stats=Counter();rows=[]
    for _,e in entries.iterrows():
        stats["entry_candidates"]+=1
        t=resolver(e,z)
        if t is None:
            stats["reject_no_"+name]+=1;continue
        r=e.to_dict();r.update(t);rows.append(r)
        stats["target_resolved"]+=1
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
    entries,entry_stats,owned=build_entry_candidates(ctx,z,raw)

    e_fvg,s_fvg=attach_targets(entries,z,resolve_fvg_target,"m1_fvg_target")
    e_str,s_str=attach_targets(entries,z,resolve_structure_target,"m1_structure_target")
    tr_fvg=v16.simulate(e_fvg,raw,1) if len(e_fvg) else pd.DataFrame()
    tr_str=v16.simulate(e_str,raw,1) if len(e_str) else pd.DataFrame()

    ctx.to_csv(out/"framework_contexts.csv",index=False)
    owned.to_csv(out/"owned_line_models.csv",index=False)
    entries.to_csv(out/"entry_candidates.csv",index=False)
    e_fvg.to_csv(out/"entries_m1_fvg_target.csv",index=False)
    e_str.to_csv(out/"entries_m1_structure_target.csv",index=False)
    tr_fvg.to_csv(out/"trades_m1_fvg_target.csv",index=False)
    tr_str.to_csv(out/"trades_m1_structure_target.csv",index=False)

    result={
      "version":"v1.24","verification":"PURE_M1_VIDEO_ENTRY_AND_TARGET_AUDIT_RAW_BIDASK",
      "timeframe_min":1,"m5_used":False,"m15_used":False,"m30_used":False,"g75_used":False,
      "raw_ticks":len(raw),"m1_bars":len(z),"mssplus_counts":counts,
      "entry_stats":dict(entry_stats),"owned_line_models":len(owned),
      "target_policy":"PURE M1 ONLY; NO RR; NO fixed/synthetic TP fallback",
      "variants":{
        "M1_FVG_TARGET":{"target_stats":dict(s_fvg),**v16.metrics(tr_fvg),
          "pattern_metrics":{k:v16.metrics(v) for k,v in tr_fvg.groupby("pattern")} if len(tr_fvg) else {}},
        "M1_STRUCTURE_TARGET":{"target_stats":dict(s_str),**v16.metrics(tr_str),
          "pattern_metrics":{k:v16.metrics(v) for k,v in tr_str.groupby("pattern")} if len(tr_str) else {}}
      },
      "interpretation_note":"Two M1-only target decoders are kept separate because the supplied video checklist names Clear Target/FVG Target but does not uniquely prove one proprietary formula. No higher-timeframe target is used."
    }
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
