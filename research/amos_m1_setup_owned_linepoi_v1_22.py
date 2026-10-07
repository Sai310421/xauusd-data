#!/usr/bin/env python3
"""AMOS M1 setup-owned line-POI decoder v1.22.

Fixes the remaining ownership error in v1.21.

Source of truth:
- Video framework is primary.
- Line chart + POI are only the decoder for the concrete entry location.
- A setup may use ONLY line structure formed by that same setup.
- No pre-existing/global V/A/QM/OCL structure may be attached afterward.

Executable M1 proxy kept unchanged for the framework core:
  Sweep -> CISD -> MSS -> Displacement -> MSS+

Ownership rule added here:
  context starts at its own Sweep
  -> all V/A/QM/OCL anchors must be formed at/after that Sweep
  -> model may complete only when the context is active
  -> the first owned terminal model physically in the delivery path becomes the POI
  -> entry is the first raw Bid/Ask quote after causal model confirmation.

No M5/M15/G75 changes. No pattern ranking/cherry-picking/freshness optimization.
"""
from __future__ import annotations
import argparse,json,math,importlib.util,sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

spec=importlib.util.spec_from_file_location("v16",Path(__file__).with_name("amos_mssplus_destination_poi_v1_16.py"))
v16=importlib.util.module_from_spec(spec);sys.modules["v16"]=v16;spec.loader.exec_module(v16)

def owned_mssplus_context(z):
    c=Counter();states={1:None,-1:None};out=[]
    for i in range(30,len(z)):
        r=z.iloc[i];p=z.iloc[i-1]
        if not np.isfinite(r.atr) or r.atr<=0:continue
        for side in (1,-1):
            s=states[side]
            if s and i-s["sweep_i"]>28:states[side]=None
        bull=bool(r.low<r.swing_lo and r.close>r.swing_lo)
        bear=bool(r.high>r.swing_hi and r.close<r.swing_hi)
        if bull:
            states[1]={"sweep_i":i,"extreme":float(r.low),"mss_ref":float(z.high.iloc[max(0,i-3):i].max()),"cisd_i":None,"mss_i":None,"disp_seen":False};c["sweep"]+=1
        if bear:
            states[-1]={"sweep_i":i,"extreme":float(r.high),"mss_ref":float(z.low.iloc[max(0,i-3):i].min()),"cisd_i":None,"mss_i":None,"disp_seen":False};c["sweep"]+=1
        for side in (1,-1):
            s=states[side]
            if not s:continue
            if s["cisd_i"] is None:
                if i>s["sweep_i"] and i-s["sweep_i"]<=8:
                    hit=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
                    if hit:s["cisd_i"]=i;c["cisd"]+=1
                elif i-s["sweep_i"]>8:states[side]=None
                continue
            if s["mss_i"] is None:
                if i-s["cisd_i"]>12:states[side]=None;continue
                if i<=s["cisd_i"]:continue
                hit=(r.close>s["mss_ref"]) if side==1 else (r.close<s["mss_ref"])
                if hit:s["mss_i"]=i;c["mss"]+=1
                else:continue
            if i<s["mss_i"] or i-s["mss_i"]>4:
                if i-s["mss_i"]>4:states[side]=None
                continue
            if float(r.disp)<.65:continue
            if not s["disp_seen"]:c["displacement"]+=1;s["disp_seen"]=True
            structural_poi=v16.v14.poi_zone(z,i,side)
            if structural_poi is None:continue
            c["mss_plus"]+=1
            out.append({
                "mssplus_time":pd.Timestamp(r.datetime)+pd.Timedelta(minutes=1),
                "travel_dir":int(side),"mssplus_close":float(r.close),
                "sweep_i":int(s["sweep_i"]),"cisd_i":int(s["cisd_i"]),"mss_i":int(s["mss_i"]),
                "disp_i":int(i),"sweep_time":pd.Timestamp(z.iloc[s["sweep_i"]].datetime),
                "sweep_extreme":float(s["extreme"]),"mssplus_poi_type":str(structural_poi[2])
            })
            states[side]=None
    return pd.DataFrame(out),dict(c)

def context_models(z,cx):
    """Build line models using ONLY anchors created inside this context."""
    start=int(cx.sweep_i); end=min(len(z)-1,int(cx.disp_i)+60)
    piv=[];seen=set();models=[]
    for i in range(max(start+4,24),end+1):
        idx=i-2
        # Pivot itself and both confirmation bars must belong to this setup.
        if idx-2 < start:continue
        c0=float(z.close.iloc[idx])
        if c0<float(z.close.iloc[idx-2:idx].min()) and c0<=float(z.close.iloc[idx+1:idx+3].min()):
            piv.append({"type":"L","idx":idx,"price":c0})
        elif c0>float(z.close.iloc[idx-2:idx].max()) and c0>=float(z.close.iloc[idx+1:idx+3].max()):
            piv.append({"type":"H","idx":idx,"price":c0})
        piv=[p for p in piv if p["idx"]>=start][-80:]
        r=z.iloc[i];a=float(r.atr)
        if not np.isfinite(a) or a<=0:continue
        cc=float(r.close);pc=float(z.close.iloc[i-1])
        def add(name,di,poi,sl,anchor_idx,key):
            if key in seen:return
            if anchor_idx<start:return
            if (di>0 and not sl<poi) or (di<0 and not sl>poi):return
            seen.add(key)
            models.append({"pattern":name,"dir":di,"poi":float(poi),"sl":float(sl),"atr":a,
                           "signal_time":pd.Timestamp(r.datetime),"signal_i":i,"anchor_i":anchor_idx})
        lows=[p for p in piv if p["type"]=="L" and 2<=i-p["idx"]<=8]
        if lows:
            q=lows[-1]
            if cc>pc and cc>float(z.close.iloc[i-2]):
                sl=float(z.low.iloc[max(start,q["idx"]-1):q["idx"]+2].min())-.10*a
                add("CLASSIC_V_BUY",1,q["price"],sl,q["idx"],f"CVB:{q['idx']}")
        highs=[p for p in piv if p["type"]=="H" and 2<=i-p["idx"]<=8]
        if highs:
            q=highs[-1]
            if cc<pc and cc<float(z.close.iloc[i-2]):
                sl=float(z.high.iloc[max(start,q["idx"]-1):q["idx"]+2].max())+.10*a
                add("CLASSIC_A_SELL",-1,q["price"],sl,q["idx"],f"CAS:{q['idx']}")
        if len(piv)>=3:
            x,y,w=piv[-3],piv[-2],piv[-1]
            if min(x["idx"],y["idx"],w["idx"])>=start:
                if x["type"]=="L" and y["type"]=="H" and w["type"]=="L" and w["price"]<x["price"]-.05*a and cc>y["price"]:
                    sl=float(z.low.iloc[max(start,w["idx"]-1):w["idx"]+2].min())-.10*a
                    add("QM_BUY",1,x["price"],sl,x["idx"],f"QMB:{x['idx']}:{w['idx']}")
                if x["type"]=="H" and y["type"]=="L" and w["type"]=="H" and w["price"]>x["price"]+.05*a and cc<y["price"]:
                    sl=float(z.high.iloc[max(start,w["idx"]-1):w["idx"]+2].max())+.10*a
                    add("QM_SELL",-1,x["price"],sl,x["idx"],f"QMS:{x['idx']}:{w['idx']}")
        # OCL body pair must also be inside this same setup.
        if i-1>=start and i-5>=start:
            po=float(z.open.iloc[i-1]);ph=float(z.high.iloc[i-1]);pl=float(z.low.iloc[i-1]);pcl=float(z.close.iloc[i-1])
            avg=float((z.close.iloc[max(start,i-20):i]-z.open.iloc[max(start,i-20):i]).abs().mean())
            body=abs(cc-float(r.open))
            if avg>0 and body>=avg:
                local_lo=float(z.low.iloc[i-5:i+1].min());local_hi=float(z.high.iloc[i-5:i+1].max())
                if pcl<po and cc>float(r.open) and cc>po and min(float(r.low),pl)<=local_lo+.15*a:
                    add("OCL_BUY",1,max(po,pcl),min(local_lo,pl)-.10*a,i-1,f"OCLB:{i-1}")
                if pcl>po and cc<float(r.open) and cc<po and max(float(r.high),ph)>=local_hi-.15*a:
                    add("OCL_SELL",-1,min(po,pcl),max(local_hi,ph)+.10*a,i-1,f"OCLS:{i-1}")
    return pd.DataFrame(models)

def add_target(row,z):
    i=int(row.signal_i); di=int(row.dir); a=float(row.atr); poi=float(row.poi)
    look=z.iloc[max(0,i-40):i]
    if di>0:
        opp=float(look.close.max()) if len(look) else poi+2*a
        return max(opp,poi+2*a)
    opp=float(look.close.min()) if len(look) else poi-2*a
    return min(opp,poi-2*a)

def owned_entries(ctx,z,raw):
    stats=Counter();out=[];all_models=[]
    raw_ns=raw.ns.to_numpy()
    for cid,cx in ctx.sort_values("mssplus_time").reset_index(drop=True).iterrows():
        stats["framework_armed"]+=1
        models=context_models(z,cx)
        if models.empty:
            stats["no_owned_line_model"]+=1;continue
        models=models.copy();models["target"]=[add_target(r,z) for _,r in models.iterrows()]
        models["context_id"]=cid;all_models.append(models)
        t0=pd.Timestamp(cx.mssplus_time);t1=t0+pd.Timedelta(minutes=60)
        travel=int(cx.travel_dir);entry_dir=-travel;origin=float(cx.mssplus_close)

        # Model must COMPLETE while/after MSS+ is active, but its anchors must have
        # been born inside the same setup (enforced by context_models).
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
        tp=float(s.target)
        if not np.isfinite(tp) or (tp-ep)*entry_dir<=0:tp=ep+entry_dir*2*risk
        out.append({"tf":1,"context_id":int(cid),"sweep_time":cx.sweep_time,"mssplus_time":t0,
                    "travel_dir":travel,"entry_dir":entry_dir,"mssplus_close":origin,
                    "pattern":str(s.pattern),"poi":float(s.poi),"anchor_i":int(s.anchor_i),
                    "poi_signal_time":pd.Timestamp(s.signal_time),"entry_ns":int(first.ns),"entry_time":first.datetime,
                    "entry":ep,"sl":float(s.sl),"tp":tp,"risk":risk,"mssplus_poi_type":str(cx.mssplus_poi_type)})
        stats["entries"]+=1;stats["entry_"+str(s.pattern)]+=1
    return pd.DataFrame(out),stats,(pd.concat(all_models,ignore_index=True) if all_models else pd.DataFrame())

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks.sort(key=lambda x:int(x.ts_event))
    ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
    bid=np.fromiter((v16.v14.fpx(x.bid_price) for x in ticks),float,count=len(ticks))
    ask=np.fromiter((v16.v14.fpx(x.ask_price) for x in ticks),float,count=len(ticks))
    raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"ns":ts,"bid":bid,"ask":ask})
    z=v16.v14.feature(v16.v14.bars(raw,1))
    ctx,counts=owned_mssplus_context(z)
    ent,stats,owned=owned_entries(ctx,z,raw)
    tr=v16.simulate(ent,raw,1) if len(ent) else pd.DataFrame()
    ctx.to_csv(out/"framework_contexts.csv",index=False)
    owned.to_csv(out/"owned_line_models.csv",index=False)
    ent.to_csv(out/"owned_destination_entries.csv",index=False)
    tr.to_csv(out/"trades.csv",index=False)
    result={"version":"v1.22","verification":"M1_VIDEO_SETUP_OWNED_LINE_POI_RAW_BIDASK",
      "timeframe_min":1,"m5_used":False,"m15_used":False,"g75_used":False,
      "raw_ticks":len(raw),"m1_bars":len(z),"mssplus_counts":counts,
      "owned_line_models":len(owned),"ownership_stats":dict(stats),
      "architecture":"video setup starts at its own Sweep -> CISD -> MSS -> Displacement/MSS+ -> only line anchors born inside same setup are eligible -> first owned terminal V/A/QM/OCL in delivery path -> raw Bid/Ask entry",
      "key_change_from_v121":"Removed global/pre-existing line-model attachment. Every pivot/body anchor must be born at or after that context's own Sweep.",
      "direction_rule":"bearish delivery -> lower BUY POI -> BUY; bullish delivery -> upper SELL POI -> SELL",
      "entry_timing":"first raw Bid/Ask quote after owned terminal line-model confirmation",
      "cost_assumption":{"commission_rt_per_lot":v16.COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":v16.CASHBACK_RT_PER_LOT},
      **v16.metrics(tr),"pattern_metrics":{k:v16.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
