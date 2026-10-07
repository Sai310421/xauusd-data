#!/usr/bin/env python3
"""AMOS M1 Classic-A inverse geometry/clear-target audit v1.27.

Scope frozen:
- M1 ONLY; no M5/M15/M30/G75.
- Video is primary.
- No fixed RR / no fixed multiple TP / no HTF target.
- Classic V BUY and non-A entries remain unchanged.
- Baseline entry ownership is v1.25.

This run asks two inverse questions:
1) Is Classic-A entry recognition too weak? Test genuine A geometry / neckline break.
2) Is the current 2-bar Close-pivot target too coarse/far? Test nearer causal M1 Close structure.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys,math
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v126",HERE/"amos_m1_classic_a_inverse_v1_26.py")
v126=importlib.util.module_from_spec(spec);sys.modules["v126"]=v126;spec.loader.exec_module(v126)
v125=v126.v125;v24=v125.v24;v16=v125.v16

def causal_close_pivots(z,start_i,end_i,radius):
    out=[]
    r=max(1,int(radius))
    for confirm_i in range(max(start_i+2*r,2*r),min(end_i+1,len(z))):
        idx=confirm_i-r
        if idx-r<start_i: continue
        c=float(z.close.iloc[idx])
        left=z.close.iloc[idx-r:idx]
        right=z.close.iloc[idx+1:idx+r+1]
        if len(left)<r or len(right)<r: continue
        if c<float(left.min()) and c<=float(right.min()):
            out.append({"type":"L","idx":idx,"confirmed_i":confirm_i,"price":c,
                        "confirmed_time":pd.Timestamp(z.datetime.iloc[confirm_i])+pd.Timedelta(minutes=1)})
        elif c>float(left.max()) and c>=float(right.max()):
            out.append({"type":"H","idx":idx,"confirmed_i":confirm_i,"price":c,
                        "confirmed_time":pd.Timestamp(z.datetime.iloc[confirm_i])+pd.Timedelta(minutes=1)})
    return out

def resolve_close_target(e,z,radius=1,scope="setup"):
    di=int(e.entry_dir);ep=float(e.entry);et=pd.Timestamp(e.entry_time)
    if scope=="setup": start=int(e.context_sweep_i)
    elif scope=="mss": start=max(0,int(e.context_mss_i)-20)
    else: start=max(0,int(e.entry_bar_i)-40)
    end=int(e.entry_bar_i);want="H" if di>0 else "L";c=[]
    for p in causal_close_pivots(z,start,end,radius):
        if p["type"]!=want or pd.Timestamp(p["confirmed_time"])>et: continue
        px=float(p["price"]);dist=(px-ep) if di>0 else (ep-px)
        if dist>0:c.append((dist,p))
    if not c:return None
    c.sort(key=lambda x:(x[0],-x[1]["idx"]))
    _,p=c[0]
    return {"tp":float(p["price"]),"target_kind":f"CLOSE_PIVOT_R{radius}_{want}",
            "target_ref_i":int(p["idx"]),"target_ref_time":pd.Timestamp(p["confirmed_time"])}

def a_geometry(row,z):
    ai=int(row.anchor_i);si=int(row.entry_bar_i);atr=max(float(z.atr.iloc[si]),1e-12)
    poi=float(row.poi)
    l0=max(int(row.context_sweep_i),ai-5)
    left=z.close.iloc[l0:ai]
    right=z.close.iloc[ai+1:si+1]
    left_leg=(poi-float(left.min()))/atr if len(left) else 0.
    right_leg=(poi-float(right.min()))/atr if len(right) else 0.
    prominence=min(left_leg,right_leg)
    symmetry=min(left_leg,right_leg)/max(left_leg,right_leg) if max(left_leg,right_leg)>0 else 0.
    neckline=float(left.min()) if len(left) else poi
    signal_close=float(z.close.iloc[si])
    return left_leg,right_leg,prominence,symmetry,neckline,signal_close

def filter_a(ent,z,mode):
    a=ent[ent.pattern=="CLASSIC_A_SELL"].copy()
    if a.empty:return a
    rows=[]
    for _,r in a.iterrows():
        ll,rl,prom,sym,neck,sc=a_geometry(r,z)
        d=r.to_dict();d.update({"a_left_leg_atr":ll,"a_right_leg_atr":rl,"a_prominence_atr":prom,
                                "a_symmetry":sym,"a_neckline":neck,"a_signal_close":sc})
        rows.append(d)
    q=pd.DataFrame(rows)
    if mode=="BASE":pass
    elif mode=="GAP07": q=q[((q.poi-q.entry)/q.signal_atr if "signal_atr" in q else (q.poi-q.entry)/1)>=0]  # replaced below
    elif mode=="A_GEOM": q=q[(q.a_prominence_atr>=0.55)&(q.a_symmetry>=0.30)]
    elif mode=="A_STRONG": q=q[(q.a_prominence_atr>=0.80)&(q.a_symmetry>=0.40)]
    elif mode=="NECK_BREAK": q=q[q.a_signal_close<q.a_neckline]
    elif mode=="GEOM_NECK": q=q[(q.a_prominence_atr>=0.55)&(q.a_symmetry>=0.30)&(q.a_signal_close<q.a_neckline)]
    return q

def add_diag_cols(ent,z):
    if ent.empty:return ent
    q=ent.copy()
    q["signal_atr"]=[float(z.atr.iloc[int(i)]) for i in q.entry_bar_i]
    q["poi_gap_atr"]=(q.poi-q.entry)/q.signal_atr
    return q

def build_variant(base,z,raw,filter_mode,target_mode):
    other=base[base.pattern!="CLASSIC_A_SELL"].copy()
    ad=add_diag_cols(base[base.pattern=="CLASSIC_A_SELL"].copy(),z)
    # geometry filtering
    if filter_mode=="GAP07":
        aa=ad[ad.poi_gap_atr>=0.70].copy()
    elif filter_mode=="BASE":
        aa=ad.copy()
    else:
        geom=filter_a(ad,z,filter_mode)
        keep=set(geom.context_id.astype(int).tolist())
        aa=ad[ad.context_id.astype(int).isin(keep)].copy()
    rows=[]
    for _,r in aa.iterrows():
        d=r.to_dict()
        if target_mode=="R2_SETUP":
            tgt=v24.resolve_structure_target(pd.Series(d),z)
        elif target_mode=="R1_SETUP":
            tgt=resolve_close_target(pd.Series(d),z,1,"setup")
        elif target_mode=="R1_MSS40":
            tgt=resolve_close_target(pd.Series(d),z,1,"mss")
        elif target_mode=="R1_PRE40":
            tgt=resolve_close_target(pd.Series(d),z,1,"pre40")
        else: tgt=None
        if tgt is None:continue
        d.update(tgt);rows.append(d)
    aa2=pd.DataFrame(rows)
    if len(aa2):
        aa2=aa2[[c for c in base.columns if c in aa2.columns]]
    else:aa2=base.iloc[:0].copy()
    out=pd.concat([other,aa2],ignore_index=True).sort_values("entry_ns").reset_index(drop=True)
    return out

def pack(ent,raw):
    tr=v16.simulate(ent,raw,1) if len(ent) else pd.DataFrame()
    def mm(pat=None):
        x=tr if pat is None else tr[tr.pattern==pat]
        return v16.metrics(x) if len(x) else v16.metrics(pd.DataFrame())
    return tr,{"ALL":mm(),"CLASSIC_A_SELL":mm("CLASSIC_A_SELL"),"CLASSIC_V_BUY":mm("CLASSIC_V_BUY")}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=v126.load_raw(a.catalog);z=v16.v14.feature(v16.v14.bars(raw,1));ctx,counts=v125.v22.owned_mssplus_context(z)
    base,stats,audit=v125.path_extreme_entries(ctx,z,raw)
    diagnostics=add_diag_cols(base[base.pattern=="CLASSIC_A_SELL"].copy(),z)
    grows=[]
    for _,r in diagnostics.iterrows():
        ll,rl,prom,sym,neck,sc=a_geometry(r,z);d=r.to_dict()
        d.update({"a_left_leg_atr":ll,"a_right_leg_atr":rl,"a_prominence_atr":prom,"a_symmetry":sym,
                  "a_neckline":neck,"a_signal_close":sc})
        grows.append(d)
    pd.DataFrame(grows).to_csv(out/"classic_a_geometry.csv",index=False)

    variants=[
      ("BASE_R2","BASE","R2_SETUP"),
      ("BASE_R1","BASE","R1_SETUP"),
      ("BASE_R1_MSS40","BASE","R1_MSS40"),
      ("BASE_R1_PRE40","BASE","R1_PRE40"),
      ("GAP07_R1","GAP07","R1_SETUP"),
      ("A_GEOM_R1","A_GEOM","R1_SETUP"),
      ("A_STRONG_R1","A_STRONG","R1_SETUP"),
      ("NECK_BREAK_R1","NECK_BREAK","R1_SETUP"),
      ("GEOM_NECK_R1","GEOM_NECK","R1_SETUP"),
    ]
    res={}
    for name,fm,tm in variants:
        ent=build_variant(base,z,raw,fm,tm);tr,mp=pack(ent,raw)
        ent.to_csv(out/f"entries_{name}.csv",index=False);tr.to_csv(out/f"trades_{name}.csv",index=False)
        res[name]={"filter":fm,"target":tm,"selected_A":int((ent.pattern=="CLASSIC_A_SELL").sum()),**mp}
    result={"version":"v1.27","verification":"PURE_M1_CLASSIC_A_GEOMETRY_AND_CAUSAL_CLEAR_TARGET_RAW_BIDASK",
      "baseline_run":37694142813,"raw_ticks":len(raw),"m1_bars":len(z),"mssplus_counts":counts,"path_stats":dict(stats),
      "constraints":["M1 only","Classic V BUY unchanged","no M5/M15/M30/G75","no fixed RR","no fixed multiple TP","no HTF target"],
      "target_note":"R1 targets are nearest causally confirmed one-bar Close pivots; they are structural prices, not RR multiples.",
      "variants":res}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8");print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
