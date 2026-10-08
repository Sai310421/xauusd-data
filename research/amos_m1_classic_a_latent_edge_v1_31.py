#!/usr/bin/env python3
"""AMOS M1 Classic A latent-edge feasibility audit v1.31.

Diagnostic only. Purpose: decide whether the remaining defect is Entry selection
or Exit/Target decoding before creating another live candidate.

Frozen:
- M1 only.
- v1.25 Classic A SELL entries are the sample.
- Raw Bid/Ask path is used after executable entry.
- No candidate is promoted from this audit.
- No fixed-RR trading rule is created.

Outputs for each A SELL:
- forward favorable/adverse excursion (MFE/MAE) over 1/2/3/5/10/15/30/60 min
- first-passage ordering for ATR-normalized diagnostic barriers
- whether current structural TP/SL failed despite favorable excursion.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys
from pathlib import Path
import numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v125",HERE/"amos_m1_path_extreme_precision_v1_25.py")
v125=importlib.util.module_from_spec(spec);sys.modules["v125"]=v125;spec.loader.exec_module(v125)
v24=v125.v24;v16=v125.v16

H=[1,2,3,5,10,15,30,60]
BARRIERS=[(0.25,0.25),(0.35,0.25),(0.50,0.25),(0.50,0.35),(0.75,0.35),(0.75,0.50),(1.00,0.50),(1.00,0.75)]

def first_passage(seg,entry,atr,tp_atr,sl_atr):
    # SELL: favorable is bid down; adverse is ask up
    tp=entry-tp_atr*atr; sl=entry+sl_atr*atr
    for _,q in seg.iterrows():
        hit_tp=float(q.bid)<=tp
        hit_sl=float(q.ask)>=sl
        if hit_tp and hit_sl:return "BOTH"
        if hit_tp:return "TP"
        if hit_sl:return "SL"
    return "NONE"

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=v125.v22.v16.load_raw(a.catalog) if hasattr(v125.v22.v16,"load_raw") else v125.v24.v16.v14.load_raw(a.catalog)
    # robust loader fallback from v1.26 convention
    if not isinstance(raw,pd.DataFrame):
        raise RuntimeError("raw loader did not return DataFrame")
    z=v16.v14.feature(v16.v14.bars(raw,1))
    ctx,counts=v125.v22.owned_mssplus_context(z)
    base,stats,audit=v125.path_extreme_entries(ctx,z,raw)
    A=base[base.pattern=="CLASSIC_A_SELL"].copy().reset_index(drop=True)
    rows=[]
    raw_ns=raw.ns.to_numpy()
    for _,r in A.iterrows():
        e=float(r.entry); ens=int(r.entry_ns); si=int(r.entry_bar_i); atr=max(float(z.atr.iloc[si]),1e-12)
        d=r.to_dict()
        for m in H:
            end=ens+int(m*60*1e9)
            i0=int(np.searchsorted(raw_ns,ens,side="left"));i1=int(np.searchsorted(raw_ns,end,side="right"))
            seg=raw.iloc[i0:i1]
            if len(seg):
                d[f"mfe_{m}m_atr"]=(e-float(seg.bid.min()))/atr
                d[f"mae_{m}m_atr"]=(float(seg.ask.max())-e)/atr
                d[f"close_{m}m_atr"]=(e-float(seg.bid.iloc[-1]))/atr
            else:
                d[f"mfe_{m}m_atr"]=np.nan;d[f"mae_{m}m_atr"]=np.nan;d[f"close_{m}m_atr"]=np.nan
        i0=int(np.searchsorted(raw_ns,ens,side="left"));i1=int(np.searchsorted(raw_ns,ens+int(60*60*1e9),side="right"))
        seg60=raw.iloc[i0:i1]
        for tp,sl in BARRIERS:
            d[f"fp_T{tp:.2f}_S{sl:.2f}"]=first_passage(seg60,e,atr,tp,sl)
        d["struct_target_atr"]=(e-float(r.tp))/atr
        d["struct_stop_atr"]=(float(r.sl)-e)/atr
        rows.append(d)
    df=pd.DataFrame(rows);df.to_csv(out/"classic_a_latent_edge.csv",index=False)
    summ={}
    for m in H:
        x=df[f"close_{m}m_atr"].dropna()
        summ[f"{m}m"]={
          "N":int(len(x)),
          "direction_correct_pct":float((x>0).mean()*100) if len(x) else 0,
          "median_forward_atr":float(x.median()) if len(x) else None,
          "median_MFE_atr":float(df[f"mfe_{m}m_atr"].median()),
          "median_MAE_atr":float(df[f"mae_{m}m_atr"].median())
        }
    fps={}
    for tp,sl in BARRIERS:
        c=df[f"fp_T{tp:.2f}_S{sl:.2f}"].value_counts().to_dict();n=len(df)
        tp_n=int(c.get("TP",0));sl_n=int(c.get("SL",0));both=int(c.get("BOTH",0));none=int(c.get("NONE",0))
        decisive=max(1,tp_n+sl_n)
        wr=100*tp_n/decisive
        pf_proxy=(tp_n*tp)/(sl_n*sl) if sl_n>0 else None
        fps[f"T{tp:.2f}_S{sl:.2f}"]={"TP":tp_n,"SL":sl_n,"BOTH":both,"NONE":none,
          "decisive_N":tp_n+sl_n,"first_pass_WR_pct":wr,"payoff_PF_proxy":pf_proxy}
    # find whether 120+ entries show latent directional edge at practical horizons
    result={"version":"v1.31","type":"DIAGNOSTIC_ONLY_LATENT_EDGE_FEASIBILITY",
      "sample":"v1.25 Classic A SELL executable Raw Bid entries","N":int(len(df)),
      "raw_ticks":int(len(raw)),"m1_bars":int(len(z)),"mssplus_counts":counts,"path_stats":dict(stats),
      "forward_distribution":summ,"first_passage_diagnostics":fps,
      "decision_rule":"If forward direction is weak and no barrier family approaches PF 1.2, repair Entry/A recognition. If forward direction is strong but current structural PF is weak, repair exit/target decoder.",
      "constraints":["M1 only","diagnostic, not promotion","no fixed-RR live rule","no M5/M15/M30/G75"]}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8");print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
