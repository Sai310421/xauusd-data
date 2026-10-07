#!/usr/bin/env python3
"""AMOS M1 Classic-A line-chart invalidation audit v1.28.

M1 only. Video/line-chart semantics are primary.
No M5/M15/M30/G75. No fixed RR. No fixed multiple TP. No HTF target.
Classic V BUY and non-A entries unchanged.

Hypothesis from v1.27:
A geometry + neckline improves direction accuracy sharply, but PF remains low
because A losses are oversized relative to the near M1 structural target.
v1.28 therefore tests structure-based Close-line invalidation for Classic A:
- anchor Close + small structural tolerance
- signal/shoulder Close + tolerance
- min(anchor-close stop, original wick stop)
Target remains causal M1 R1 Close pivot.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys
from pathlib import Path
import numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v127",HERE/"amos_m1_classic_a_geometry_target_v1_27.py")
v127=importlib.util.module_from_spec(spec);sys.modules["v127"]=v127;spec.loader.exec_module(v127)
v125=v127.v125;v16=v127.v16

def qualify(row,z,mode):
    ll,rl,prom,sym,neck,sc=v127.a_geometry(row,z)
    if mode=="BASE": return True
    if mode=="NECK": return sc<neck
    if mode=="GEOM_NECK": return prom>=0.55 and sym>=0.30 and sc<neck
    if mode=="LOOSE_NECK": return prom>=0.35 and sym>=0.20 and sc<neck
    return False

def line_stop(row,z,kind,buf_atr):
    ai=int(row.anchor_i);si=int(row.entry_bar_i);atr=max(float(z.atr.iloc[si]),1e-12)
    anchor_close=float(z.close.iloc[ai]); signal_open=float(z.open.iloc[si]); signal_close=float(z.close.iloc[si])
    orig=float(row.sl)
    if kind=="ANCHOR_CLOSE":
        cand=anchor_close+buf_atr*atr
    elif kind=="SIGNAL_OPEN":
        cand=max(signal_open,signal_close)+buf_atr*atr
    elif kind=="MIN_ANCHOR_WICK":
        cand=min(orig,anchor_close+buf_atr*atr)
    else:
        cand=orig
    # SELL stop must remain above executable entry
    return cand if cand>float(row.entry) else orig

def build(base,z,filter_mode,stop_kind,buf_atr):
    other=base[base.pattern!="CLASSIC_A_SELL"].copy()
    rows=[]
    for _,r in base[base.pattern=="CLASSIC_A_SELL"].iterrows():
        if not qualify(r,z,filter_mode): continue
        d=r.to_dict()
        d["sl"]=line_stop(r,z,stop_kind,buf_atr)
        d["risk"]=(float(d["entry"])-float(d["sl"]))*-1
        if d["risk"]<=0: continue
        tgt=v127.resolve_close_target(pd.Series(d),z,1,"setup")
        if tgt is None: continue
        d.update(tgt);rows.append(d)
    aa=pd.DataFrame(rows)
    if len(aa): aa=aa[[c for c in base.columns if c in aa.columns]]
    else: aa=base.iloc[:0].copy()
    return pd.concat([other,aa],ignore_index=True).sort_values("entry_ns").reset_index(drop=True)

def pack(ent,raw):
    tr=v16.simulate(ent,raw,1) if len(ent) else pd.DataFrame()
    def m(p=None):
        x=tr if p is None else tr[tr.pattern==p]
        return v16.metrics(x) if len(x) else v16.metrics(pd.DataFrame())
    return tr,{"ALL":m(),"CLASSIC_A_SELL":m("CLASSIC_A_SELL"),"CLASSIC_V_BUY":m("CLASSIC_V_BUY")}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=v127.v126.load_raw(a.catalog);z=v16.v14.feature(v16.v14.bars(raw,1));ctx,counts=v125.v22.owned_mssplus_context(z)
    base,stats,audit=v125.path_extreme_entries(ctx,z,raw)
    variants=[]
    for fm in ["NECK","GEOM_NECK","LOOSE_NECK"]:
        for sk in ["ANCHOR_CLOSE","SIGNAL_OPEN","MIN_ANCHOR_WICK"]:
            for b in [0.00,0.03,0.05,0.10]:
                variants.append((fm,sk,b))
    res={}
    for fm,sk,b in variants:
        name=f"{fm}_{sk}_B{int(round(b*100)):02d}"
        ent=build(base,z,fm,sk,b);tr,mp=pack(ent,raw)
        tr.to_csv(out/f"trades_{name}.csv",index=False)
        res[name]={"filter":fm,"stop_kind":sk,"buffer_atr":b,"selected_A":int((ent.pattern=="CLASSIC_A_SELL").sum()),**mp}
    # rank by PF with N>=10, and separately by N>=30
    def rank(min_n):
        rr=[]
        for k,v in res.items():
            a=v["CLASSIC_A_SELL"]; n=int(a["N"]); pf=float(a["PF"]) if a["PF"] is not None else 0.
            if n>=min_n: rr.append({"name":k,"N":n,"WR_pct":a["WR_pct"],"PF":pf,"Net_USD":a["Net_USD"],"DD_pct":a.get("MaxClosedDD_pct_initial")})
        return sorted(rr,key=lambda x:(x["PF"],x["WR_pct"],x["N"]),reverse=True)[:15]
    result={"version":"v1.28","verification":"PURE_M1_CLASSIC_A_LINE_CHART_INVALIDATION_RAW_BIDASK",
      "baseline_run":37694142813,"raw_ticks":len(raw),"m1_bars":len(z),"mssplus_counts":counts,"path_stats":dict(stats),
      "constraints":["M1 only","Classic V BUY unchanged","no M5/M15/M30/G75","no fixed RR","no fixed multiple TP","no HTF target","M1 R1 causal Close-pivot target"],
      "hypothesis":"v1.27 directional accuracy improved, but wick-based stop creates poor payoff; test line-chart structural invalidation.",
      "top_N10":rank(10),"top_N30":rank(30),"variants":res}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8");print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
