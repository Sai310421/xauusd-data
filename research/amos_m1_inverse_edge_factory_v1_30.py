#!/usr/bin/env python3
"""AMOS M1 Inverse EDGE Factory v1.30 — target-count inversion.

Purpose:
- Start from desired Classic A SELL EDGE, not from arbitrary filters.
- Preserve the high-WR A-shape kernel found in v1.27-v1.29, then broaden it
  continuously until target trade count is recovered.
- Compare structurally significant causal M1 Close targets (no fixed RR).

Frozen scope:
M1 only; no M5/M15/M30/G75; no fixed RR; no HTF target; Classic V unchanged.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys,math
from pathlib import Path
import numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v129",HERE/"amos_m1_inverse_edge_factory_v1_29.py")
v129=importlib.util.module_from_spec(spec);sys.modules["v129"]=v129;spec.loader.exec_module(v129)
v128=v129.v128;v127=v129.v127;v125=v129.v125;v16=v129.v16

TARGET_N=120;TARGET_WR=45.0;TARGET_PF=1.20

def geom_features(r,z):
    ll,rl,prom,sym,neck,sc=v127.a_geometry(r,z)
    si=int(r.entry_bar_i);atr=max(float(z.atr.iloc[si]),1e-12)
    neck_break=(neck-sc)/atr
    anchor_gap=(float(r.poi)-float(r.entry))/atr
    # quality intentionally favors the v1.27 high-WR kernel, but remains continuous.
    score=1.40*max(prom,0.0)+0.90*max(sym,0.0)+1.20*max(neck_break,0.0)+0.25*max(anchor_gap,0.0)
    return {"a_prominence_atr":prom,"a_symmetry":sym,"neck_break_atr":neck_break,
            "anchor_gap_atr":anchor_gap,"quality_score":score}

def causal_pivots_r1(z,start,end):
    out=[]
    for ci in range(max(start+2,2),min(end+1,len(z))):
        idx=ci-1;c=float(z.close.iloc[idx])
        if idx-1<start or idx+1>=len(z):continue
        if c<float(z.close.iloc[idx-1]) and c<=float(z.close.iloc[idx+1]):
            out.append(("L",idx,ci,c))
        elif c>float(z.close.iloc[idx-1]) and c>=float(z.close.iloc[idx+1]):
            out.append(("H",idx,ci,c))
    return out

def significant_target(r,z,min_prom):
    di=int(r.entry_dir);ep=float(r.entry);et=pd.Timestamp(r.entry_time)
    start=int(r.context_sweep_i);end=int(r.entry_bar_i);want="L" if di<0 else "H"
    atr=max(float(z.atr.iloc[end]),1e-12);cands=[]
    for typ,idx,ci,px in causal_pivots_r1(z,start,end):
        if typ!=want:continue
        ctime=pd.Timestamp(z.datetime.iloc[ci])+pd.Timedelta(minutes=1)
        if ctime>et:continue
        dist=(ep-px) if di<0 else (px-ep)
        if dist<=0:continue
        lo=max(start,idx-2);hi=min(len(z)-1,idx+2)
        if typ=="L":
            shoulder=min(float(z.close.iloc[lo:idx].max()) if idx>lo else px,
                         float(z.close.iloc[idx+1:hi+1].max()) if hi>idx else px)
            prom=(shoulder-px)/atr
        else:
            shoulder=max(float(z.close.iloc[lo:idx].min()) if idx>lo else px,
                         float(z.close.iloc[idx+1:hi+1].min()) if hi>idx else px)
            prom=(px-shoulder)/atr
        if prom+1e-12<min_prom:continue
        cands.append((dist,-prom,-idx,px,idx,ctime,prom))
    if not cands:return None
    cands.sort()
    _,_,_,px,idx,ctime,prom=cands[0]
    return {"tp":float(px),"target_kind":f"R1_SIGNIFICANT_{want}_P{min_prom:.2f}",
            "target_ref_i":int(idx),"target_ref_time":ctime,"target_prominence_atr":float(prom)}

def build(base,z,retain_frac,min_prom,stop_buf=0.10):
    other=base[base.pattern!="CLASSIC_A_SELL"].copy()
    A=base[base.pattern=="CLASSIC_A_SELL"].copy()
    feats=[]
    for ix,r in A.iterrows():
        f=geom_features(r,z);f["_ix"]=ix;feats.append(f)
    fd=pd.DataFrame(feats).sort_values("quality_score",ascending=False)
    keep_n=max(1,int(round(len(fd)*retain_frac)))
    keep=set(fd.head(keep_n)["_ix"].tolist())
    rows=[]
    for ix,r in A.iterrows():
        if ix not in keep:continue
        d=r.to_dict()
        ai=int(r.anchor_i);si=int(r.entry_bar_i);atr=max(float(z.atr.iloc[si]),1e-12)
        sl=float(z.close.iloc[ai])+stop_buf*atr
        if sl<=float(r.entry): sl=float(r.sl)
        d["sl"]=sl;d["risk"]=sl-float(r.entry)
        if d["risk"]<=0:continue
        tgt=significant_target(pd.Series(d),z,min_prom)
        if tgt is None:continue
        d.update(tgt);rows.append(d)
    aa=pd.DataFrame(rows)
    if len(aa):aa=aa[[c for c in base.columns if c in aa.columns]]
    else:aa=base.iloc[:0].copy()
    return pd.concat([other,aa],ignore_index=True).sort_values("entry_ns").reset_index(drop=True),fd

def gap(m):
    n=int(m.get("N") or 0);wr=float(m.get("WR_pct") or 0);pf=float(m.get("PF") or 0)
    gn=max(0,TARGET_N-n)/TARGET_N;gw=max(0,TARGET_WR-wr)/TARGET_WR;gp=max(0,TARGET_PF-pf)/TARGET_PF
    return {"gap_N":gn,"gap_WR":gw,"gap_PF":gp,"inverse_score":3*gp+2*gw+gn,
            "passes":bool(n>=TARGET_N and wr>=TARGET_WR and pf>=TARGET_PF)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--lane",choices=["real","random"],required=True)
    ap.add_argument("--catalog",default="catalog/raw_bidask");ap.add_argument("--out",required=True)
    ap.add_argument("--seed",type=int,default=20261005);ap.add_argument("--vol-mult",type=float,default=1.0)
    ap.add_argument("--days",type=int,default=10);ap.add_argument("--ticks-per-minute",type=int,default=20)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.lane=="real":
        raw=v127.v126.load_raw(a.catalog);meta={"lane":"real_raw","authoritative":True,"raw_ticks":len(raw)}
    else:
        raw=v129.make_random_quotes(a.seed,a.vol_mult,a.days,a.ticks_per_minute)
        meta={"lane":"random_stress","authoritative":False,"seed":a.seed,"vol_mult":a.vol_mult,"days":a.days,"synthetic_quotes":len(raw)}
    z=v16.v14.feature(v16.v14.bars(raw,1));ctx,counts=v125.v22.owned_mssplus_context(z);base,stats,audit=v125.path_extreme_entries(ctx,z,raw)
    res={}
    for frac in [0.55,0.65,0.75,0.85,1.00]:
      for prom in [0.00,0.15,0.30,0.50]:
        name=f"Q{int(frac*100):02d}_TPROM{int(prom*100):02d}"
        ent,fd=build(base,z,frac,prom,0.10)
        tr,mp=v128.pack(ent,raw);m=mp["CLASSIC_A_SELL"]
        res[name]={"retain_frac":frac,"target_prominence_atr":prom,"selected_A":int((ent.pattern=="CLASSIC_A_SELL").sum()),**mp,**gap(m)}
    rank=sorted([{"name":k,**v} for k,v in res.items()],key=lambda x:(x["inverse_score"],-(x["CLASSIC_A_SELL"].get("PF") or 0),-(x["CLASSIC_A_SELL"].get("N") or 0)))
    result={"version":"v1.30","architecture":"TARGET_COUNT_INVERSION_SHARED_CORE",
      "lane_meta":meta,"target_edge":{"N_min":120,"WR_min_pct":45.0,"PF_min":1.20},
      "constraints":["M1 only","Classic V unchanged","no M5/M15/M30/G75","no fixed RR","no fixed multiple TP","no HTF target"],
      "logic":"Rank all Classic A by continuous similarity to the high-WR A kernel, retain 55-100%, then use nearest causal R1 Close pivot with structural prominence floor.",
      "m1_bars":len(z),"mssplus_counts":counts,"path_stats":dict(stats),"top_inverse":rank[:12],"candidates":res}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8");print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
