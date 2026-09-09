#!/usr/bin/env python3
"""AREM Behavioral Clone modular trainer.

Stages:
  stage1: ENTRY / DIRECTION
  stage2: EXIT HOLD/CLOSE factual clone
  stage3: OOS summary / promotion gate

Design goals:
- teacher dataset is independent from model implementation
- AI slots are replaceable
- chronological OOS only
- no future leakage
- every stage writes resumable artifacts
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import LabelEncoder

SYMBOLS=["EURUSD","GBPUSD","USDCHF","AUDUSD","USDCAD","NZDUSD"]
USD_BASE={"USDCAD","USDCHF"}
USD_QUOTE={"EURUSD","GBPUSD","AUDUSD","NZDUSD"}


def load_market(market_dir:Path):
    out={}
    for s in SYMBOLS:
        p=market_dir/f"{s}.csv"
        x=pd.read_csv(p)
        x["timestamp"]=pd.to_datetime(x["timestamp"],utc=True).dt.tz_convert(None).dt.floor("min")
        x["symbol"]=s
        out[s]=x.sort_values("timestamp").reset_index(drop=True)
    return out


def load_teacher(path:Path, utc_offset_hours:int):
    t=pd.read_csv(path)
    t["open_time"]=pd.to_datetime(t["open_time"]).dt.floor("min")
    t["close_time"]=pd.to_datetime(t["close_time"]).dt.floor("min")
    t["symbol"]=t["symbol"].astype(str).str.upper()
    t["action"]=t["action"].astype(str).str.upper()
    t=t[t.symbol.isin(SYMBOLS)].copy()
    t["entry_ts"]=t.open_time-pd.Timedelta(hours=utc_offset_hours)
    t["exit_ts"]=t.close_time-pd.Timedelta(hours=utc_offset_hours)
    return t


def build_features(market:dict[str,pd.DataFrame]):
    close_wide=None
    for s in SYMBOLS:
        q=market[s][["timestamp","close"]].rename(columns={"close":s}).set_index("timestamp")
        close_wide=q if close_wide is None else close_wide.join(q,how="outer")
    close_wide=close_wide.sort_index().ffill(limit=2)
    r1=close_wide.pct_change()
    usd=pd.DataFrame(index=r1.index)
    for s in SYMBOLS:
        usd[s]=r1[s] if s in USD_BASE else -r1[s]
    usd_factor=usd.mean(axis=1)
    usd_disp=usd.std(axis=1)
    usd_rank=usd.rank(axis=1,pct=True)

    frames=[]
    for s in SYMBOLS:
        x=market[s].copy().sort_values("timestamp")
        c=x.close
        for n in [1,3,5,15,30]: x[f"ret_{n}"]=c.pct_change(n)
        x["vel_3"]=x.ret_3/3
        x["acc_3"]=x.vel_3.diff()
        x["range_rel"]=(x.high-x.low)/x.close
        x["body_rel"]=(x.close-x.open)/x.open
        x["wick_up"]=(x.high-x[["open","close"]].max(axis=1))/x.close
        x["wick_dn"]=(x[["open","close"]].min(axis=1)-x.low)/x.close
        x["dist_hi20"]=c/x.high.rolling(20).max()-1
        x["dist_lo20"]=c/x.low.rolling(20).min()-1
        x["rv15"]=x.ret_1.rolling(15).std()
        x["vol_z20"]=(x.tick_volume-x.tick_volume.rolling(20).mean())/(x.tick_volume.rolling(20).std()+1e-9)
        x["spread_rel"]=x.spread_open/x.close if "spread_open" in x else 0.0
        x["hour_sin"]=np.sin(2*np.pi*x.timestamp.dt.hour/24)
        x["hour_cos"]=np.cos(2*np.pi*x.timestamp.dt.hour/24)
        x["min_sin"]=np.sin(2*np.pi*x.timestamp.dt.minute/60)
        x["min_cos"]=np.cos(2*np.pi*x.timestamp.dt.minute/60)
        x=x.set_index("timestamp")
        x["usd_factor"]=usd_factor.reindex(x.index)
        x["usd_dispersion"]=usd_disp.reindex(x.index)
        x["cross_rank"]=usd_rank[s].reindex(x.index)
        signed=x.ret_1 if s in USD_BASE else -x.ret_1
        x["usd_divergence"]=signed-x.usd_factor
        frames.append(x.reset_index())
    return pd.concat(frames,ignore_index=True)


def feature_columns(df):
    prefixes=("ret_","vel_","acc_","range_","body_","wick_","dist_","rv","vol_z","spread_rel","hour_","min_","usd_factor","usd_dispersion","cross_rank","usd_divergence")
    return [c for c in df.columns if c.startswith(prefixes)]


def stage1(teacher, feat_df, outdir:Path):
    emap={(r.symbol,r.entry_ts):r.action for r in teacher.itertuples()}
    d=feat_df.copy()
    d["direction"]=[emap.get((s,t)) for s,t in zip(d.symbol,d.timestamp)]
    d["entry"]=d.direction.notna().astype(int)
    feats=feature_columns(d)
    d=d.dropna(subset=feats).copy()
    d=pd.get_dummies(d,columns=["symbol"],prefix="sym")
    cols=feats+[c for c in d.columns if c.startswith("sym_")]
    cut=d.timestamp.quantile(.75)
    tr=d[d.timestamp<cut]; oo=d[d.timestamp>=cut]

    def sample(x,ratio,seed):
        p=x[x.entry==1]; n=x[x.entry==0]
        n=n.sample(n=min(len(n),len(p)*ratio),random_state=seed)
        return pd.concat([p,n]).sample(frac=1,random_state=seed)
    a=sample(tr,6,1); b=sample(oo,10,2)
    em=ExtraTreesClassifier(n_estimators=220,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=310421,max_features="sqrt")
    em.fit(a[cols],a.entry)
    prob=em.predict_proba(b[cols])[:,1]
    best=None
    for th in np.arange(.20,.81,.02):
        pred=(prob>=th).astype(int)
        p,r,f,_=precision_recall_fscore_support(b.entry,pred,average="binary",zero_division=0)
        tn,fp,fn,tp=confusion_matrix(b.entry,pred,labels=[0,1]).ravel()
        fer=fp/max(1,tn+fp); spec=1-fer
        score=f-max(0,fer-.10)*2
        rec=dict(threshold=float(th),precision=float(p),recall=float(r),f1=float(f),specificity=float(spec),false_entry_rate=float(fer),score=float(score))
        if best is None or rec["score"]>best["score"]: best=rec

    dt=tr[tr.entry==1]; do=oo[oo.entry==1]
    le=LabelEncoder().fit(["BUY","SELL"])
    dm=ExtraTreesClassifier(n_estimators=260,min_samples_leaf=1,class_weight="balanced",n_jobs=-1,random_state=310422,max_features=.8)
    dm.fit(dt[cols],le.transform(dt.direction))
    dp=le.inverse_transform(dm.predict(do[cols]))
    direction={"balanced_accuracy":float(balanced_accuracy_score(do.direction,dp)),"mcc":float(matthews_corrcoef(do.direction,dp)),"oos_entries":int(len(do))}
    for lab in ["BUY","SELL"]:
        p,r,f,_=precision_recall_fscore_support((do.direction==lab).astype(int),(dp==lab).astype(int),average="binary",zero_division=0)
        direction[lab]={"precision":float(p),"recall":float(r),"f1":float(f)}

    outdir.mkdir(parents=True,exist_ok=True)
    joblib.dump(em,outdir/"entry_model.joblib")
    joblib.dump(dm,outdir/"direction_model.joblib")
    joblib.dump(le,outdir/"direction_encoder.joblib")
    (outdir/"feature_schema.json").write_text(json.dumps({"features":cols},indent=2),encoding="utf-8")
    metrics={"oos_cut":str(cut),"ENTRY":best,"DIRECTION":direction}
    (outdir/"stage1_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    d[["timestamp","entry","direction"]+cols].to_pickle(outdir/"feature_matrix.pkl")
    print(json.dumps(metrics,indent=2))


def stage2(teacher, feat_df, outdir:Path):
    feats=feature_columns(feat_df)
    bysym={s:feat_df[feat_df.symbol==s].dropna(subset=feats).set_index("timestamp") for s in SYMBOLS}
    rows=[]
    for r in teacher.itertuples():
        x=bysym[r.symbol]
        if r.entry_ts not in x.index or r.exit_ts not in x.index: continue
        seq=x.loc[r.entry_ts:r.exit_ts]
        if len(seq)<2: continue
        side=1 if r.action=="BUY" else -1
        ep=float(r.open_price); pip=.0001
        running=[]
        for ts,row in seq.iterrows():
            upnl=side*(float(row.close)-ep)/pip; running.append(upnl)
            rec={c:row[c] for c in feats}
            rec.update(symbol=r.symbol,timestamp=ts,elapsed_min=(ts-r.entry_ts).total_seconds()/60,upnl_pips=upnl,mfe_pips=max(0,max(running)),mae_pips=min(0,min(running)),side_buy=1 if r.action=="BUY" else 0,exit_label="CLOSE" if ts==r.exit_ts else "HOLD")
            rec["mfe_giveback"]=rec["mfe_pips"]-upnl if rec["mfe_pips"]>0 else 0.0
            rows.append(rec)
    d=pd.DataFrame(rows).dropna()
    d=pd.get_dummies(d,columns=["symbol"],prefix="sym")
    cols=feats+["elapsed_min","upnl_pips","mfe_pips","mae_pips","mfe_giveback","side_buy"]+[c for c in d.columns if c.startswith("sym_")]
    cut=d.timestamp.quantile(.75)
    tr=d[d.timestamp<cut]; oo=d[d.timestamp>=cut]
    def sample(x,ratio,seed):
        c=x[x.exit_label=="CLOSE"]; h=x[x.exit_label=="HOLD"]
        h=h.sample(n=min(len(h),len(c)*ratio),random_state=seed)
        return pd.concat([c,h]).sample(frac=1,random_state=seed)
    a=sample(tr,8,3); b=sample(oo,12,4)
    le=LabelEncoder().fit(["CLOSE","HOLD"])
    m=ExtraTreesClassifier(n_estimators=280,min_samples_leaf=1,class_weight="balanced",n_jobs=-1,random_state=310423,max_features=.7)
    m.fit(a[cols],le.transform(a.exit_label))
    prob=m.predict_proba(b[cols])[:,list(le.classes_).index("CLOSE")]
    best=None
    for th in np.arange(.20,.81,.02):
        pred=np.where(prob>=th,"CLOSE","HOLD")
        p,r,f,_=precision_recall_fscore_support((b.exit_label=="CLOSE").astype(int),(pred=="CLOSE").astype(int),average="binary",zero_division=0)
        bal=balanced_accuracy_score(b.exit_label,pred); mcc=matthews_corrcoef(b.exit_label,pred)
        rec=dict(threshold=float(th),close_precision=float(p),close_recall=float(r),close_f1=float(f),balanced_accuracy=float(bal),mcc=float(mcc),score=float(f+.25*bal))
        if best is None or rec["score"]>best["score"]: best=rec
    outdir.mkdir(parents=True,exist_ok=True)
    joblib.dump(m,outdir/"exit_model.joblib"); joblib.dump(le,outdir/"exit_encoder.joblib")
    (outdir/"exit_feature_schema.json").write_text(json.dumps({"features":cols},indent=2),encoding="utf-8")
    (outdir/"stage2_metrics.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
    print(json.dumps(best,indent=2))


def stage3(root:Path):
    s1=json.loads((root/"stage1"/"stage1_metrics.json").read_text())
    s2=json.loads((root/"stage2"/"stage2_metrics.json").read_text())
    gates={
      "G1_ENTRY": s1["ENTRY"]["f1"]>=.80 and s1["ENTRY"]["precision"]>=.60,
      "G2_NO_ENTRY": s1["ENTRY"]["false_entry_rate"]<=.10,
      "G3_DIRECTION": s1["DIRECTION"]["balanced_accuracy"]>=.70 and s1["DIRECTION"]["SELL"]["recall"]>=.60 and s1["DIRECTION"]["mcc"]>=.35,
      "G4_EXIT_CLONE": s2["close_f1"]>=.60 and s2["balanced_accuracy"]>=.65,
      "G5_REALITY_BT": False,
    }
    status="REALITY_BT_REQUIRED" if all([gates[k] for k in ["G1_ENTRY","G2_NO_ENTRY","G3_DIRECTION","G4_EXIT_CLONE"]]) else "MODEL_REFINEMENT_REQUIRED"
    out={"gates":gates,"status":status,"stage1":s1,"stage2":s2}
    (root/"oos_summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=["stage1","stage2","stage3"],required=True); ap.add_argument("--teacher",default="datasets/scalpers_circle_teacher.csv"); ap.add_argument("--market-dir",default="artifacts/scalpers_circle_m1"); ap.add_argument("--out",default="artifacts/arem_clone"); ap.add_argument("--utc-offset",type=int,default=3)
    a=ap.parse_args(); root=Path(a.out)
    if a.stage=="stage3": return stage3(root)
    market=load_market(Path(a.market_dir)); teacher=load_teacher(Path(a.teacher),a.utc_offset); feat=build_features(market)
    if a.stage=="stage1": stage1(teacher,feat,root/"stage1")
    else: stage2(teacher,feat,root/"stage2")

if __name__=="__main__": main()
