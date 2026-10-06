#!/usr/bin/env python3
"""
AMOS Checklist Learning v1.5
Fixed gates are never bypassed. ML only ranks setups AFTER:
Sweep -> MSS -> Volume -> Equilibrium -> Pullback.

Strict chronological OOS: first 60% train, last 40% test.
Threshold is learned from TRAIN scores only (60th percentile).
"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

FEATURES=[
    "rr","acc_eff","acc_range_atr","bars_from_sweep",
    "hour_sin","hour_cos","m15_body_atr","m15_range_atr",
    "m15_close_pos","m15_efficiency"
]

def atr(df,n=14):
    p=df.close.shift(1)
    tr=pd.concat([(df.high-df.low).abs(),(df.high-p).abs(),(df.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def pf(r):
    r=np.asarray(r,float); gp=r[r>0].sum(); gl=-r[r<0].sum()
    return float(gp/gl) if gl>0 else (float("inf") if gp>0 else 0.0)

def kpi(x):
    if len(x)==0:return {"N":0,"WR_pct":0.0,"PF_R":0.0,"sum_R":0.0}
    return {"N":int(len(x)),"WR_pct":float(100*(x.R>0).mean()),"PF_R":pf(x.R),"sum_R":float(x.R.sum())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--base-out",default="results/amos-checklist-learning-v1-5/base")
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    base=Path(a.base_out);base.mkdir(parents=True,exist_ok=True)

    subprocess.run([sys.executable,"research/amos_checklist_parity_v1_3/ordered_pattern_bt.py",
                    "--data",a.data,"--out",str(base)],check=True)
    td=pd.read_csv(base/"trades.csv")
    td=td[td.pattern=="P5_SWEEP_MSS_VOL_EQ"].copy()
    td["entry_time"]=pd.to_datetime(td.entry_time)
    td=td.sort_values("entry_time").reset_index(drop=True)

    d=pd.read_csv(a.data);d.columns=[c.lower() for c in d.columns]
    d.datetime=pd.to_datetime(d.datetime)
    for c in ["open","high","low","close"]:d[c]=pd.to_numeric(d[c],errors="coerce")
    m=d.set_index("datetime").resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    m["atr"]=atr(m)
    m["body_atr"]=(m.close-m.open).abs()/m.atr
    m["range_atr"]=(m.high-m.low)/m.atr
    m["close_pos"]=(m.close-m.low)/(m.high-m.low).replace(0,np.nan)
    m["efficiency"]=(m.close-m.open).abs()/(m.high-m.low).replace(0,np.nan)

    ix=m.index.searchsorted(td.entry_time.values,side="right")-1
    valid=ix>=0;td=td.loc[valid].reset_index(drop=True);ix=ix[valid]
    td["m15_body_atr"]=m.body_atr.iloc[ix].to_numpy()
    td["m15_range_atr"]=m.range_atr.iloc[ix].to_numpy()
    td["m15_close_pos"]=m.close_pos.iloc[ix].to_numpy()
    td["m15_efficiency"]=m.efficiency.iloc[ix].to_numpy()
    h=td.entry_time.dt.hour+td.entry_time.dt.minute/60
    td["hour_sin"]=np.sin(2*np.pi*h/24);td["hour_cos"]=np.cos(2*np.pi*h/24)
    td["y"]=(td.R>0).astype(int)

    cut=max(20,int(len(td)*0.60))
    train=td.iloc[:cut].copy();test=td.iloc[cut:].copy()
    if train.y.nunique()<2 or len(test)<5:
        raise SystemExit(f"insufficient chronological sample train={len(train)} test={len(test)} classes={train.y.nunique()}")

    model=Pipeline([
      ("imp",SimpleImputer(strategy="median")),
      ("scale",StandardScaler()),
      ("clf",LogisticRegression(C=0.5,class_weight="balanced",max_iter=2000,random_state=42))
    ])
    model.fit(train[FEATURES],train.y)
    train["score"]=model.predict_proba(train[FEATURES])[:,1]
    test["score"]=model.predict_proba(test[FEATURES])[:,1]

    # Threshold is frozen using TRAIN only; no OOS tuning.
    threshold=float(train.score.quantile(0.60))
    train["selected"]=train.score>=threshold
    test["selected"]=test.score>=threshold

    clf=model.named_steps["clf"];coef=dict(zip(FEATURES,map(float,clf.coef_[0])))
    result={
      "architecture":"M15 context features -> fixed P5 ordered gates -> ML rank only -> entry",
      "fixed_gate_order":["Liquidity Sweep","MSS","Volume Influx","Equilibrium","Pullback"],
      "ml_permission":"RANK_ONLY_AFTER_ALL_FIXED_GATES_PASS",
      "leakage_control":"chronological 60/40 split; threshold from train only; OOS untouched",
      "features":FEATURES,"total_p5":len(td),"train_n":len(train),"oos_n":len(test),
      "threshold_train_q60":threshold,
      "train_all":kpi(train),"train_selected":kpi(train[train.selected]),
      "oos_all":kpi(test),"oos_selected":kpi(test[test.selected]),
      "oos_rejected":kpi(test[~test.selected]),
      "coefficients_standardized":coef
    }
    td.to_csv(out/"learning_dataset.csv",index=False)
    train.to_csv(out/"train_scored.csv",index=False);test.to_csv(out/"oos_scored.csv",index=False)
    (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
