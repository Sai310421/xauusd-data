from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from virtual_binary_structure_pa_v2 import feature_frame, fpx
from virtual_binary_direct_udp_v5 import make_features, pf_from, max_dd_pct, trade_metric

def tick_micro_features(ticks):
    rows=[]
    for t in ticks:
        ts=pd.Timestamp(int(t.ts_event),unit="ns",tz="UTC")
        bid=fpx(t.bid_price); ask=fpx(t.ask_price); mid=(bid+ask)/2.0
        rows.append((ts,bid,ask,mid,ask-bid))
    d=pd.DataFrame(rows,columns=["ts","bid","ask","mid","spread"]).sort_values("ts").set_index("ts")
    d["dmid"]=d.mid.diff()
    d["abs_dmid"]=d.dmid.abs()
    d["up"]=(d.dmid>0).astype(float)
    d["dn"]=(d.dmid<0).astype(float)
    d["flat"]=(d.dmid==0).astype(float)
    d["spread_chg"]=d.spread.diff()
    d["dmid2"]=d.dmid.diff()

    g=d.resample("5s",label="right",closed="right")
    m=pd.DataFrame(index=g.size().index)
    m["tm_tick_count"]=g.mid.count()
    m["tm_up_count"]=g.up.sum()
    m["tm_dn_count"]=g.dn.sum()
    m["tm_flat_count"]=g.flat.sum()
    denom=(m.tm_up_count+m.tm_dn_count).replace(0,np.nan)
    m["tm_tick_imbalance"]=(m.tm_up_count-m.tm_dn_count)/denom
    m["tm_mid_first"]=g.mid.first()
    m["tm_mid_last"]=g.mid.last()
    m["tm_mid_range"]=g.mid.max()-g.mid.min()
    m["tm_mid_std"]=g.mid.std()
    m["tm_signed_move"]=m.tm_mid_last-m.tm_mid_first
    m["tm_abs_path"]=g.abs_dmid.sum()
    m["tm_path_eff"]=m.tm_signed_move.abs()/m.tm_abs_path.replace(0,np.nan)
    m["tm_last_dmid"]=g.dmid.last()
    m["tm_mean_dmid"]=g.dmid.mean()
    m["tm_std_dmid"]=g.dmid.std()
    m["tm_last_accel"]=g.dmid2.last()
    m["tm_mean_accel"]=g.dmid2.mean()
    m["tm_spread_mean"]=g.spread.mean()
    m["tm_spread_std"]=g.spread.std()
    m["tm_spread_min"]=g.spread.min()
    m["tm_spread_max"]=g.spread.max()
    m["tm_spread_last"]=g.spread.last()
    m["tm_spread_slope"]=g.spread_chg.sum()
    # multi-bucket memory, still causal
    for c in ["tm_tick_imbalance","tm_signed_move","tm_path_eff","tm_last_dmid","tm_mean_dmid","tm_last_accel","tm_spread_mean","tm_spread_std","tm_tick_count"]:
        for k in [1,2,3,6,12]:
            m[f"{c}_lag{k}"]=m[c].shift(k)
    return m.replace([np.inf,-np.inf],np.nan)

def fit_cls(train,features,target):
    m=HistGradientBoostingClassifier(max_iter=260,learning_rate=0.04,max_leaf_nodes=31,
        min_samples_leaf=120,l2_regularization=1.2,random_state=310421)
    m.fit(train[features],train[target])
    return m

def infer_probs(model,frame,features):
    p=model.predict_proba(frame[features]); cls=list(model.classes_); idx={c:i for i,c in enumerate(cls)}
    pu=p[:,idx[1]] if 1 in idx else np.zeros(len(frame))
    pdn=p[:,idx[-1]] if -1 in idx else np.zeros(len(frame))
    return pu,pdn

def choose_threshold(val,pu,pd,long_pnl,short_pnl):
    rows=[]
    for th in [0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.66,0.70,0.74]:
        side=np.zeros(len(val),int)
        side[(pu>=th)&(pu>pd)]=1
        side[(pd>=th)&(pd>pu)]=-1
        met=trade_metric(f"thr_{th:.2f}",side,long_pnl,short_pnl)
        met["threshold"]=th; rows.append(met)
    elig=[r for r in rows if r["trades"]>=150]
    best=max(elig,key=lambda r:(r["PF"],r["net_usd_0p01lot_equiv"])) if elig else max(rows,key=lambda r:r["net_usd_0p01lot_equiv"])
    return best,rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_tickmicro_v6")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    a=ap.parse_args()

    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    q=feature_frame(ticks)
    ctx=make_features(q).add_prefix("ctx_")
    tm=tick_micro_features(ticks)
    q=q.join(ctx).join(tm)

    slip=2*a.slippage_side
    q["future_mid"]=(q.future_bid+q.future_ask)/2
    q["raw_delta60"]=q.future_mid-q.mid
    q["long_pnl"]=q.future_bid-q.ask-slip
    q["short_pnl"]=q.bid-q.future_ask-slip
    q["raw_dir"]=np.where(q.raw_delta60>0,1,np.where(q.raw_delta60<0,-1,0))

    ctx_cols=[c for c in q.columns if c.startswith("ctx_")]
    tm_cols=[c for c in q.columns if c.startswith("tm_")]
    all_cols=ctx_cols+tm_cols

    d=q.loc[q.contiguous60,all_cols+["raw_dir","long_pnl","short_pnl"]].replace([np.inf,-np.inf],np.nan).dropna().copy()
    n=len(d); i1=int(n*.60); i2=int(n*.80)
    tr,va,oo=d.iloc[:i1],d.iloc[i1:i2],d.iloc[i2:]

    stages=[("CONTEXT_ONLY",ctx_cols),("TICK_MICRO_ONLY",tm_cols),("CONTEXT_PLUS_TICK_MICRO",all_cols)]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; grids={}
    pred_out={"ts":oo.index.astype(str),"raw_true":oo.raw_dir.values,"long_pnl":oo.long_pnl.values,"short_pnl":oo.short_pnl.values}
    for name,features in stages:
        model=fit_cls(tr,features,"raw_dir")
        vpu,vpd=infer_probs(model,va,features)
        best,grid=choose_threshold(va,vpu,vpd,va.long_pnl.values,va.short_pnl.values)
        opu,opd=infer_probs(model,oo,features)
        side=np.zeros(len(oo),int)
        th=float(best["threshold"])
        side[(opu>=th)&(opu>opd)]=1
        side[(opd>=th)&(opd>opu)]=-1
        k=trade_metric(name,side,oo.long_pnl.values,oo.short_pnl.values)
        arg=np.where(opu>opd,1,-1)
        mask=oo.raw_dir.values!=0
        k["direction_accuracy"]=float((arg[mask]==oo.raw_dir.values[mask]).mean())
        k["threshold"]=th
        k["feature_count"]=len(features)
        rows.append(k); grids[name]=grid
        pred_out[name+"_pup"]=opu; pred_out[name+"_pdown"]=opd; pred_out[name+"_side"]=side

    pd.DataFrame(rows).to_csv(out/"oos_kpi.csv",index=False)
    pd.DataFrame(pred_out).to_csv(out/"oos_predictions.csv",index=False)
    for name,grid in grids.items():
        pd.DataFrame(grid).to_csv(out/f"validation_{name}.csv",index=False)

    result={
      "experiment":"AMOS_VirtualBinary_TickMicro_v6",
      "raw_ticks":len(ticks),"usable_samples":n,
      "split":{"train":len(tr),"validation":len(va),"oos":len(oo),"method":"chronological 60/20/20"},
      "oos":rows,
      "fixed_requirements":{"decision_clock":"every 5 seconds","positions":"independent","horizon":"exact +60s","overlap":"allowed"},
      "key_change":"add raw-tick microstructure inside each 5-second decision bucket",
      "tick_micro_features":["up/down quote imbalance","tick arrival count","signed move","path efficiency","quote acceleration","spread mean/std/range/slope","5-60s causal lags"],
      "dataset_manifest":man,"nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
