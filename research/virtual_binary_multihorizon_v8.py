from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from sklearn.ensemble import HistGradientBoostingClassifier
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from virtual_binary_structure_pa_v2 import feature_frame
from virtual_binary_direct_udp_v5 import make_features, trade_metric
from virtual_binary_tickmicro_v6 import tick_micro_features, fit_cls, infer_probs
from virtual_binary_metagate_v7 import make_meta_features, score_mode

HORIZONS=[30,45,60,75,90,120]

def select_rule(val_true,val_pred,val_pc):
    rows=[]
    lows=[0.20,0.25,0.30,0.35,0.40,0.45]
    highs=[0.55,0.60,0.65,0.70,0.75,0.80]
    for hi in highs:
        use=val_pc>=hi
        row=score_mode(f"TRUST_hi{hi:.2f}",val_true,val_pred,use)
        row.update({"mode":"trust","low":None,"high":hi}); rows.append(row)
    for lo in lows:
        use=val_pc<=lo
        row=score_mode(f"FLIP_lo{lo:.2f}",val_true,-val_pred,use)
        row.update({"mode":"flip","low":lo,"high":None}); rows.append(row)
    for lo in lows:
        for hi in highs:
            use=(val_pc<=lo)|(val_pc>=hi)
            pred=np.where(val_pc<=lo,-val_pred,val_pred)
            row=score_mode(f"HYBRID_lo{lo:.2f}_hi{hi:.2f}",val_true,pred,use)
            row.update({"mode":"hybrid","low":lo,"high":hi}); rows.append(row)
    eligible=[r for r in rows if r["signals"]>=300]
    best=max(eligible,key=lambda r:(r["direction_accuracy"],r["coverage_pct"])) if eligible else max(rows,key=lambda r:r["direction_accuracy"])
    return best,rows

def apply_rule(pred,pc,best):
    if best["mode"]=="trust":
        use=pc>=best["high"]; final=pred.copy()
    elif best["mode"]=="flip":
        use=pc<=best["low"]; final=-pred
    else:
        use=(pc<=best["low"])|(pc>=best["high"])
        final=np.where(pc<=best["low"],-pred,pred)
    return final,use

def fit_meta(meta_train,base,base_cols):
    pu,pd=infer_probs(base,meta_train,base_cols)
    pred=np.where(pu>=pd,1,-1)
    y=(pred==meta_train.raw_dir.values).astype(int)
    X=make_meta_features(meta_train,pu,pd)
    m=HistGradientBoostingClassifier(
        max_iter=180,learning_rate=0.04,max_leaf_nodes=15,
        min_samples_leaf=160,l2_regularization=2.0,random_state=310421
    )
    m.fit(X,y)
    return m

def eval_meta(fr,base,meta,base_cols):
    pu,pd=infer_probs(base,fr,base_cols)
    pred=np.where(pu>=pd,1,-1)
    X=make_meta_features(fr,pu,pd)
    pc=meta.predict_proba(X)[:, list(meta.classes_).index(1)]
    return pred,pc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_multihorizon_v8")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    a=ap.parse_args()

    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])

    q=feature_frame(ticks)
    q=q.join(make_features(q).add_prefix("ctx_")).join(tick_micro_features(ticks))
    ctx_cols=[c for c in q.columns if c.startswith("ctx_")]
    tm_cols=[c for c in q.columns if c.startswith("tm_")]
    base_cols=ctx_cols+tm_cols

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; all_grids={}
    slip=2*a.slippage_side

    for h in HORIZONS:
        steps=h//5
        x=q.copy()
        x["future_bid_h"]=x.bid.shift(-steps)
        x["future_ask_h"]=x.ask.shift(-steps)
        x["future_ts_h"]=x.index.to_series().shift(-steps)
        x["future_mid_h"]=(x.future_bid_h+x.future_ask_h)/2
        x["raw_delta_h"]=x.future_mid_h-x.mid
        x["raw_dir"]=np.where(x.raw_delta_h>0,1,np.where(x.raw_delta_h<0,-1,0))
        x["long_pnl"]=x.future_bid_h-x.ask-slip
        x["short_pnl"]=x.bid-x.future_ask_h-slip
        x["contiguous_h"]=(x.future_ts_h-x.index.to_series()).dt.total_seconds().eq(h)

        d=x.loc[x.contiguous_h,base_cols+["raw_dir","long_pnl","short_pnl"]].replace([np.inf,-np.inf],np.nan).dropna().copy()
        n=len(d); i1=int(n*.50); i2=int(n*.70); i3=int(n*.85)
        bt=d.iloc[:i1]; mt=d.iloc[i1:i2]; va=d.iloc[i2:i3]; oo=d.iloc[i3:]

        base=fit_cls(bt,base_cols,"raw_dir")
        meta=fit_meta(mt,base,base_cols)

        vpred,vpc=eval_meta(va,base,meta,base_cols)
        best,grid=select_rule(va.raw_dir.values,vpred,vpc)
        all_grids[str(h)]=grid

        opred,opc=eval_meta(oo,base,meta,base_cols)
        final,use=apply_rule(opred,opc,best)
        baseline_acc=float((opred==oo.raw_dir.values).mean())
        selected_acc=float((final[use]==oo.raw_dir.values[use]).mean()) if use.sum() else 0.0
        side=np.zeros(len(oo),int); side[use]=final[use]
        kpi=trade_metric(f"H{h}",side,oo.long_pnl.values,oo.short_pnl.values)
        kpi.update({
            "horizon_sec":h,
            "baseline_direction_accuracy":baseline_acc,
            "selected_direction_accuracy":selected_acc,
            "coverage_pct":float(100*use.sum()/len(oo)),
            "selected_signals":int(use.sum()),
            "selected_mode":best["mode"],
            "selected_low":best.get("low"),
            "selected_high":best.get("high"),
            "usable_samples":n,
            "oos_samples":len(oo)
        })
        all_rows.append(kpi)

        pd.DataFrame({
            "ts":oo.index.astype(str),"raw_true":oo.raw_dir.values,
            "base_pred":opred,"p_correct":opc,"selected":use.astype(int),
            "final_pred":final,"long_pnl":oo.long_pnl.values,"short_pnl":oo.short_pnl.values
        }).to_csv(out/f"oos_predictions_H{h}.csv",index=False)
        pd.DataFrame(grid).to_csv(out/f"validation_metagate_H{h}.csv",index=False)

    pd.DataFrame(all_rows).sort_values("horizon_sec").to_csv(out/"multihorizon_kpi.csv",index=False)
    best_acc=max(all_rows,key=lambda r:r["selected_direction_accuracy"])
    best_pf=max(all_rows,key=lambda r:r["PF"])
    result={
        "experiment":"AMOS_VirtualBinary_MultiHorizon_v8",
        "horizons_sec":HORIZONS,
        "raw_ticks":len(ticks),
        "fixed_requirements":{"decision_clock":"every 5 seconds","positions":"independent","overlap":"allowed"},
        "oos":all_rows,
        "best_by_direction_accuracy":best_acc,
        "best_by_pf":best_pf,
        "design":"same context+tick-micro base model and per-horizon MetaGate; each horizon independently trained and validated",
        "dataset_manifest":man,
        "nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
