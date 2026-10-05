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

def make_meta_features(frame, pu, pdn):
    z=pd.DataFrame(index=frame.index)
    z["p_up"]=pu
    z["p_down"]=pdn
    z["confidence"]=np.maximum(pu,pdn)
    z["margin"]=np.abs(pu-pdn)
    # compact regime/context set only
    wanted=[
        "ctx_m1_loc","ctx_m5_loc","ctx_m15_loc","ctx_h1_loc",
        "ctx_m1_eff","ctx_m5_eff","ctx_m15_eff","ctx_h1_eff",
        "ctx_phase_htf_trend","ctx_phase_htf_loc",
        "ctx_spread_ratio","ctx_vol30","ctx_vol60","ctx_trend_eff30",
        "tm_tick_imbalance","tm_path_eff","tm_tick_count","tm_spread_mean",
        "tm_signed_move","tm_last_accel"
    ]
    for c in wanted:
        if c in frame.columns:
            z[c]=frame[c]
    h=frame.index.hour + frame.index.minute/60.0
    z["hour_sin"]=np.sin(2*np.pi*h/24)
    z["hour_cos"]=np.cos(2*np.pi*h/24)
    return z.replace([np.inf,-np.inf],np.nan)

def score_mode(name, true_dir, pred_dir, use_mask):
    n=int(use_mask.sum())
    if n==0:
        return {"name":name,"signals":0,"coverage_pct":0.0,"direction_accuracy":0.0}
    acc=float((pred_dir[use_mask]==true_dir[use_mask]).mean())
    return {"name":name,"signals":n,"coverage_pct":float(100*n/len(true_dir)),"direction_accuracy":acc}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_metagate_v7")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    a=ap.parse_args()

    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])

    q=feature_frame(ticks)
    q=q.join(make_features(q).add_prefix("ctx_")).join(tick_micro_features(ticks))
    slip=2*a.slippage_side
    q["future_mid"]=(q.future_bid+q.future_ask)/2
    q["raw_delta60"]=q.future_mid-q.mid
    q["raw_dir"]=np.where(q.raw_delta60>0,1,np.where(q.raw_delta60<0,-1,0))
    q["long_pnl"]=q.future_bid-q.ask-slip
    q["short_pnl"]=q.bid-q.future_ask-slip

    ctx_cols=[c for c in q.columns if c.startswith("ctx_")]
    tm_cols=[c for c in q.columns if c.startswith("tm_")]
    base_cols=ctx_cols+tm_cols

    d=q.loc[q.contiguous60,base_cols+["raw_dir","long_pnl","short_pnl"]].replace([np.inf,-np.inf],np.nan).dropna().copy()
    n=len(d)
    i1=int(n*.50); i2=int(n*.70); i3=int(n*.85)
    base_train=d.iloc[:i1]
    meta_train=d.iloc[i1:i2]
    val=d.iloc[i2:i3]
    oos=d.iloc[i3:]

    base=fit_cls(base_train,base_cols,"raw_dir")

    # out-of-sample predictions for meta training
    mtu,mtd=infer_probs(base,meta_train,base_cols)
    mt_pred=np.where(mtu>=mtd,1,-1)
    mt_true=meta_train.raw_dir.values
    mt_y=(mt_pred==mt_true).astype(int)
    mtX=make_meta_features(meta_train,mtu,mtd)

    meta=HistGradientBoostingClassifier(
        max_iter=180,learning_rate=0.04,max_leaf_nodes=15,
        min_samples_leaf=160,l2_regularization=2.0,random_state=310421
    )
    meta.fit(mtX,mt_y)

    def eval_frame(fr):
        pu,pdn=infer_probs(base,fr,base_cols)
        pred=np.where(pu>=pdn,1,-1)
        X=make_meta_features(fr,pu,pdn)
        pc=meta.predict_proba(X)[:, list(meta.classes_).index(1)]
        return pred,pc

    vpred,vpc=eval_frame(val)
    vtrue=val.raw_dir.values
    candidates=[]
    lows=[0.20,0.25,0.30,0.35,0.40,0.45]
    highs=[0.55,0.60,0.65,0.70,0.75,0.80]
    for hi in highs:
        mask=vpc>=hi
        row=score_mode(f"TRUST_hi{hi:.2f}",vtrue,vpred,mask)
        row.update({"mode":"trust","low":None,"high":hi})
        candidates.append(row)
    for lo in lows:
        mask=vpc<=lo
        row=score_mode(f"FLIP_lo{lo:.2f}",vtrue,-vpred,mask)
        row.update({"mode":"flip","low":lo,"high":None})
        candidates.append(row)
    for lo in lows:
        for hi in highs:
            use=(vpc<=lo)|(vpc>=hi)
            hybrid=np.where(vpc<=lo,-vpred,vpred)
            row=score_mode(f"HYBRID_lo{lo:.2f}_hi{hi:.2f}",vtrue,hybrid,use)
            row.update({"mode":"hybrid","low":lo,"high":hi})
            candidates.append(row)

    eligible=[r for r in candidates if r["signals"]>=500]
    best=max(eligible,key=lambda r:(r["direction_accuracy"],r["coverage_pct"])) if eligible else max(candidates,key=lambda r:r["direction_accuracy"])

    opred,opc=eval_frame(oos)
    otrue=oos.raw_dir.values
    if best["mode"]=="trust":
        use=opc>=best["high"]; final=opred.copy()
    elif best["mode"]=="flip":
        use=opc<=best["low"]; final=-opred
    else:
        use=(opc<=best["low"])|(opc>=best["high"])
        final=np.where(opc<=best["low"],-opred,opred)

    baseline_acc=float((opred==otrue).mean())
    meta_acc=float((final[use]==otrue[use]).mean()) if use.sum() else 0.0

    # executable trade diagnostics on selected signals
    side=np.zeros(len(oos),int); side[use]=final[use]
    kpi=trade_metric("METAGATE_SELECTED",side,oos.long_pnl.values,oos.short_pnl.values)
    kpi["direction_accuracy"]=meta_acc
    kpi["coverage_pct"]=float(100*use.sum()/len(oos))

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(candidates).to_csv(out/"validation_metagate_grid.csv",index=False)
    pd.DataFrame([{
        "baseline_direction_accuracy":baseline_acc,
        "metagate_direction_accuracy":meta_acc,
        "signals":int(use.sum()),
        "coverage_pct":float(100*use.sum()/len(oos)),
        **{f"selected_{k}":v for k,v in best.items() if k not in ("name",)}
    }]).to_csv(out/"accuracy_summary.csv",index=False)
    pd.DataFrame([kpi]).to_csv(out/"oos_kpi.csv",index=False)
    pd.DataFrame({
        "ts":oos.index.astype(str),"raw_true":otrue,"base_pred":opred,
        "p_correct":opc,"selected":use.astype(int),"final_pred":final,
        "long_pnl":oos.long_pnl.values,"short_pnl":oos.short_pnl.values
    }).to_csv(out/"oos_predictions.csv",index=False)

    result={
        "experiment":"AMOS_VirtualBinary_MetaGate_v7",
        "raw_ticks":len(ticks),"usable_samples":n,
        "split":{"base_train":len(base_train),"meta_train":len(meta_train),"validation":len(val),"oos":len(oos),"method":"chronological 50/20/15/15"},
        "baseline_oos_direction_accuracy":baseline_acc,
        "selected_rule":best,
        "metagate_oos_direction_accuracy":meta_acc,
        "metagate_oos_signals":int(use.sum()),
        "metagate_oos_coverage_pct":float(100*use.sum()/len(oos)),
        "oos_trade_kpi":kpi,
        "design":"base direction model + out-of-sample correctness meta-model; trust high p(correct), flip low p(correct), pass uncertain",
        "dataset_manifest":man,
        "nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
