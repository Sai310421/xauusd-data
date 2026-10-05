from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from virtual_binary_structure_pa_v2 import feature_frame
from virtual_binary_phase_contrev_v4 import add_phase_features

def pf_from(pnl):
    pnl=np.asarray(pnl,float)
    gp=pnl[pnl>0].sum(); gl=-pnl[pnl<0].sum()
    return float(gp/gl) if gl>0 else (float("inf") if gp>0 else 0.0)

def max_dd_pct(pnl, initial=1000.0):
    pnl=np.asarray(pnl,float)
    if not len(pnl): return 0.0
    eq=initial+np.cumsum(pnl)
    peak=np.maximum.accumulate(np.r_[initial,eq])[1:]
    return float(np.max((peak-eq)/np.maximum(peak,1e-12)*100))

def trade_metric(name, side, long_pnl, short_pnl):
    side=np.asarray(side,int)
    pnl=np.where(side>0,long_pnl,np.where(side<0,short_pnl,0.0))
    t=pnl[side!=0]
    return {
        "name":name,"signals":int(len(side)),"trades":int((side!=0).sum()),
        "pass_rate_pct":float(100*(side==0).sum()/max(1,len(side))),
        "WR_pct":float(100*(t>0).sum()/max(1,len(t))),
        "PF":pf_from(t),"net_usd_0p01lot_equiv":float(t.sum()),
        "return_pct_on_1000":float(t.sum()/10.0),
        "avg_trade_usd":float(t.mean()) if len(t) else 0.0,
        "max_DD_pct":max_dd_pct(t)
    }

def make_features(q):
    phase=add_phase_features(q)
    x=pd.DataFrame(index=q.index)
    # Micro 5s/15s/30s price action.
    for c in ["r5","r10","r15","r30","r60","vel5","accel5","vol15","vol30","vol60",
              "spread","spread_ratio","ticks_5s","tick_mean30","trend_eff30",
              "pa_body_ratio","pa_upper_wick","pa_lower_wick","pa_close_loc",
              "pa_inside","pa_outside","pa_break_up","pa_break_dn"]:
        x[c]=q[c]

    # Explicit "where are we?" context, not raw OHLC levels.
    for p in ["m1","m5","m15","h1"]:
        x[f"{p}_loc"]=2*q[f"{p}_range_loc"].fillna(.5)-1
        x[f"{p}_dist_hi"]=q[f"{p}_dist_hi"]
        x[f"{p}_dist_lo"]=q[f"{p}_dist_lo"]
        x[f"{p}_eff"]=q[f"{p}_eff"]
        x[f"{p}_slope3"]=q[f"{p}_slope3"]
        x[f"{p}_bos"]=q[f"{p}_bos_up"]-q[f"{p}_bos_dn"]
        x[f"{p}_structure"]=(q[f"{p}_hh"]+q[f"{p}_hl"]-q[f"{p}_lh"]-q[f"{p}_ll"])/2

    # Phase/sweep/exhaustion features.
    keep=["micro_dir","micro_accel","wick_bias","close_strength","break_bias","body_strength",
          "sweep_high_30","sweep_low_30","sweep_high_60","sweep_low_60",
          "htf_trend","htf_loc","continuation_prior","reversal_prior",
          "m1_late_up","m1_late_dn","m5_late_up","m5_late_dn",
          "m15_late_up","m15_late_dn","h1_late_up","h1_late_dn",
          "m1_pullback_up","m1_pullback_dn","m5_pullback_up","m5_pullback_dn",
          "m15_pullback_up","m15_pullback_dn","h1_pullback_up","h1_pullback_dn"]
    for c in keep:
        if c in phase: x["phase_"+c]=phase[c]
    return x.replace([np.inf,-np.inf],np.nan)

def fit_cls(train,features,target):
    m=HistGradientBoostingClassifier(max_iter=240,learning_rate=0.045,max_leaf_nodes=31,
        min_samples_leaf=120,l2_regularization=1.0,random_state=310421)
    m.fit(train[features],train[target])
    return m

def fit_reg(train,features,target):
    m=HistGradientBoostingRegressor(max_iter=220,learning_rate=0.045,max_leaf_nodes=23,
        min_samples_leaf=120,l2_regularization=1.0,random_state=310421)
    m.fit(train[features],train[target])
    return m

def probs_for(model, frame, features):
    p=model.predict_proba(frame[features]); cls=list(model.classes_); idx={c:i for i,c in enumerate(cls)}
    pup=p[:,idx[1]] if 1 in idx else np.zeros(len(frame))
    pdn=p[:,idx[-1]] if -1 in idx else np.zeros(len(frame))
    ppass=p[:,idx[0]] if 0 in idx else np.zeros(len(frame))
    return pup,pdn,ppass

def choose_threshold(val,pup,pdn,pred_move,long_pnl,short_pnl,min_trades=150):
    rows=[]
    for th in [0.45,0.50,0.54,0.58,0.62,0.66,0.70,0.74,0.78]:
        for move_th in [0.00,0.10,0.20,0.30,0.40,0.60]:
            side=np.zeros(len(val),int)
            up=(pup>=th)&(pup>pdn)&(pred_move>=move_th)
            dn=(pdn>=th)&(pdn>pup)&(pred_move<=-move_th)
            side[up]=1; side[dn]=-1
            met=trade_metric(f"p{th:.2f}_m{move_th:.2f}",side,long_pnl,short_pnl)
            met["p_threshold"]=th; met["move_threshold"]=move_th; rows.append(met)
    elig=[r for r in rows if r["trades"]>=min_trades]
    best=max(elig,key=lambda r:(r["PF"],r["net_usd_0p01lot_equiv"])) if elig else max(rows,key=lambda r:r["net_usd_0p01lot_equiv"])
    return best,rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_direct_udp_v5")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    a=ap.parse_args()

    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    q=feature_frame(ticks)
    feats=make_features(q); q=q.join(feats.add_prefix("v5_"))
    feature_cols=[c for c in q.columns if c.startswith("v5_")]

    slip=2*a.slippage_side
    q["future_mid"]=(q.future_bid+q.future_ask)/2
    q["raw_delta60"]=q.future_mid-q.mid
    q["long_pnl"]=q.future_bid-q.ask-slip
    q["short_pnl"]=q.bid-q.future_ask-slip

    # Pure binary-style direction target: exact price relation at +60s.
    q["raw_dir"]=np.where(q.raw_delta60>0,1,np.where(q.raw_delta60<0,-1,0))
    # FX-executable three-state target.
    q["exec_dir"]=0
    q.loc[(q.long_pnl>0)&(q.long_pnl>=q.short_pnl),"exec_dir"]=1
    q.loc[(q.short_pnl>0)&(q.short_pnl>q.long_pnl),"exec_dir"]=-1

    cols=feature_cols+["raw_dir","exec_dir","raw_delta60","long_pnl","short_pnl"]
    d=q.loc[q.contiguous60,cols].replace([np.inf,-np.inf],np.nan).dropna().copy()
    n=len(d); i1=int(n*.60); i2=int(n*.80)
    train,val,oos=d.iloc[:i1],d.iloc[i1:i2],d.iloc[i2:]

    raw_model=fit_cls(train,feature_cols,"raw_dir")
    exec_model=fit_cls(train,feature_cols,"exec_dir")
    move_model=fit_reg(train,feature_cols,"raw_delta60")

    raw_pu,raw_pd,raw_pp=probs_for(raw_model,val,feature_cols)
    ex_pu,ex_pd,ex_pp=probs_for(exec_model,val,feature_cols)
    val_move=move_model.predict(val[feature_cols])

    raw_best,raw_grid=choose_threshold(val,raw_pu,raw_pd,val_move,val.long_pnl.values,val.short_pnl.values)
    ex_best,ex_grid=choose_threshold(val,ex_pu,ex_pd,val_move,val.long_pnl.values,val.short_pnl.values)

    def apply(frame,model,best):
        pu,pd,pp=probs_for(model,frame,feature_cols)
        mv=move_model.predict(frame[feature_cols])
        side=np.zeros(len(frame),int)
        up=(pu>=best["p_threshold"])&(pu>pd)&(mv>=best["move_threshold"])
        dn=(pd>=best["p_threshold"])&(pd>pu)&(mv<=-best["move_threshold"])
        side[up]=1; side[dn]=-1
        return side,pu,pd,pp,mv

    raw_side,rpu,rpd,rpp,rmv=apply(oos,raw_model,raw_best)
    ex_side,epu,epd,epp,emv=apply(oos,exec_model,ex_best)

    raw_pred=np.where(rpu>rpd,1,-1)
    dir_acc=float(accuracy_score(oos.raw_dir.values,raw_pred))
    raw_kpi=trade_metric("PURE_BINARY_DIRECTION_PLUS_MOVE_FILTER",raw_side,oos.long_pnl.values,oos.short_pnl.values)
    ex_kpi=trade_metric("EXECUTABLE_UP_DOWN_PASS_PLUS_MOVE",ex_side,oos.long_pnl.values,oos.short_pnl.values)

    # Direction-only diagnostic with no predicted-move gate, threshold fixed at validation choice.
    th=raw_best["p_threshold"]
    dside=np.where((rpu>=th)&(rpu>rpd),1,np.where((rpd>=th)&(rpd>rpu),-1,0))
    dir_only=trade_metric("PURE_BINARY_DIRECTION_ONLY",dside,oos.long_pnl.values,oos.short_pnl.values)

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(raw_grid).to_csv(out/"validation_raw_grid.csv",index=False)
    pd.DataFrame(ex_grid).to_csv(out/"validation_exec_grid.csv",index=False)
    pd.DataFrame([raw_kpi,ex_kpi,dir_only]).to_csv(out/"oos_kpi.csv",index=False)
    pd.DataFrame({
        "ts":oos.index.astype(str),"raw_true":oos.raw_dir.values,"exec_true":oos.exec_dir.values,
        "raw_pup":rpu,"raw_pdown":rpd,"raw_ppass":rpp,"exec_pup":epu,"exec_pdown":epd,"exec_ppass":epp,
        "pred_move60":rmv,"raw_side":raw_side,"exec_side":ex_side,
        "long_pnl":oos.long_pnl.values,"short_pnl":oos.short_pnl.values
    }).to_csv(out/"oos_predictions.csv",index=False)

    result={
      "experiment":"AMOS_VirtualBinary_Direct_UDP_v5",
      "raw_ticks":len(ticks),"usable_samples":n,
      "split":{"train":len(train),"validation":len(val),"oos":len(oos),"method":"chronological 60/20/20"},
      "pure_binary_direction_accuracy":dir_acc,
      "validation_selected":{"raw":raw_best,"exec":ex_best},
      "oos":[raw_kpi,ex_kpi,dir_only],
      "fixed_requirements":{
        "decision_clock":"every 5 seconds",
        "positions":"each decision/position independent",
        "horizon":"exact +60 seconds",
        "overlap":"allowed; BUY and SELL may coexist"
      },
      "design":{
        "raw_model":"UP/DOWN at exact +60s, binary-option style",
        "exec_model":"UP/DOWN/PASS from executable bid/ask PnL after spread/slippage",
        "magnitude_model":"expected raw +60s move; used as confidence/magnitude gate",
        "context":"M15/H1 location + M1/M5 structure + 5s/15s/30s price action + phase/sweep/exhaustion"
      },
      "anti_leakage":["completed bars only","exact +60s continuity","chronological split","validation-only threshold selection"],
      "dataset_manifest":man,"nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
