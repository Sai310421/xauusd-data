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

def metric(name,side,long_pnl,short_pnl):
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

def add_phase_features(q):
    z=pd.DataFrame(index=q.index)

    # Micro action
    z["micro_dir"]=np.sign(q["r15"]+q["r30"])
    z["micro_accel"]=np.sign(q["accel5"])
    z["wick_bias"]=q["pa_lower_wick"].fillna(0)-q["pa_upper_wick"].fillna(0)
    z["close_strength"]=2*q["pa_close_loc"].fillna(.5)-1
    z["break_bias"]=q["pa_break_up"].fillna(0)-q["pa_break_dn"].fillna(0)
    z["body_strength"]=(q["pa_body_ratio"].fillna(0)*np.sign(q["pa_body"].fillna(0))).clip(-1,1)

    # HTF trend/location/sweep/exhaustion proxies
    for p in ["m1","m5","m15","h1"]:
        loc=(2*q[f"{p}_range_loc"].fillna(.5)-1).clip(-1,1)
        dir_raw=np.sign(q[f"{p}_slope3"].fillna(0)) + 1.5*(q[f"{p}_bos_up"].fillna(0)-q[f"{p}_bos_dn"].fillna(0))
        dir_raw += 0.5*(q[f"{p}_hh"].fillna(0)+q[f"{p}_hl"].fillna(0)-q[f"{p}_lh"].fillna(0)-q[f"{p}_ll"].fillna(0))
        direction=np.tanh(dir_raw)
        eff=q[f"{p}_eff"].fillna(0).clip(0,1)

        z[f"{p}_trend"]=direction
        z[f"{p}_loc"]=loc
        z[f"{p}_eff"]=eff
        z[f"{p}_near_high"]=(loc>0.65).astype(float)
        z[f"{p}_near_low"]=(loc<-0.65).astype(float)
        z[f"{p}_late_up"]=(direction>0.35).astype(float)*(loc>0.55).astype(float)*eff
        z[f"{p}_late_dn"]=(direction<-0.35).astype(float)*(loc<-0.55).astype(float)*eff
        z[f"{p}_pullback_up"]=(direction>0.35).astype(float)*(loc<0.15).astype(float)
        z[f"{p}_pullback_dn"]=(direction<-0.35).astype(float)*(loc>-0.15).astype(float)

    # Liquidity/sweep-style features from current micro bar vs recent range.
    hi6=q["high"].shift(1).rolling(6).max()
    lo6=q["low"].shift(1).rolling(6).min()
    z["sweep_high_30"] = ((q["high"]>hi6) & (q["close"]<hi6)).astype(float)
    z["sweep_low_30"]  = ((q["low"]<lo6) & (q["close"]>lo6)).astype(float)
    hi12=q["high"].shift(1).rolling(12).max()
    lo12=q["low"].shift(1).rolling(12).min()
    z["sweep_high_60"] = ((q["high"]>hi12) & (q["close"]<hi12)).astype(float)
    z["sweep_low_60"]  = ((q["low"]<lo12) & (q["close"]>lo12)).astype(float)

    # Continuation / reversal priors
    z["htf_trend"]=(z["m5_trend"]+z["m15_trend"]+z["h1_trend"])/3
    z["htf_loc"]=(z["m5_loc"]+z["m15_loc"]+z["h1_loc"])/3
    z["continuation_prior"]=z["htf_trend"]*(
        0.45*z["micro_dir"]+0.25*z["micro_accel"]+0.20*z["break_bias"]+0.10*z["body_strength"]
    )
    exhaustion = (
        z["m5_late_up"]+z["m15_late_up"]+z["h1_late_up"]
        -z["m5_late_dn"]-z["m15_late_dn"]-z["h1_late_dn"]
    )/3
    sweep = (z["sweep_high_30"]+z["sweep_high_60"]-z["sweep_low_30"]-z["sweep_low_60"])/2
    wick_rev = -np.sign(z["htf_trend"])*z["wick_bias"]
    z["reversal_prior"]=exhaustion + sweep + 0.5*wick_rev

    return z.replace([np.inf,-np.inf],np.nan)

def fit_classifier(train,features):
    m=HistGradientBoostingClassifier(max_iter=220,learning_rate=0.05,max_leaf_nodes=23,
        min_samples_leaf=100,l2_regularization=1.0,random_state=310421)
    m.fit(train[features],train["mode"])
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_phase_contrev_v4")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    a=ap.parse_args()

    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    q=feature_frame(ticks)
    q=q.join(add_phase_features(q).add_prefix("phase_"))

    slip=2*a.slippage_side
    q["long_pnl"]=q.future_bid-q.ask-slip
    q["short_pnl"]=q.bid-q.future_ask-slip

    phase_cols=[c for c in q.columns if c.startswith("phase_")]
    d=q.loc[q.contiguous60,phase_cols+["long_pnl","short_pnl"]].replace([np.inf,-np.inf],np.nan).dropna().copy()

    # mode labels: +1 continuation, -1 reversal, 0 pass.
    # Determine the prevailing HTF trend sign from compressed trend feature.
    trend=np.sign(d["phase_htf_trend"].values)
    long=d.long_pnl.values; short=d.short_pnl.values
    best_dir=np.where(long>short,1,-1)
    best_pnl=np.maximum(long,short)
    d["mode"]=0
    cont=(best_pnl>0)&(best_dir==trend)&(trend!=0)
    rev=(best_pnl>0)&(best_dir==-trend)&(trend!=0)
    d.loc[cont,"mode"]=1
    d.loc[rev,"mode"]=-1

    n=len(d); i1=int(n*.60); i2=int(n*.80)
    train,val,oos=d.iloc[:i1],d.iloc[i1:i2],d.iloc[i2:]
    model=fit_classifier(train,phase_cols)
    classes=list(model.classes_); idx={c:i for i,c in enumerate(classes)}

    def infer(fr,thr):
        p=model.predict_proba(fr[phase_cols])
        pc=p[:,idx[1]] if 1 in idx else np.zeros(len(fr))
        pr=p[:,idx[-1]] if -1 in idx else np.zeros(len(fr))
        trend=np.sign(fr["phase_htf_trend"].values).astype(int)
        side=np.zeros(len(fr),int)
        cont=(pc>=thr)&(pc>pr)&(trend!=0)
        rev=(pr>=thr)&(pr>pc)&(trend!=0)
        side[cont]=trend[cont]
        side[rev]=-trend[rev]
        return side,pc,pr

    grid=[0.40,0.45,0.50,0.55,0.60,0.65,0.70]
    val_rows=[]
    for th in grid:
        s,_,_=infer(val,th)
        m=metric(f"thr_{th:.2f}",s,val.long_pnl.values,val.short_pnl.values)
        m["threshold"]=th; val_rows.append(m)
    elig=[x for x in val_rows if x["trades"]>=150]
    best=max(elig,key=lambda x:(x["PF"],x["net_usd_0p01lot_equiv"])) if elig else max(val_rows,key=lambda x:x["net_usd_0p01lot_equiv"])
    th=float(best["threshold"])

    side,pc,pr=infer(oos,th)
    base=metric("PHASE_CONTREV",side,oos.long_pnl.values,oos.short_pnl.values)

    # Diagnostics: flip-only tests.
    flip_all=-side
    buy_flip=np.where(side>0,-1,side)
    sell_flip=np.where(side<0,1,side)
    diags=[
      metric("PHASE_CONTREV",side,oos.long_pnl.values,oos.short_pnl.values),
      metric("FLIP_ALL",flip_all,oos.long_pnl.values,oos.short_pnl.values),
      metric("FLIP_BUY_ONLY",buy_flip,oos.long_pnl.values,oos.short_pnl.values),
      metric("FLIP_SELL_ONLY",sell_flip,oos.long_pnl.values,oos.short_pnl.values),
    ]

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(val_rows).to_csv(out/"validation_thresholds.csv",index=False)
    pd.DataFrame(diags).to_csv(out/"oos_kpi.csv",index=False)
    pd.DataFrame({"ts":oos.index.astype(str),"mode_true":oos.mode.values,"p_cont":pc,"p_rev":pr,"side":side,
                  "htf_trend":np.sign(oos["phase_htf_trend"].values).astype(int),
                  "long_pnl":oos.long_pnl.values,"short_pnl":oos.short_pnl.values}).to_csv(out/"oos_predictions.csv",index=False)

    result={
      "experiment":"AMOS_VirtualBinary_Phase_ContRev_v4",
      "raw_ticks":len(ticks),"usable_samples":n,
      "split":{"train":len(train),"validation":len(val),"oos":len(oos),"method":"chronological 60/20/20"},
      "threshold":th,"validation":val_rows,"oos":diags,
      "design":{
        "key_change":"separate HTF trend direction from 60s trade direction",
        "mode":"classify continuation vs reversal vs pass first",
        "phase_features":"swing location, BOS/trend, micro PA, sweeps, late-trend/exhaustion, pullback proxies",
        "execution":"BUY ask -> +60s bid; SELL bid -> +60s ask; fixed60 only"
      },
      "dataset_manifest":man,"nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
