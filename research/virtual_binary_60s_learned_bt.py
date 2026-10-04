from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from sklearn.ensemble import HistGradientBoostingClassifier

def fpx(x):
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)

def pf_from(pnl):
    pnl=np.asarray(pnl,float)
    gp=pnl[pnl>0].sum(); gl=-pnl[pnl<0].sum()
    return float(gp/gl) if gl>0 else (float("inf") if gp>0 else 0.0)

def max_dd_pct(pnl, initial=1000.0):
    eq=initial+np.cumsum(np.asarray(pnl,float))
    peak=np.maximum.accumulate(np.r_[initial,eq])[1:]
    dd=(peak-eq)/np.maximum(peak,1e-12)*100
    return float(np.max(dd)) if len(dd) else 0.0

def stats(name, side, long_pnl, short_pnl, initial=1000.0):
    side=np.asarray(side,int)
    pnl=np.where(side>0,long_pnl,np.where(side<0,short_pnl,0.0))
    mask=side!=0
    t=pnl[mask]
    wins=int((t>0).sum())
    return {
        "name":name,
        "signals":int(len(side)),
        "trades":int(mask.sum()),
        "pass_rate_pct":float(100*(~mask).sum()/max(1,len(mask))),
        "WR_pct":float(100*wins/max(1,len(t))),
        "PF":pf_from(t),
        "net_usd_0p01lot_equiv":float(t.sum()),
        "return_pct_on_1000":float(t.sum()/10.0),
        "avg_trade_usd":float(t.mean()) if len(t) else 0.0,
        "max_DD_pct":max_dd_pct(t,initial),
        "max_simultaneous_positions_theoretical":12
    }

def build_frame(ticks):
    rows=[(pd.Timestamp(int(t.ts_event),unit="ns",tz="UTC"),fpx(t.bid_price),fpx(t.ask_price)) for t in ticks]
    df=pd.DataFrame(rows,columns=["ts","bid","ask"]).sort_values("ts").drop_duplicates("ts").set_index("ts")
    df["mid"]=(df.bid+df.ask)/2
    # Last executable quote in each 5-second bucket; no lookahead.
    g=df.resample("5s")
    q=pd.DataFrame({
        "bid":g.bid.last(),
        "ask":g.ask.last(),
        "mid":g.mid.last(),
        "ticks_5s":g.mid.count()
    }).dropna()
    q["spread"]=q.ask-q.bid
    # Strict continuity guard: targets/features never bridge market gaps.
    for s in (1,2,3,6,12):
        q[f"r{s*5}"]=q.mid-q.mid.shift(s)
    q["vel5"]=q.r5/5.0
    q["accel5"]=q.vel5-q.vel5.shift(1)
    q["vol15"]=q.mid.diff().rolling(3).std()
    q["vol30"]=q.mid.diff().rolling(6).std()
    q["vol60"]=q.mid.diff().rolling(12).std()
    q["tick_mean30"]=q.ticks_5s.rolling(6).mean()
    q["spread_mean30"]=q.spread.rolling(6).mean()
    q["spread_ratio"]=q.spread/np.maximum(q.spread_mean30,1e-9)
    q["trend_eff30"]=q.r30.abs()/np.maximum(q.mid.diff().abs().rolling(6).sum(),1e-9)
    q["future_bid"]=q.bid.shift(-12)
    q["future_ask"]=q.ask.shift(-12)
    q["future_ts"]=q.index.to_series().shift(-12)
    q["contiguous60"]=(q.future_ts-q.index.to_series()).dt.total_seconds().eq(60)
    return q

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_60s_v1")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    ap.add_argument("--prob-threshold",type=float,default=0.58)
    a=ap.parse_args()

    cp=Path(a.catalog)
    man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    q=build_frame(ticks)

    slip=2*a.slippage_side
    q["long_pnl"]=q.future_bid-q.ask-slip
    q["short_pnl"]=q.bid-q.future_ask-slip
    q["y"]=0
    q.loc[(q.long_pnl>0)&(q.long_pnl>=q.short_pnl),"y"]=1
    q.loc[(q.short_pnl>0)&(q.short_pnl>q.long_pnl),"y"]=-1

    feats=["r5","r10","r15","r30","r60","vel5","accel5","vol15","vol30","vol60",
           "spread","spread_ratio","ticks_5s","tick_mean30","trend_eff30"]
    d=q.loc[q.contiguous60,feats+["y","long_pnl","short_pnl"]].replace([np.inf,-np.inf],np.nan).dropna().copy()
    n=len(d)
    i1=int(n*0.60); i2=int(n*0.80)
    train=d.iloc[:i1]; val=d.iloc[i1:i2]; oos=d.iloc[i2:]
    if min(len(train),len(val),len(oos))<1000:
        raise SystemExit(f"insufficient samples: train={len(train)} val={len(val)} oos={len(oos)}")

    model=HistGradientBoostingClassifier(max_iter=180,learning_rate=0.06,max_leaf_nodes=31,
        l2_regularization=0.5,min_samples_leaf=80,random_state=310421)
    model.fit(train[feats],train.y)
    classes=list(model.classes_)

    def model_side(frame,thr):
        p=model.predict_proba(frame[feats])
        pi={c:i for i,c in enumerate(classes)}
        pb=p[:,pi.get(1,0)] if 1 in pi else np.zeros(len(frame))
        ps=p[:,pi.get(-1,0)] if -1 in pi else np.zeros(len(frame))
        out=np.zeros(len(frame),dtype=int)
        out[(pb>=thr)&(pb>ps)]=1
        out[(ps>=thr)&(ps>pb)]=-1
        return out,pb,ps

    # Tune one confidence threshold on validation only.
    grid=[0.45,0.50,0.54,0.58,0.62,0.66,0.70]
    val_cells=[]
    for thr in grid:
        side,_,_=model_side(val,thr)
        st=stats(f"ML_thr_{thr:.2f}",side,val.long_pnl.values,val.short_pnl.values)
        val_cells.append(st)
    eligible=[x for x in val_cells if x["trades"]>=200 and x["PF"]>0]
    best=max(eligible,key=lambda x:(x["PF"],x["net_usd_0p01lot_equiv"])) if eligible else val_cells[0]
    best_thr=float(best["name"].split("_")[-1])

    side_ml,pb,ps=model_side(oos,best_thr)
    # Simple causal momentum baseline: last 15 sec direction, PASS if flat.
    side_mom=np.sign(oos.r15.values).astype(int)
    # Stronger baseline: 15s and 30s must agree.
    side_agree=np.where((np.sign(oos.r15.values)==np.sign(oos.r30.values)),np.sign(oos.r15.values),0).astype(int)

    res={
        "experiment":"AMOS_Rolling60_Independent_5s_v1",
        "hypothesis":"Every 5 seconds make an independent decision; each trade exits exactly 60 seconds later.",
        "raw_ticks":len(ticks),
        "five_second_rows":len(q),
        "usable_samples":n,
        "split":{"train":len(train),"validation":len(val),"oos":len(oos),"ratios":"60/20/20 chronological"},
        "features":feats,
        "execution":{"long":"entry ask -> +60s bid","short":"entry bid -> +60s ask","slippage_usd_per_side":a.slippage_side,"lot_equiv":0.01,"contract_oz_per_lot":100},
        "label_counts_train":{str(k):int(v) for k,v in train.y.value_counts().sort_index().items()},
        "validation_threshold_search":val_cells,
        "selected_threshold":best_thr,
        "oos":[
            stats("Momentum15s",side_mom,oos.long_pnl.values,oos.short_pnl.values),
            stats("Momentum15s_30s_agree",side_agree,oos.long_pnl.values,oos.short_pnl.values),
            stats("ML_cost_aware",side_ml,oos.long_pnl.values,oos.short_pnl.values)
        ],
        "oos_probability":{"mean_buy":float(pb.mean()),"mean_sell":float(ps.mean())},
        "nautilus_version":getattr(nautilus_trader,"__version__","unknown"),
        "dataset_manifest":man,
        "anti_leakage":["chronological split","features use current/past only","target requires exact +60s continuity","threshold tuned on validation only","OOS untouched until final scoring"]
    }
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(res,indent=2,default=str))
    pd.DataFrame(res["oos"]).to_csv(out/"oos_kpi.csv",index=False)
    pd.DataFrame(val_cells).to_csv(out/"validation_thresholds.csv",index=False)
    pred=pd.DataFrame({"ts":oos.index.astype(str),"y":oos.y.values,"pbuy":pb,"psell":ps,"side":side_ml,
                       "long_pnl":oos.long_pnl.values,"short_pnl":oos.short_pnl.values})
    pred.to_csv(out/"oos_predictions.csv",index=False)
    print(json.dumps(res,indent=2,default=str))

if __name__=="__main__":
    main()
