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
    return float(x.as_double()) if hasattr(x,"as_double") else float(x)

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

def metrics(name, side, long_pnl, short_pnl):
    side=np.asarray(side,int)
    pnl=np.where(side>0,long_pnl,np.where(side<0,short_pnl,0.0))
    mask=side!=0; t=pnl[mask]
    return {
        "name":name,"signals":int(len(side)),"trades":int(mask.sum()),
        "pass_rate_pct":float(100*(~mask).sum()/max(1,len(mask))),
        "WR_pct":float(100*(t>0).sum()/max(1,len(t))),
        "PF":pf_from(t),"net_usd_0p01lot_equiv":float(t.sum()),
        "return_pct_on_1000":float(t.sum()/10.0),
        "avg_trade_usd":float(t.mean()) if len(t) else 0.0,
        "max_DD_pct":max_dd_pct(t)
    }

def raw5(ticks):
    rows=[(pd.Timestamp(int(t.ts_event),unit="ns",tz="UTC"),fpx(t.bid_price),fpx(t.ask_price)) for t in ticks]
    df=pd.DataFrame(rows,columns=["ts","bid","ask"]).sort_values("ts").drop_duplicates("ts").set_index("ts")
    df["mid"]=(df.bid+df.ask)/2
    g=df.resample("5s",label="right",closed="right")
    x=pd.DataFrame({
        "bid":g.bid.last(),"ask":g.ask.last(),"open":g.mid.first(),
        "high":g.mid.max(),"low":g.mid.min(),"close":g.mid.last(),
        "ticks_5s":g.mid.count()
    }).dropna()
    x["mid"]=x.close
    x["spread"]=x.ask-x.bid
    return x

def add_bar_features(base, rule, prefix, roll_n=20):
    b=base[["open","high","low","close"]].resample(rule,label="right",closed="right").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    rng=(b.high-b.low).replace(0,np.nan)
    b[prefix+"_body"]=b.close-b.open
    b[prefix+"_range"]=rng
    b[prefix+"_body_ratio"]=(b.close-b.open).abs()/rng
    b[prefix+"_upper_wick"]=(b.high-b[["open","close"]].max(axis=1))/rng
    b[prefix+"_lower_wick"]=(b[["open","close"]].min(axis=1)-b.low)/rng
    b[prefix+"_close_loc"]=(b.close-b.low)/rng
    rh=b.high.rolling(roll_n).max(); rl=b.low.rolling(roll_n).min(); rr=(rh-rl).replace(0,np.nan)
    b[prefix+"_range_loc"]=(b.close-rl)/rr
    b[prefix+"_dist_hi"]=(rh-b.close)/rr
    b[prefix+"_dist_lo"]=(b.close-rl)/rr
    diff=b.close.diff()
    b[prefix+"_eff"]=b.close.diff(roll_n).abs()/diff.abs().rolling(roll_n).sum().replace(0,np.nan)
    b[prefix+"_slope3"]=b.close.diff(3)
    b[prefix+"_hh"]= (b.high>b.high.shift(1)).astype(float)
    b[prefix+"_hl"]= (b.low>b.low.shift(1)).astype(float)
    b[prefix+"_lh"]= (b.high<b.high.shift(1)).astype(float)
    b[prefix+"_ll"]= (b.low<b.low.shift(1)).astype(float)
    prev_hi=b.high.shift(1).rolling(roll_n).max()
    prev_lo=b.low.shift(1).rolling(roll_n).min()
    b[prefix+"_bos_up"]=(b.close>prev_hi).astype(float)
    b[prefix+"_bos_dn"]=(b.close<prev_lo).astype(float)
    cols=[c for c in b.columns if c.startswith(prefix+"_")]
    return b[cols].reindex(base.index,method="ffill")

def feature_frame(ticks):
    q=raw5(ticks)
    for s in (1,2,3,6,12):
        q[f"r{s*5}"]=q.mid-q.mid.shift(s)
    q["vel5"]=q.r5/5.0
    q["accel5"]=q.vel5-q.vel5.shift(1)
    q["vol15"]=q.mid.diff().rolling(3).std()
    q["vol30"]=q.mid.diff().rolling(6).std()
    q["vol60"]=q.mid.diff().rolling(12).std()
    q["spread_mean30"]=q.spread.rolling(6).mean()
    q["spread_ratio"]=q.spread/np.maximum(q.spread_mean30,1e-9)
    q["tick_mean30"]=q.ticks_5s.rolling(6).mean()
    q["trend_eff30"]=q.r30.abs()/np.maximum(q.mid.diff().abs().rolling(6).sum(),1e-9)
    # Price action from native 5-second bars.
    r=(q.high-q.low).replace(0,np.nan)
    q["pa_body"]=q.close-q.open
    q["pa_body_ratio"]=(q.close-q.open).abs()/r
    q["pa_upper_wick"]=(q.high-q[["open","close"]].max(axis=1))/r
    q["pa_lower_wick"]=(q[["open","close"]].min(axis=1)-q.low)/r
    q["pa_close_loc"]=(q.close-q.low)/r
    q["pa_inside"]=((q.high<=q.high.shift(1))&(q.low>=q.low.shift(1))).astype(float)
    q["pa_outside"]=((q.high>q.high.shift(1))&(q.low<q.low.shift(1))).astype(float)
    q["pa_break_up"]=(q.close>q.high.shift(1)).astype(float)
    q["pa_break_dn"]=(q.close<q.low.shift(1)).astype(float)

    # Multi-timeframe completed-bar context. No future bars are used.
    for rule,pfx,n in [("15s","s15",20),("30s","s30",20),("1min","m1",20),("5min","m5",20),("15min","m15",16),("1h","h1",12)]:
        q=q.join(add_bar_features(q,rule,pfx,n))

    q["future_bid"]=q.bid.shift(-12); q["future_ask"]=q.ask.shift(-12)
    q["future_ts"]=q.index.to_series().shift(-12)
    q["contiguous60"]=(q.future_ts-q.index.to_series()).dt.total_seconds().eq(60)
    return q

def fit_and_score(train,val,oos,features,long_col="long_pnl",short_col="short_pnl"):
    model=HistGradientBoostingClassifier(max_iter=220,learning_rate=0.05,max_leaf_nodes=31,
        min_samples_leaf=100,l2_regularization=0.8,random_state=310421)
    model.fit(train[features],train.y)
    classes=list(model.classes_); pi={c:i for i,c in enumerate(classes)}
    def probs(fr):
        p=model.predict_proba(fr[features])
        pb=p[:,pi[1]] if 1 in pi else np.zeros(len(fr))
        ps=p[:,pi[-1]] if -1 in pi else np.zeros(len(fr))
        return pb,ps
    def side(fr,thr):
        pb,ps=probs(fr); s=np.zeros(len(fr),int)
        s[(pb>=thr)&(pb>ps)]=1; s[(ps>=thr)&(ps>pb)]=-1
        return s,pb,ps
    grid=[0.45,0.50,0.54,0.58,0.62,0.66,0.70,0.74]
    vc=[]
    for th in grid:
        s,_,_=side(val,th)
        vc.append(metrics(f"thr_{th:.2f}",s,val[long_col].values,val[short_col].values))
    elig=[x for x in vc if x["trades"]>=150]
    best=max(elig,key=lambda x:(x["PF"],x["net_usd_0p01lot_equiv"])) if elig else vc[0]
    th=float(best["name"].split("_")[-1])
    s,pb,ps=side(oos,th)
    return model,th,metrics("oos",s,oos[long_col].values,oos[short_col].values),vc,pb,ps,s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_structure_pa_v2")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    a=ap.parse_args()
    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    q=feature_frame(ticks)
    slip=2*a.slippage_side
    q["long_pnl"]=q.future_bid-q.ask-slip
    q["short_pnl"]=q.bid-q.future_ask-slip
    q["y"]=0
    q.loc[(q.long_pnl>0)&(q.long_pnl>=q.short_pnl),"y"]=1
    q.loc[(q.short_pnl>0)&(q.short_pnl>q.long_pnl),"y"]=-1

    base=["r5","r10","r15","r30","r60","vel5","accel5","vol15","vol30","vol60","spread","spread_ratio","ticks_5s","tick_mean30","trend_eff30"]
    pa=base+["pa_body","pa_body_ratio","pa_upper_wick","pa_lower_wick","pa_close_loc","pa_inside","pa_outside","pa_break_up","pa_break_dn"]
    s15=[c for c in q.columns if c.startswith("s15_") or c.startswith("s30_")]
    struct=[c for c in q.columns if c.startswith("m1_") or c.startswith("m5_")]
    context=[c for c in q.columns if c.startswith("m15_") or c.startswith("h1_")]
    stages=[
        ("V1_BASE",base),
        ("PLUS_PRICE_ACTION",pa+s15),
        ("PLUS_M1_M5_STRUCTURE",pa+s15+struct),
        ("PLUS_M15_H1_CONTEXT",pa+s15+struct+context)
    ]
    allcols=sorted(set(sum((x[1] for x in stages),[])))
    d=q.loc[q.contiguous60,allcols+["y","long_pnl","short_pnl"]].replace([np.inf,-np.inf],np.nan).dropna().copy()
    n=len(d); i1=int(n*.60); i2=int(n*.80)
    train,val,oos=d.iloc[:i1],d.iloc[i1:i2],d.iloc[i2:]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    results=[]; val_tables={}
    for name,features in stages:
        _,th,st,vc,pb,ps,side=fit_and_score(train,val,oos,features)
        st["stage"]=name; st["threshold"]=th; st["feature_count"]=len(features)
        results.append(st); val_tables[name]=vc
        pd.DataFrame({"ts":oos.index.astype(str),"y":oos.y.values,"pbuy":pb,"psell":ps,"side":side,
                      "long_pnl":oos.long_pnl.values,"short_pnl":oos.short_pnl.values}).to_csv(out/f"oos_predictions_{name}.csv",index=False)
    summary={
        "experiment":"AMOS_VirtualBinary_StructurePA_60s_v2",
        "hypothesis":"5-second independent 60-second forecasts conditioned on price action, market structure and higher-timeframe location.",
        "raw_ticks":len(ticks),"usable_samples":n,
        "split":{"train":len(train),"validation":len(val),"oos":len(oos),"method":"chronological 60/20/20"},
        "execution":{"buy":"ask -> +60s bid","sell":"bid -> +60s ask","slippage_per_side":a.slippage_side},
        "stages":results,"validation":val_tables,
        "anti_leakage":["completed bars only","forward-filled only after bar close","exact +60s target continuity","chronological split","threshold selected on validation only"],
        "dataset_manifest":man,"nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    pd.DataFrame(results).to_csv(out/"stage_kpi.csv",index=False)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
