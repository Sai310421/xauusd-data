from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from sklearn.ensemble import HistGradientBoostingRegressor
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from virtual_binary_structure_pa_v2 import feature_frame, fpx

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

def summarize(name,trades):
    if not trades:
        return {"name":name,"trades":0,"WR_pct":0.0,"PF":0.0,"net_usd_0p01lot_equiv":0.0,"return_pct_on_1000":0.0,"avg_trade_usd":0.0,"max_DD_pct":0.0,"avg_hold_sec":0.0}
    p=np.asarray([x["pnl"] for x in trades],float)
    h=np.asarray([x["hold_sec"] for x in trades],float)
    return {"name":name,"trades":len(trades),"WR_pct":float(100*(p>0).sum()/len(p)),"PF":pf_from(p),
            "net_usd_0p01lot_equiv":float(p.sum()),"return_pct_on_1000":float(p.sum()/10),
            "avg_trade_usd":float(p.mean()),"max_DD_pct":max_dd_pct(p),"avg_hold_sec":float(h.mean()),
            "hold_gt60_pct":float(100*(h>60).sum()/len(h))}

def make_state(q):
    z=pd.DataFrame(index=q.index)
    # Micro price-action summary: deliberately compressed.
    z["micro_momentum"]=np.tanh((q["r15"]+q["r30"])/(2*np.maximum(q["vol30"],1e-6)))
    z["micro_accel"]=np.tanh(q["accel5"]/np.maximum(q["vol15"],1e-6))
    z["wick_imbalance"]=q["pa_lower_wick"].fillna(0)-q["pa_upper_wick"].fillna(0)
    z["close_strength"]=2*q["pa_close_loc"].fillna(.5)-1
    z["break_state"]=q["pa_break_up"].fillna(0)-q["pa_break_dn"].fillna(0)
    z["spread_pressure"]=q["spread_ratio"].clip(0,5)
    z["tick_pressure"]=(q["ticks_5s"]/np.maximum(q["tick_mean30"],1)).clip(0,5)

    # "Where are we?" scores. One small state per TF, not dozens of raw inputs.
    for pfx in ["m1","m5","m15","h1"]:
        slope=np.sign(q[f"{pfx}_slope3"].fillna(0))
        bos=q[f"{pfx}_bos_up"].fillna(0)-q[f"{pfx}_bos_dn"].fillna(0)
        struct=(q[f"{pfx}_hh"].fillna(0)+q[f"{pfx}_hl"].fillna(0)-q[f"{pfx}_lh"].fillna(0)-q[f"{pfx}_ll"].fillna(0))/2
        loc=2*q[f"{pfx}_range_loc"].fillna(.5)-1
        eff=q[f"{pfx}_eff"].fillna(0).clip(0,1)
        z[f"{pfx}_direction"]=np.tanh(0.8*slope+1.2*bos+0.7*struct)
        z[f"{pfx}_location"]=loc.clip(-1,1)
        z[f"{pfx}_efficiency"]=eff
        # continuation score: direction aligned with location away from exhaustion extremes.
        z[f"{pfx}_continuation"]=z[f"{pfx}_direction"]*(1-np.abs(z[f"{pfx}_location"]))*z[f"{pfx}_efficiency"]
    z["context_alignment"]=(z["m1_direction"]+z["m5_direction"]+z["m15_direction"]+z["h1_direction"])/4
    z["context_continuation"]=(z["m1_continuation"]+z["m5_continuation"]+z["m15_continuation"]+z["h1_continuation"])/4
    z["micro_context_agree"]=z["micro_momentum"]*z["context_alignment"]
    return z.replace([np.inf,-np.inf],np.nan)

def fit_reg(train,features,target):
    m=HistGradientBoostingRegressor(max_iter=220,learning_rate=0.045,max_leaf_nodes=23,
        min_samples_leaf=120,l2_regularization=1.0,random_state=310421,loss="squared_error")
    m.fit(train[features],train[target])
    return m

def choose_threshold(val,evl,evs):
    # Validation-only EV threshold. Require enough trades and optimize PF then net.
    candidates=[0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.60]
    rows=[]
    for th in candidates:
        side=np.where((evl>th)&(evl>evs),1,np.where((evs>th)&(evs>evl),-1,0))
        pnl=np.where(side>0,val.long_pnl.values,np.where(side<0,val.short_pnl.values,0.0))
        t=pnl[side!=0]
        rows.append({"threshold":th,"trades":int(len(t)),"PF":pf_from(t),"net":float(t.sum()),"WR":float(100*(t>0).sum()/max(1,len(t)))})
    eligible=[x for x in rows if x["trades"]>=150]
    best=max(eligible,key=lambda x:(x["PF"],x["net"])) if eligible else max(rows,key=lambda x:x["net"])
    return float(best["threshold"]),rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_state_ev_hold_v3")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    ap.add_argument("--max-hold-sec",type=int,default=180)
    a=ap.parse_args()

    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    q=feature_frame(ticks)
    state=make_state(q)
    q=q.join(state.add_prefix("state_"))

    slip=2*a.slippage_side
    q["long_pnl"]=q.future_bid-q.ask-slip
    q["short_pnl"]=q.bid-q.future_ask-slip
    state_cols=[c for c in q.columns if c.startswith("state_")]
    d=q.loc[q.contiguous60,state_cols+["bid","ask","future_bid","future_ask","long_pnl","short_pnl"]].replace([np.inf,-np.inf],np.nan).dropna().copy()

    n=len(d); i1=int(n*.60); i2=int(n*.80)
    train,val,oos=d.iloc[:i1],d.iloc[i1:i2],d.iloc[i2:]
    ml=fit_reg(train,state_cols,"long_pnl"); ms=fit_reg(train,state_cols,"short_pnl")
    evl_val=ml.predict(val[state_cols]); evs_val=ms.predict(val[state_cols])
    entry_th,threshold_table=choose_threshold(val,evl_val,evs_val)

    evl=ml.predict(oos[state_cols]); evs=ms.predict(oos[state_cols])
    oos=oos.copy(); oos["ev_long"]=evl; oos["ev_short"]=evs
    oos["side"]=np.where((evl>entry_th)&(evl>evs),1,np.where((evs>entry_th)&(evs>evl),-1,0))

    # Fixed 60s baseline using the EV model.
    fixed=[]
    for ts,r in oos[oos.side!=0].iterrows():
        side=int(r.side)
        pnl=float(r.long_pnl if side>0 else r.short_pnl)
        fixed.append({"entry":str(ts),"side":side,"pnl":pnl,"hold_sec":60})

    # Learned hold: at each 60s checkpoint, re-evaluate expected NEXT 60s in same direction.
    # Exit uses executable bid/ask. Initial entry pays spread + slippage; each extension does not re-pay spread.
    full=q.join(make_state(q).add_prefix("state2_"))
    full_state=[c for c in full.columns if c.startswith("state2_")]
    # map feature names to trained names by stripping state2_/state_
    rename={c:"state_"+c[len("state2_"):] for c in full_state}
    hold=[]
    max_steps=max(1,a.max_hold_sec//60)
    for ts,r in oos[oos.side!=0].iterrows():
        side=int(r.side); entry_ask=float(r.ask); entry_bid=float(r.bid)
        exit_ts=ts+pd.Timedelta(seconds=60)
        hold_sec=60
        for step in range(1,max_steps):
            if exit_ts not in full.index: break
            rr=full.loc[exit_ts]
            feat=pd.DataFrame([{rename[c]:rr[c] for c in full_state}])
            if feat.isna().any(axis=None): break
            next_ev=float(ml.predict(feat[state_cols])[0] if side>0 else ms.predict(feat[state_cols])[0])
            # Continue only if expected incremental 60s edge remains above entry threshold.
            if next_ev<=entry_th: break
            hold_sec += 60
            exit_ts += pd.Timedelta(seconds=60)
        if exit_ts not in full.index:
            exit_ts=ts+pd.Timedelta(seconds=hold_sec)
        if exit_ts not in full.index: continue
        ex=full.loc[exit_ts]
        if side>0:
            pnl=float(ex.bid-entry_ask-slip)
        else:
            pnl=float(entry_bid-ex.ask-slip)
        hold.append({"entry":str(ts),"side":side,"pnl":pnl,"hold_sec":hold_sec})

    result={
        "experiment":"AMOS_VirtualBinary_StateEV_Hold_v3",
        "raw_ticks":len(ticks),"usable_samples":n,
        "split":{"train":len(train),"validation":len(val),"oos":len(oos),"method":"chronological 60/20/20"},
        "state_features":state_cols,"entry_threshold":entry_th,"validation_thresholds":threshold_table,
        "kpi":[summarize("EV_FIXED_60",fixed),summarize("EV_LEARNED_HOLD",hold)],
        "design":{
            "entry":"every 5s independently; choose long/short/pass from predicted cost-after 60s PnL",
            "context":"compressed M1/M5/M15/H1 direction, range location, efficiency and continuation state",
            "price_action":"compressed micro momentum, acceleration, wick imbalance, close strength and breakout state",
            "hold":"re-evaluate same-direction expected incremental 60s edge at 60s checkpoints; maximum hold configured",
            "execution":"BUY ask -> exit bid, SELL bid -> exit ask, slippage applied"
        },
        "anti_leakage":["chronological split","completed-bar context only","validation-only entry threshold","hold decision uses state at checkpoint only"],
        "dataset_manifest":man,"nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    pd.DataFrame(result["kpi"]).to_csv(out/"kpi.csv",index=False)
    pd.DataFrame(threshold_table).to_csv(out/"validation_thresholds.csv",index=False)
    pd.DataFrame(fixed).to_csv(out/"trades_fixed60.csv",index=False)
    pd.DataFrame(hold).to_csv(out/"trades_learned_hold.csv",index=False)
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
