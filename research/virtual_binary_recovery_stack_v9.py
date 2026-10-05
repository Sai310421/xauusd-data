from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from virtual_binary_structure_pa_v2 import feature_frame
from virtual_binary_direct_udp_v5 import make_features
from virtual_binary_tickmicro_v6 import tick_micro_features, fit_cls
from virtual_binary_multihorizon_v8 import fit_meta, eval_meta, select_rule, apply_rule

HORIZONS=[75,90]
ADD_DISTANCES=[0.30,0.50,0.80,1.00]
MAX_LAYERS=[3,5,8,10]
PROGRESSIONS={
    "1_1_1":[1.0],
    "1_1p25_1p5":[1.0,1.25,1.50],
    "1_1p5_2":[1.0,1.50,2.00],
}
BASKET_TARGETS=[0.0,1.0,3.0,5.0]
MAX_HOLD_SEC=300
BASE_LOT=0.01

def lot_weight(prog, layer_idx):
    if len(prog)==1:
        return prog[0]
    if layer_idx < len(prog):
        return prog[layer_idx]
    # Continue the last slope conservatively instead of geometric martingale.
    step=prog[-1]-prog[-2]
    return prog[-1] + step*(layer_idx-(len(prog)-1))

def simulate_one(q, start_pos, side, add_dist, max_layers, prog, basket_target, slip_side):
    entry_row=q.iloc[start_pos]
    prices=[]; weights=[]; layers=0
    entry_price=float(entry_row.ask if side>0 else entry_row.bid)
    prices.append(entry_price); weights.append(lot_weight(prog,0)); layers=1
    last_add_ref=entry_price
    worst_float=0.0
    best_float=-1e18
    exit_pos=None; recovered=False
    max_steps=MAX_HOLD_SEC//5

    def basket_pnl(row):
        exit_px=float(row.bid if side>0 else row.ask)
        gross=0.0
        for p,w in zip(prices,weights):
            gross += ((exit_px-p) if side>0 else (p-exit_px))*w
        # one-way adverse slippage at each entry and one exit, scaled by lot weights
        entry_slip=sum(weights)*slip_side
        exit_slip=sum(weights)*slip_side
        return gross-entry_slip-exit_slip

    for k in range(1,max_steps+1):
        pos=start_pos+k
        if pos>=len(q): break
        row=q.iloc[pos]
        # Do not bridge gaps. 5-second grid must remain contiguous.
        dt=(q.index[pos]-q.index[pos-1]).total_seconds()
        if dt!=5: break

        px=float(row.bid if side>0 else row.ask)

        # Add only on adverse movement from the last add reference.
        while layers<max_layers:
            adverse=(last_add_ref-px) if side>0 else (px-last_add_ref)
            if adverse + 1e-12 < add_dist:
                break
            w=lot_weight(prog,layers)
            add_px=float(row.ask if side>0 else row.bid)
            prices.append(add_px); weights.append(w); layers+=1
            last_add_ref=add_px

        pnl=basket_pnl(row)
        worst_float=min(worst_float,pnl)
        best_float=max(best_float,pnl)
        if pnl>=basket_target:
            exit_pos=pos; recovered=True; break

    if exit_pos is None:
        exit_pos=min(start_pos+max_steps,len(q)-1)
        # if gap caused break, exit at last contiguous observed point
        if exit_pos<=start_pos: exit_pos=start_pos
    exit_row=q.iloc[exit_pos]
    final_pnl=basket_pnl(exit_row)

    total_w=sum(weights)
    avg_entry=sum(p*w for p,w in zip(prices,weights))/max(total_w,1e-12)
    return {
        "final_pnl":float(final_pnl),
        "recovered":bool(recovered),
        "layers":int(layers),
        "weight_sum":float(total_w),
        "lots":float(BASE_LOT*total_w),
        "avg_entry":float(avg_entry),
        "hold_sec":float((q.index[exit_pos]-q.index[start_pos]).total_seconds()),
        "worst_float":float(worst_float),
        "best_float":float(best_float),
        "exit_pos":int(exit_pos),
    }

def dd_pct_from_pnl(pnl, initial=1000.0):
    if not pnl: return 0.0
    arr=np.asarray(pnl,float)
    eq=initial+np.cumsum(arr)
    peak=np.maximum.accumulate(np.r_[initial,eq])[1:]
    return float(np.max((peak-eq)/np.maximum(peak,1e-12)*100))

def pf_from(pnl):
    arr=np.asarray(pnl,float)
    gp=arr[arr>0].sum(); gl=-arr[arr<0].sum()
    return float(gp/gl) if gl>0 else (float("inf") if gp>0 else 0.0)

def build_seed(q, base_cols, horizon):
    steps=horizon//5
    x=q.copy()
    x["future_bid_h"]=x.bid.shift(-steps)
    x["future_ask_h"]=x.ask.shift(-steps)
    x["future_ts_h"]=x.index.to_series().shift(-steps)
    x["future_mid_h"]=(x.future_bid_h+x.future_ask_h)/2
    x["raw_delta_h"]=x.future_mid_h-x.mid
    x["raw_dir"]=np.where(x.raw_delta_h>0,1,np.where(x.raw_delta_h<0,-1,0))
    x["contiguous_h"]=(x.future_ts_h-x.index.to_series()).dt.total_seconds().eq(horizon)
    d=x.loc[x.contiguous_h,base_cols+["raw_dir"]].replace([np.inf,-np.inf],np.nan).dropna().copy()
    n=len(d); i1=int(n*.50); i2=int(n*.70); i3=int(n*.85)
    bt=d.iloc[:i1]; mt=d.iloc[i1:i2]; va=d.iloc[i2:i3]; oo=d.iloc[i3:]
    base=fit_cls(bt,base_cols,"raw_dir")
    meta=fit_meta(mt,base,base_cols)
    vpred,vpc=eval_meta(va,base,meta,base_cols)
    best,_=select_rule(va.raw_dir.values,vpred,vpc)
    opred,opc=eval_meta(oo,base,meta,base_cols)
    final,use=apply_rule(opred,opc,best)
    # Map selected OOS timestamps back to q positions.
    pos_map=pd.Series(np.arange(len(q)),index=q.index)
    selected=pd.DataFrame({"ts":oo.index[use],"side":final[use]})
    selected["pos"]=selected.ts.map(pos_map).astype(int)
    return best, selected

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_recovery_stack_v9")
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

    seeds={}
    rules={}
    for h in HORIZONS:
        rule,sel=build_seed(q,base_cols,h)
        rules[str(h)]=rule
        seeds[f"H{h}_ORIG"]=sel.copy()
        rev=sel.copy(); rev["side"]=-rev["side"]
        seeds[f"H{h}_REV"]=rev

    results=[]
    detail_rows=[]
    for seed_name,entries in seeds.items():
        for add_dist in ADD_DISTANCES:
            for max_layers in MAX_LAYERS:
                for prog_name,prog in PROGRESSIONS.items():
                    for target in BASKET_TARGETS:
                        pnls=[]; rec=[]; lays=[]; lots=[]; holds=[]; worst=[]
                        for _,e in entries.iterrows():
                            r=simulate_one(
                                q,int(e.pos),int(e.side),add_dist,max_layers,prog,target,a.slippage_side
                            )
                            pnls.append(r["final_pnl"]); rec.append(r["recovered"]); lays.append(r["layers"])
                            lots.append(r["lots"]); holds.append(r["hold_sec"]); worst.append(r["worst_float"])
                        if not pnls:
                            continue
                        row={
                            "seed":seed_name,
                            "add_distance":add_dist,
                            "max_layers_cfg":max_layers,
                            "progression":prog_name,
                            "basket_target_usd":target,
                            "trades":len(pnls),
                            "recovery_rate_pct":float(100*np.mean(rec)),
                            "WR_pct":float(100*np.mean(np.asarray(pnls)>0)),
                            "PF":pf_from(pnls),
                            "net_usd_0p01lot_equiv":float(np.sum(pnls)),
                            "avg_trade_usd":float(np.mean(pnls)),
                            "max_closed_equity_DD_pct_on_1000":dd_pct_from_pnl(pnls),
                            "max_basket_floating_loss_usd":float(min(worst)),
                            "avg_layers":float(np.mean(lays)),
                            "max_layers_used":int(max(lays)),
                            "avg_lots_per_basket":float(np.mean(lots)),
                            "max_lots_per_basket":float(max(lots)),
                            "avg_recovery_hold_sec":float(np.mean(holds)),
                            "max_hold_sec":float(max(holds)),
                        }
                        results.append(row)

    rdf=pd.DataFrame(results)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rdf.to_csv(out/"recovery_grid_kpi.csv",index=False)

    # Rank with a conservative score: PF first, then net, but flag extreme DD/exposure.
    valid=rdf[(rdf.trades>=100)].copy()
    best_pf=valid.sort_values(["PF","net_usd_0p01lot_equiv"],ascending=[False,False]).head(20)
    best_net=valid.sort_values(["net_usd_0p01lot_equiv","PF"],ascending=[False,False]).head(20)
    best_pf.to_csv(out/"top20_by_pf.csv",index=False)
    best_net.to_csv(out/"top20_by_net.csv",index=False)

    result={
        "experiment":"AMOS_VirtualBinary_RecoveryStack_v9",
        "raw_ticks":len(ticks),
        "seed_rules":rules,
        "seed_counts":{k:int(len(v)) for k,v in seeds.items()},
        "grid":{
            "add_distance":ADD_DISTANCES,
            "max_layers":MAX_LAYERS,
            "progressions":PROGRESSIONS,
            "basket_targets":BASKET_TARGETS,
            "max_hold_sec":MAX_HOLD_SEC,
            "base_lot":BASE_LOT
        },
        "best_by_pf":best_pf.iloc[0].to_dict() if len(best_pf) else None,
        "best_by_net":best_net.iloc[0].to_dict() if len(best_net) else None,
        "risk_note":"Per-basket floating DD and exposure are measured; aggregate account-level overlapping floating DD/margin is a required next gate before live use.",
        "fixed_requirements":{"decision_clock":"every 5 seconds","positions":"independent","overlap":"allowed"},
        "dataset_manifest":man,
        "nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
