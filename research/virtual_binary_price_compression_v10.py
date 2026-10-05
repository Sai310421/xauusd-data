from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from virtual_binary_structure_pa_v2 import feature_frame
from virtual_binary_direct_udp_v5 import make_features
from virtual_binary_tickmicro_v6 import tick_micro_features
from virtual_binary_recovery_stack_v9 import build_seed, pf_from, dd_pct_from_pnl

HORIZONS=[75,90]
COMPRESSION=[0.75,0.50,0.33,0.25]  # remaining avg-price gap after each recovery add
MAX_LAYERS=[5,8,10]
MAX_ADD_MULT=[0.5,1.0,2.0]          # cap each new add as multiple of current basket weight
BASKET_TARGETS=[0.0,1.0,3.0]
MAX_HOLD_SEC=300
BASE_LOT=0.01

def basket_state(prices,weights):
    Q=float(sum(weights))
    avg=float(sum(p*w for p,w in zip(prices,weights))/max(Q,1e-12))
    return Q,avg

def simulate_one(q,start_pos,side,compression,max_layers,max_add_mult,target,slip_side):
    row0=q.iloc[start_pos]
    entry=float(row0.ask if side>0 else row0.bid)
    prices=[entry]; weights=[1.0]
    layers=1; worst_float=0.0; best_float=-1e18; recovered=False
    last_pos=start_pos

    def basket_pnl(row):
        exit_px=float(row.bid if side>0 else row.ask)
        gross=sum((((exit_px-p) if side>0 else (p-exit_px))*w) for p,w in zip(prices,weights))
        return gross - slip_side*sum(weights) - slip_side*sum(weights)

    for k in range(1,MAX_HOLD_SEC//5+1):
        pos=start_pos+k
        if pos>=len(q): break
        if (q.index[pos]-q.index[pos-1]).total_seconds()!=5: break
        last_pos=pos
        row=q.iloc[pos]
        pnl=basket_pnl(row)
        worst_float=min(worst_float,pnl); best_float=max(best_float,pnl)

        if pnl>=target:
            recovered=True
            break

        # User requirement: every 5 seconds, if basket is negative,
        # add in same direction specifically to pull weighted average toward price.
        if pnl<0 and layers<max_layers:
            add_px=float(row.ask if side>0 else row.bid)
            Q,avg=basket_state(prices,weights)
            gap=(avg-add_px) if side>0 else (add_px-avg)
            if gap>0:
                # desired new average keeps only 'compression' fraction of the old gap.
                desired=(add_px + compression*(avg-add_px)) if side>0 else (add_px - compression*(add_px-avg))
                den=(desired-add_px) if side>0 else (add_px-desired)
                num=(avg-desired) if side>0 else (desired-avg)
                q_req=Q*num/max(den,1e-12)
                q_add=max(0.0,min(q_req,Q*max_add_mult))
                if q_add>1e-9:
                    prices.append(add_px); weights.append(q_add); layers+=1
                    pnl=basket_pnl(row)
                    worst_float=min(worst_float,pnl); best_float=max(best_float,pnl)
                    if pnl>=target:
                        recovered=True
                        break

    final_row=q.iloc[last_pos]
    final_pnl=basket_pnl(final_row)
    Q,avg=basket_state(prices,weights)
    market=float(final_row.bid if side>0 else final_row.ask)
    final_gap=abs(avg-market)
    return {
        "final_pnl":float(final_pnl),"recovered":bool(recovered),"layers":int(layers),
        "weight_sum":float(Q),"lots":float(BASE_LOT*Q),"avg_entry":float(avg),
        "final_gap_to_price":float(final_gap),
        "hold_sec":float((q.index[last_pos]-q.index[start_pos]).total_seconds()),
        "worst_float":float(worst_float),"best_float":float(best_float)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",default="catalog/raw_bidask")
    ap.add_argument("--out",default="results/virtual_binary_price_compression_v10")
    ap.add_argument("--slippage-side",type=float,default=0.10)
    a=ap.parse_args()

    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])

    q=feature_frame(ticks)
    q=q.join(make_features(q).add_prefix("ctx_")).join(tick_micro_features(ticks))
    base_cols=[c for c in q.columns if c.startswith("ctx_") or c.startswith("tm_")]

    seeds={}; rules={}
    for h in HORIZONS:
        rule,sel=build_seed(q,base_cols,h)
        rules[str(h)]=rule
        seeds[f"H{h}_ORIG"]=sel.copy()
        rev=sel.copy(); rev["side"]=-rev["side"]
        seeds[f"H{h}_REV"]=rev

    rows=[]
    for seed_name,entries in seeds.items():
        for comp in COMPRESSION:
            for max_layers in MAX_LAYERS:
                for cap in MAX_ADD_MULT:
                    for target in BASKET_TARGETS:
                        pnls=[]; rec=[]; lays=[]; lots=[]; holds=[]; worst=[]; gaps=[]
                        for _,e in entries.iterrows():
                            r=simulate_one(q,int(e.pos),int(e.side),comp,max_layers,cap,target,a.slippage_side)
                            pnls.append(r["final_pnl"]); rec.append(r["recovered"]); lays.append(r["layers"])
                            lots.append(r["lots"]); holds.append(r["hold_sec"]); worst.append(r["worst_float"])
                            gaps.append(r["final_gap_to_price"])
                        if not pnls: continue
                        rows.append({
                            "seed":seed_name,"compression_remaining_gap":comp,
                            "max_layers_cfg":max_layers,"max_add_multiple_current_weight":cap,
                            "basket_target_usd":target,"trades":len(pnls),
                            "recovery_rate_pct":float(100*np.mean(rec)),
                            "WR_pct":float(100*np.mean(np.asarray(pnls)>0)),
                            "PF":pf_from(pnls),"net_usd_0p01lot_equiv":float(np.sum(pnls)),
                            "avg_trade_usd":float(np.mean(pnls)),
                            "closed_equity_DD_pct_on_1000":dd_pct_from_pnl(pnls),
                            "max_basket_floating_loss_usd":float(min(worst)),
                            "avg_layers":float(np.mean(lays)),"max_layers_used":int(max(lays)),
                            "avg_lots_per_basket":float(np.mean(lots)),"max_lots_per_basket":float(max(lots)),
                            "avg_hold_sec":float(np.mean(holds)),
                            "avg_final_gap_to_price":float(np.mean(gaps))
                        })

    df=pd.DataFrame(rows)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"price_compression_grid.csv",index=False)

    # Keep only nontrivial samples and rank several ways.
    valid=df[df.trades>=100].copy()
    valid.sort_values(["PF","net_usd_0p01lot_equiv"],ascending=[False,False]).head(30).to_csv(out/"top30_by_pf.csv",index=False)
    valid.sort_values(["recovery_rate_pct","PF"],ascending=[False,False]).head(30).to_csv(out/"top30_by_recovery.csv",index=False)
    valid.sort_values(["avg_final_gap_to_price","PF"],ascending=[True,False]).head(30).to_csv(out/"top30_by_price_compression.csv",index=False)

    summary={
        "experiment":"AMOS_VirtualBinary_PriceCompression_v10",
        "raw_ticks":len(ticks),"seed_rules":rules,
        "seed_counts":{k:int(len(v)) for k,v in seeds.items()},
        "grid":{"compression_remaining_gap":COMPRESSION,"max_layers":MAX_LAYERS,
                "max_add_multiple_current_weight":MAX_ADD_MULT,"basket_targets":BASKET_TARGETS,
                "max_hold_sec":MAX_HOLD_SEC,"base_lot":BASE_LOT},
        "rule":"Every 5 seconds, if an independent basket is negative, calculate same-direction add size to compress its weighted average entry toward current executable price, subject to layer and lot caps.",
        "best_by_pf":valid.sort_values(["PF","net_usd_0p01lot_equiv"],ascending=[False,False]).iloc[0].to_dict() if len(valid) else None,
        "best_by_compression":valid.sort_values(["avg_final_gap_to_price","PF"],ascending=[True,False]).iloc[0].to_dict() if len(valid) else None,
        "risk_note":"This test reports per-basket floating loss and exposure. Aggregate simultaneous-account margin/DD must be tested before considering deployment.",
        "dataset_manifest":man,
        "nautilus_version":getattr(nautilus_trader,"__version__","unknown")
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
