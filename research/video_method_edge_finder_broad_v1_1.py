#!/usr/bin/env python3
"""Broad numerical EDGE sweep v1.1 derived from the uploaded video's workflow.

Source video procedure is preserved: sweep -> MC lane metrics -> filter -> EV -> risk.
AMOS extension: 4 transparent numeric families x 4 thresholds, 60/20/20
discovery/validation/final-OOS. No chart-pattern hindsight and no OOS-only selection.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from statistics import mean
import numpy as np
import nautilus_trader
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.nautilus_catalog_compat import query_quote_ticks_compat
from research.video_method_edge_finder_v1 import _f,_q,_bars,_bootstrap_metrics,_risk_mc

SCHEMA="AMOS.VideoMethodEdgeFinder.BroadNumeric.v1.1"
P_QUANTILES={0:.50,1:.65,2:.80,3:.90}
FAMILIES=("IMPULSE_CONT","IMPULSE_REV","STREAK3_CONT","MEAN_REVERT20")

def _rolling3(rets):
    out=[None]*len(rets)
    for i in range(2,len(rets)): out[i]=rets[i]+rets[i-1]+rets[i-2]
    return out

def _z20(prices):
    out=[None]*len(prices)
    for i in range(19,len(prices)):
        w=np.asarray(prices[i-19:i+1],dtype=float); sd=float(w.std())
        out[i]=0.0 if sd<=1e-15 else float((prices[i]-w.mean())/sd)
    return out

def _feature(prices,rets,family):
    if family in ("IMPULSE_CONT","IMPULSE_REV"): return list(rets)
    if family=="STREAK3_CONT": return _rolling3(rets)
    if family=="MEAN_REVERT20": return _z20(prices)
    raise ValueError(family)

def _threshold(train_feature,q):
    vals=[abs(x) for x in train_feature if x is not None and math.isfinite(x)]
    if len(vals)<30: raise ValueError("insufficient feature")
    return _q(vals,q)

def _trades_family(rets,feature,threshold,direction,family,hold=3):
    sign=1. if direction=="Long" else -1.; out=[]
    for i in range(1,len(rets)-hold):
        x=feature[i-1]
        if x is None: continue
        if family in ("IMPULSE_CONT","STREAK3_CONT"): ok=sign*x>=threshold
        elif family=="IMPULSE_REV": ok=-sign*x>=threshold
        elif family=="MEAN_REVERT20": ok=-sign*x>=threshold
        else: raise ValueError(family)
        if not ok: continue
        f=1.
        for j in range(i,i+hold): f*=1+rets[j]
        out.append(sign*(f-1))
    return out

def _stage_ok(m,strict):
    if not m: return False
    if strict: return m["MedRF"]>=1.5 and m["NoLossPct"]<=3.0 and m["EV"]>0 and m["Trades"]>=80
    return m["MedRF"]>=1.0 and m["NoLossPct"]<=10.0 and m["EV"]>0 and m["Trades"]>=30

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--catalog",required=True); ap.add_argument("--symbols",nargs="+",required=True)
    ap.add_argument("--experiment-id",required=True); ap.add_argument("--paths",type=int,default=2000); ap.add_argument("--path-trades",type=int,default=400)
    ap.add_argument("--raw-bidask-only",action="store_true"); args=ap.parse_args()
    if not args.raw_bidask_only: raise SystemExit("RAW_BIDASK_ONLY_REQUIRED")
    cat=ParquetDataCatalog(args.catalog); inst={x.id.symbol.value.replace("/",""):x for x in cat.instruments()}
    rows=[]; final_trade_map={}; raw_counts={}
    for symbol in args.symbols:
        ticks=query_quote_ticks_compat(cat,identifiers=[inst[symbol].id.value]); raw_counts[symbol]=len(ticks)
        _,prices,rets,_=_bars(ticks,5); n=len(rets); a=int(n*.60); b=int(n*.80)
        # prices from _bars align one-to-one with rets.
        for family in FAMILIES:
            feat=_feature(prices,rets,family)
            for p,q in P_QUANTILES.items():
                th=_threshold(feat[:a],q)
                for direction in ("Long","Short"):
                    tr=_trades_family(rets[:a],feat[:a],th,direction,family)
                    va=_trades_family(rets[a:b],feat[a:b],th,direction,family)
                    oo=_trades_family(rets[b:],feat[b:],th,direction,family)
                    mtr=_bootstrap_metrics(tr,args.paths,args.path_trades,seed=11+p)
                    mva=_bootstrap_metrics(va,args.paths,args.path_trades,seed=111+p)
                    moo=_bootstrap_metrics(oo,args.paths,args.path_trades,seed=1111+p)
                    disc=_stage_ok(mtr,True); val=bool(disc and _stage_ok(mva,False)); final=bool(val and _stage_ok(moo,False))
                    row={"Pair":symbol,"Family":family,"Dir":direction,"P":p,"Quantile":q,"Threshold":th,
                         "DiscoveryCandidate":disc,"ValidationPassed":val,"FinalOOSConfirmed":final}
                    for prefix,m in (("IS",mtr),("VAL",mva),("OOS",moo)):
                        for k in ("MedProfit","P90Profit","MedRF","NoLossPct","WinRatePct","AvgWin","AvgLoss","EV","Trades"):
                            row[f"{prefix}_{k}"]=None if not m else m[k]
                    rows.append(row)
                    if final: final_trade_map[(symbol,family,direction,p)]=oo
    rows.sort(key=lambda r:(not r["FinalOOSConfirmed"],not r["ValidationPassed"],not r["DiscoveryCandidate"],-(r["OOS_EV"] or -1e9)))
    final=[r for r in rows if r["FinalOOSConfirmed"]]; validation=[r for r in rows if r["ValidationPassed"]]; discovery=[r for r in rows if r["DiscoveryCandidate"]]
    risk=[]
    for r in final[:8]:
        tr=final_trade_map[(r["Pair"],r["Family"],r["Dir"],r["P"])]
        risk.append({"Pair":r["Pair"],"Family":r["Family"],"Dir":r["Dir"],"P":r["P"],
          "mc":[_risk_mc(tr,x,args.paths,args.path_trades,seed=11+int(x*1000)) for x in (.005,.01,.02,.05,.10)]})
    out=Path("results/edge-finder")/args.experiment_id; out.mkdir(parents=True,exist_ok=True)
    with (out/"sweep.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"schema":SCHEMA,"source_video_sweep_label":"SYNTHETIC","source_hidden_P_definition_known":False,
      "amos_extension":{"families":FAMILIES,"P_quantiles":P_QUANTILES,"timeframe":"M5","hold_bars":3},
      "lane_count":len(rows),"discovery_candidates":len(discovery),"validation_passed":len(validation),"final_oos_confirmed":len(final),
      "top_final_oos":final[:12],"risk_mc":risk,"raw_tick_counts":raw_counts,
      "selection":{"discovery":"first 60%","validation":"next 20%","final_oos":"last 20%"},
      "filters":{"discovery":"MedRF>=1.5, NoLoss<=3%, EV>0, N>=80","validation_final":"MedRF>=1.0, NoLoss<=10%, EV>0, N>=30",
        "video_MedProfit_300_not_applied":"unknown source sizing/contract basis"},
      "monte_carlo":{"paths":args.paths,"trades_per_path":args.path_trades},
      "execution_allowed":False,"production_weighting_allowed":False}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
