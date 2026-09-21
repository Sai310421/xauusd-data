from __future__ import annotations
"""Frozen-candidate confirmation on an unseen Raw Bid/Ask period.

Candidate is frozen from discovery run 35603417081. This script never changes
the candidate based on OOS results. Neighbor lanes are robustness diagnostics.
"""
import argparse,json
from pathlib import Path
from statistics import median
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.nautilus_catalog_compat import query_quote_ticks_compat
from research.video_procedure_edge_search_v1 import bars,trades_for_lane,metrics,monte_carlo

SCHEMA="AMOS.VideoProcedureEdgeConfirmation.v1.1"
FROZEN={"tf":"M15","edge":"T1_EMA21","direction":1,"threshold":.5,"tp_atr":1.5,"sl_atr":1.0,"horizon":4,
        "discovery_run_id":35603417081,"discovery_N":245,"discovery_PF":1.2747350094799224,"discovery_EV_R":0.09086505499711309}

def stressed_bars(B,mult):
    return [(ts,o,h,l,c,s*mult) for ts,o,h,l,c,s in B]

def eval_lane(B,threshold,tp,sl,spread_mult=1.0):
    rs=trades_for_lane(stressed_bars(B,spread_mult),FROZEN["edge"],FROZEN["direction"],threshold,tp,sl,FROZEN["horizon"])
    return {**metrics(rs),"spread_mult":spread_mult,"monte_carlo":[monte_carlo(rs,r,seed=20260922+int(100*spread_mult)) for r in (.5,1.,2.)]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--experiment-id",required=True);a=ap.parse_args()
    cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    raw=query_quote_ticks_compat(cat,identifiers=[inst.id.value]);B=bars(raw,15)
    base=eval_lane(B,.5,1.5,1.0,1.0)
    stress={str(x):eval_lane(B,.5,1.5,1.0,x) for x in (1.5,2.0)}
    neighbors=[]
    for th in (.4,.5,.6):
      for tp in (1.25,1.5,1.75):
       for sl in (.75,1.0,1.25):
        m=eval_lane(B,th,tp,sl,1.0);neighbors.append({"threshold":th,"tp_atr":tp,"sl_atr":sl,**{k:m[k] for k in ("N","WR","PF","EV_R","median_R","no_loss_share")}})
    positive=sum(x["PF"]>=1.0 and x["EV_R"]>0 for x in neighbors)
    robust=sum(x["PF"]>=1.2 and x["EV_R"]>0 and x["N"]>=70 for x in neighbors)
    decision="PASS" if base["N"]>=100 and base["PF"]>=1.2 and base["EV_R"]>0 and stress["1.5"]["PF"]>=1.0 and stress["1.5"]["EV_R"]>0 else "FAIL"
    out={"schema":SCHEMA,"verification_level":"UNSEEN_PERIOD_RAW_BIDASK_FROZEN_CANDIDATE","candidate":FROZEN,"raw_ticks":len(raw),
         "oos_base":base,"cost_stress":stress,"neighbor_robustness":{"lanes":len(neighbors),"positive_lanes":positive,"pf12_positive_n70_lanes":robust,
         "median_neighbor_pf":median(x["PF"] for x in neighbors),"median_neighbor_ev_R":median(x["EV_R"] for x in neighbors)},
         "neighbors":neighbors,"confirmation_gate":decision,"selection_policy":"FROZEN_BEFORE_OOS_NO_RETUNING_FROM_OOS",
         "production_weighting_allowed":False,"execution_allowed":False}
    p=Path("results/video-edge-confirmation")/a.experiment_id;p.mkdir(parents=True,exist_ok=True);(p/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
