#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np, pandas as pd

def streak_loss(p):
    best=cur=0
    for x in p:
        cur=cur+1 if x<0 else 0
        best=max(best,cur)
    return best

def diag(tr, capital=1000.0, seed=75):
    if tr.empty: return {"N":0,"status":"INSUFFICIENT"}
    p=tr["pnl"].astype(float).to_numpy()
    n=len(p); eq=capital+np.cumsum(p); peak=np.maximum.accumulate(np.r_[capital,eq])[:-1]
    dd=(peak-eq)/np.maximum(peak,1e-12)
    q05=float(np.quantile(p,.05)); tail=p[p<=q05]
    rng=np.random.default_rng(seed)
    boot=np.array([rng.choice(p,n,replace=True).mean() for _ in range(5000)])
    absdev=np.abs(p-np.mean(p)); transport=float(np.quantile(absdev,.75))
    daily=tr.assign(day=pd.to_datetime(tr.exit_time,utc=True).dt.date).groupby("day").pnl.sum()
    return {
      "N":n,
      "max_loss_streak":streak_loss(p),
      "worst_trade_USD":float(p.min()),
      "VaR95_trade_USD":q05,
      "CVaR95_trade_USD":float(tail.mean()),
      "realized_equity_MaxDD_pct":float(dd.max()*100),
      "first_passage_DD_3_5":bool((dd>=.035).any()),
      "first_passage_DD_5":bool((dd>=.05).any()),
      "first_passage_DD_10":bool((dd>=.10).any()),
      "bootstrap_EV_USD":float(boot.mean()),
      "bootstrap_EV_CI95_USD":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
      "bootstrap_P_EV_gt_0":float((boot>0).mean()),
      "daily_downside_dev_USD":float(np.sqrt(np.mean(np.minimum(daily.to_numpy(),0.0)**2))),
      "wasserstein_stress_proxy":{"radius_0_10_EV":float(p.mean()-.10*transport),"radius_0_25_EV":float(p.mean()-.25*transport),"radius_0_50_EV":float(p.mean()-.50*transport)},
      "note":"Wasserstein values are empirical transport-stress proxies, not a formal DRO certificate."
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dir",default="results/amos_gold_5in1_v101"); ap.add_argument("--capital",type=float,default=1000); a=ap.parse_args()
    d=Path(a.dir)
    try: tr=pd.read_csv(d/"trades.csv") if (d/"trades.csv").exists() else pd.DataFrame()
    except pd.errors.EmptyDataError: tr=pd.DataFrame()
    out=diag(tr,a.capital)
    if (d/"summary.json").exists():
        s=json.loads((d/"summary.json").read_text()); out["engine_scope"]=s.get("engine_scope","ALL"); out["rawtick_MaxDD_pct"]=s.get("MaxDD_pct")
    (d/"math_edge_diagnostic.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
