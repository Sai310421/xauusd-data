#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    docs=[]
    for p in root.rglob("result.json"):
        try: docs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception: pass
    real=[d for d in docs if d.get("lane_meta",{}).get("lane")=="real_raw"]
    rnd=[d for d in docs if d.get("lane_meta",{}).get("lane")=="random_stress"]
    if not real: raise SystemExit("real lane result missing")
    R=real[0]
    rows=[]
    for name,rv in R["candidates"].items():
        rm=rv["CLASSIC_A_SELL"]
        pfs=[];wrs=[];ns=[];zero=0
        for d in rnd:
            c=d.get("candidates",{}).get(name)
            if not c: continue
            m=c["CLASSIC_A_SELL"]; n=int(m.get("N") or 0)
            ns.append(n); wrs.append(float(m.get("WR_pct") or 0.0)); pfs.append(float(m.get("PF") or 0.0))
            if n==0: zero+=1
        stress_runs=len(pfs)
        pf_med=statistics.median(pfs) if pfs else None
        pf_min=min(pfs) if pfs else None
        wr_med=statistics.median(wrs) if wrs else None
        n_med=statistics.median(ns) if ns else None
        # Stress is diagnostic only. Flag collapse; do not override real gate.
        fragility="UNKNOWN"
        if pfs:
            fragility="RED" if (zero/stress_runs>=0.5 or pf_med<0.35) else ("AMBER" if pf_med<0.60 else "GREEN")
        rows.append({
          "candidate":name,
          "real_N":int(rm.get("N") or 0),
          "real_WR_pct":float(rm.get("WR_pct") or 0.0),
          "real_PF":float(rm.get("PF") or 0.0),
          "real_Net_USD":float(rm.get("Net_USD") or 0.0),
          "inverse_score":float(rv.get("inverse_score") or 999),
          "passes_real_gate":bool(rv.get("passes_real_gate")),
          "random_runs":stress_runs,
          "random_PF_median":pf_med,"random_PF_min":pf_min,
          "random_WR_median":wr_med,"random_N_median":n_med,
          "fragility":fragility
        })
    rows.sort(key=lambda x:(not x["passes_real_gate"],x["inverse_score"],-x["real_PF"]))
    summary={
      "version":"v1.29",
      "architecture":"MIDDLE_INTEGRATION_SHARED_CORE_PARALLEL_LANES_REMERGE",
      "promotion_rule":"Real Raw Tick gate is authoritative. RandomChart is stress/fragility only.",
      "target_edge":{"N_min":120,"WR_min_pct":45.0,"PF_min":1.20},
      "real_lane_count":len(real),"random_lane_count":len(rnd),
      "passed_real_gate":[x["candidate"] for x in rows if x["passes_real_gate"]],
      "top10":rows[:10],"all_candidates":rows
    }
    (out/"scoreboard.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    import csv
    with open(out/"scoreboard.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
