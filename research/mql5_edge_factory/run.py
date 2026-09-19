from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .classify import classify
from .metrics import matched_delta, metrics


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def floats(rows, key):
    out=[]
    for r in rows:
        v=r.get(key)
        if v not in (None, ""):
            out.append(float(v))
    return out


def main():
    ap=argparse.ArgumentParser(description="MQL5 observable EDGE numeric validator")
    ap.add_argument("--trades", required=True, help="Normalized trade CSV")
    ap.add_argument("--control", help="Optional matched-control trade CSV")
    ap.add_argument("--oos-positive", choices=["true","false","unknown"], default="unknown")
    ap.add_argument("--out", default="edge_result.json")
    args=ap.parse_args()

    rows=load_csv(args.trades)
    m=metrics(floats(rows,"pnl"), floats(rows,"mfe") or None, floats(rows,"mae") or None)
    delta=None
    comparison=None
    if args.control:
        crows=load_csv(args.control)
        comparison=matched_delta(floats(rows,"pnl"), floats(crows,"pnl"))
        delta=comparison["delta_expectancy"]
    oos={"true":True,"false":False,"unknown":None}[args.oos_positive]
    verdict=classify(m.n,m.expectancy,m.pf,m.max_dd,delta,oos)
    result={"metrics":m.to_dict(),"matched_control":comparison,"verdict":verdict,
            "note":"library_ready is intentionally false; promotion requires a separate explicit Raw Tick/OOS gate."}
    Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=True),encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=True))

if __name__ == "__main__":
    main()
