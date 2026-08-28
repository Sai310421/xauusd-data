from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("wfo","monte_carlo","parameter_stability","regime_segmentation","master_checker")


def evaluate(payload: dict) -> dict:
    missing=[k for k in FIELDS if payload.get(k) is None]
    failed=[k for k in FIELDS if isinstance(payload.get(k),dict) and payload[k].get("status") not in ("PASS",)]
    status="INVALID" if missing else ("REJECT" if failed else "PASS")
    return {"schema":"amos-robustness-gate-v1","status":status,"missing":missing,"failed":failed}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("input",type=Path); ap.add_argument("--output",type=Path); a=ap.parse_args()
    r=evaluate(json.loads(a.input.read_text(encoding="utf-8"))); t=json.dumps(r,indent=2,sort_keys=True); print(t)
    if a.output: a.output.write_text(t+"\n",encoding="utf-8")
    return 0 if r["status"]=="PASS" else 2

if __name__=="__main__": raise SystemExit(main())
