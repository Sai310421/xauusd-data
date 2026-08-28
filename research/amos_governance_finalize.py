from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.amos_reality_grade import evaluate as grade_evaluate


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def finalize(root: Path) -> dict:
    evidence = load(root / "amos_evidence.json") or {}
    evidence_gate = load(root / "amos_evidence_gate.json") or {"promotion_eligible": False, "status": "INVALID"}
    robustness = load(root / "amos_robustness_gate.json") or {"status": "INVALID"}
    live = load(root / "amos_live_parity_gate.json") or {"status": "INVALID"}
    portfolio = load(root / "amos_portfolio_gate.json") or {"status": "INVALID"}
    metrics = evidence.get("metrics") if isinstance(evidence.get("metrics"), dict) else {}

    grade_input = {
        "evidence_gate": evidence_gate,
        "robustness": robustness,
        "live_parity": live,
        "portfolio_gate": portfolio,
        "final_metrics": metrics,
    }
    grade = grade_evaluate(grade_input)

    statuses = {
        "evidence_wr5": "PASS" if evidence_gate.get("promotion_eligible") is True else "INVALID",
        "robustness": robustness.get("status", "INVALID"),
        "live_parity": live.get("status", "INVALID"),
        "portfolio": portfolio.get("status", "INVALID"),
    }
    if "DEMOTE" in statuses.values():
        decision = "DEMOTE"
    elif "REJECT" in statuses.values():
        decision = "REJECT"
    elif all(v in ("PASS", "NOT_APPLICABLE") for v in statuses.values()) and grade.get("grade") != "INVALID":
        decision = "ELIGIBLE_FOR_PROMOTION_REVIEW"
    else:
        decision = "INVALID"

    return {
        "schema": "amos-governance-final-v1",
        "experiment_id": evidence.get("experiment_id"),
        "git_sha": evidence.get("git_sha"),
        "lane_class": evidence.get("lane_class"),
        "statuses": statuses,
        "reality_grade": grade,
        "decision": decision,
        "rule": "Missing mandatory evidence remains INVALID; no synthetic PASS is permitted.",
    }


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); ap.add_argument("--output",type=Path); a=ap.parse_args()
    r=finalize(a.root); t=json.dumps(r,indent=2,sort_keys=True); print(t)
    out=a.output or (a.root/"amos_governance_final.json"); out.write_text(t+"\n",encoding="utf-8")
    return 0 if r["decision"]=="ELIGIBLE_FOR_PROMOTION_REVIEW" else 2

if __name__=="__main__": raise SystemExit(main())
