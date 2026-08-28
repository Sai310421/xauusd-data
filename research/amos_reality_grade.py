from __future__ import annotations

import argparse
import json
from pathlib import Path

FINAL_KEYS = ("monthly_trading_return","monthly_cashback_return","combined_monthly_return","max_dd","pf","ruin_probability","rescue_rate","zr_dependency","be_or_better_exit_rate")


def evaluate(evidence: dict) -> dict:
    gate = evidence.get("evidence_gate") or {}
    robustness = evidence.get("robustness") or {}
    live = evidence.get("live_parity") or {}
    portfolio = evidence.get("portfolio_gate") or {}
    metrics = evidence.get("final_metrics") or {}

    missing = [k for k in FINAL_KEYS if metrics.get(k) is None]
    prerequisites = {
        "evidence_gate": gate.get("promotion_eligible") is True,
        "robustness": robustness.get("status") == "PASS",
        "live_parity": live.get("status") == "PASS",
        "portfolio": portfolio.get("status") in ("PASS", "NOT_APPLICABLE"),
    }
    if missing or not all(prerequisites.values()):
        return {"schema":"amos-reality-grade-v1","grade":"INVALID","missing_metrics":missing,"prerequisites":prerequisites}

    m = {k: float(metrics[k]) for k in FINAL_KEYS}
    passes = (
        m["monthly_trading_return"] >= 40
        and m["monthly_cashback_return"] >= 10
        and m["combined_monthly_return"] >= 50
        and m["max_dd"] <= 15
        and m["pf"] >= 1.8
        and m["ruin_probability"] < 0.01
        and m["rescue_rate"] <= 0.05
        and 0.02 <= m["zr_dependency"] <= 0.05
        and m["be_or_better_exit_rate"] >= 0.90
    )
    if not passes:
        grade = "C"
    elif m["max_dd"] <= 10 and m["pf"] >= 2.5 and m["combined_monthly_return"] >= 70:
        grade = "A+"
    elif m["max_dd"] <= 10 and m["combined_monthly_return"] >= 50:
        grade = "A"
    else:
        grade = "B"
    return {"schema":"amos-reality-grade-v1","grade":grade,"missing_metrics":[],"prerequisites":prerequisites,"final_constraints_pass":passes}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("input", type=Path); ap.add_argument("--output", type=Path); a=ap.parse_args()
    r=evaluate(json.loads(a.input.read_text(encoding="utf-8"))); t=json.dumps(r,indent=2,sort_keys=True); print(t)
    if a.output: a.output.write_text(t+"\n",encoding="utf-8")
    return 0 if r["grade"] != "INVALID" else 2

if __name__ == "__main__": raise SystemExit(main())
