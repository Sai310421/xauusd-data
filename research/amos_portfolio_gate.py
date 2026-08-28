from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = (
    "engine_contribution",
    "cross_engine_dependence",
    "aggregate_exposure",
    "event_budget",
    "margin_stress",
    "concentration",
)


def evaluate(payload: dict) -> dict:
    missing = [k for k in FIELDS if payload.get(k) is None]
    violations = payload.get("violations") if isinstance(payload.get("violations"), list) else []
    status = "INVALID" if missing else ("REJECT" if violations else "PASS")
    return {"schema":"amos-portfolio-gate-v1","status":status,"missing":missing,"violations":violations}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = evaluate(json.loads(args.input.read_text(encoding="utf-8")))
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
