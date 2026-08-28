from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("return", "pf", "drawdown", "slippage", "fill_rate", "exposure", "unresolved_inventory", "basket_age")


def rel_error(expected, observed):
    try:
        e = float(expected)
        o = float(observed)
    except (TypeError, ValueError):
        return None
    denom = abs(e) if abs(e) > 1e-12 else 1.0
    return abs(o - e) / denom


def evaluate(payload: dict, tolerance: float) -> dict:
    expected = payload.get("expected") if isinstance(payload.get("expected"), dict) else {}
    observed = payload.get("observed") if isinstance(payload.get("observed"), dict) else {}
    missing = [f for f in FIELDS if expected.get(f) is None or observed.get(f) is None]
    errors = {f: rel_error(expected.get(f), observed.get(f)) for f in FIELDS if f not in missing}
    breaches = {k: v for k, v in errors.items() if v is not None and v > tolerance}
    status = "INVALID" if missing else ("DEMOTE" if breaches else "PASS")
    return {"schema":"amos-live-parity-v1","status":status,"tolerance":tolerance,"missing":missing,"relative_errors":errors,"breaches":breaches}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.25)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = evaluate(json.loads(args.input.read_text(encoding="utf-8")), args.tolerance)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
