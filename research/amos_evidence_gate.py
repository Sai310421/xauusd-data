from __future__ import annotations

import argparse
import json
from pathlib import Path

WR5_FIELDS = (
    "spread",
    "round_trip_commission",
    "slippage",
    "execution_delay",
    "swap",
    "cashback_assumptions",
    "mark_to_market_equity",
    "floating_drawdown",
    "current_open_position_state",
    "aggregate_exposure",
    "margin_level",
    "basket_age",
    "unresolved_inventory",
    "event_price_pitch_budget",
)

EVIDENCE_FIELDS = (
    "experiment_id",
    "run_id",
    "git_sha",
    "lane_class",
    "runner",
    "dataset_or_catalog_manifest",
    "configuration",
    "test_period",
    "sizing",
    "metrics",
    "wr5",
)


def _missing(mapping: dict, fields: tuple[str, ...]) -> list[str]:
    return [k for k in fields if k not in mapping or mapping[k] is None]


def evaluate(payload: dict) -> dict:
    missing_evidence = _missing(payload, EVIDENCE_FIELDS)
    wr5 = payload.get("wr5") if isinstance(payload.get("wr5"), dict) else {}
    missing_wr5 = _missing(wr5, WR5_FIELDS)

    result = {
        "schema": "amos-evidence-gate-v1",
        "evidence_status": "VALID" if not missing_evidence else "INVALID",
        "wr5_status": "VALID_FOR_DECISION" if not missing_wr5 else "INVALID",
        "missing_evidence": missing_evidence,
        "missing_wr5": missing_wr5,
        "promotion_eligible": not missing_evidence and not missing_wr5,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    result = evaluate(payload)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["promotion_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
