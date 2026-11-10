"""Phase 1 contract validation for RAGE H1 macro signals."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "rage-dual-compound-hedge/contracts/h1_macro_signal_v1.schema.json"


def ready_signal() -> dict[str, object]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return {
        "schema_version": "h1-macro-signal/v1",
        "signal_id": "usd-risk-20261110T000000Z",
        "as_of_utc": now.isoformat().replace("+00:00", "Z"),
        "valid_until_utc": (now + timedelta(minutes=90)).isoformat().replace("+00:00", "Z"),
        "signal_status": "READY",
        "stop_reason": None,
        "model_version": "shadow-v001",
        "portfolio_risk": 77.0,
        "portfolio_regime": "HIGH",
        "gold_prob_up2": 0.74,
        "stock_crash_prob": 0.68,
        "smallcap_stress": 0.82,
        "breadth_div": -2.47,
        "psi_status": "NORMAL",
        "probability_ordering_pass": True,
        "feature_schema_pass": True,
        "manifest_integrity_pass": True,
        "source_hash": "a" * 64,
        "top_drivers": ["flow_vx", "smallcap_stress", "breadth_div"],
    }


def errors(value: dict[str, object]) -> list[object]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return list(validator.iter_errors(value))


def main() -> None:
    assert not errors(ready_signal()), "READY fixture must conform"

    stopped = copy.deepcopy(ready_signal())
    stopped["signal_status"] = "STOPPED"
    stopped["stop_reason"] = "MODEL_DRIFT_CRITICAL"
    assert not errors(stopped), "STOPPED fixture must conform"

    missing_gate = copy.deepcopy(ready_signal())
    missing_gate["probability_ordering_pass"] = False
    assert errors(missing_gate), "failing ordering gate must be rejected"

    out_of_range = copy.deepcopy(ready_signal())
    out_of_range["portfolio_risk"] = 101
    assert errors(out_of_range), "out-of-range portfolio risk must be rejected"

    bad_ready = copy.deepcopy(ready_signal())
    bad_ready["stop_reason"] = "STALE_DATA"
    assert errors(bad_ready), "READY signal may not include stop reason"


if __name__ == "__main__":
    main()
