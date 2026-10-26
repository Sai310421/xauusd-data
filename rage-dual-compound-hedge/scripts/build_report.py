"""Write the evidence JSON. Hypothesis fields are not upgraded to PASS."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from adapters.monte_carlo.structural import run_overlay, run_v1_intensity_overlay  # noqa: E402
from rage_dch.reporting import empty_market_report  # noqa: E402


def main() -> int:
    report = empty_market_report()
    report["structural_overlay"] = asdict(run_overlay())
    report["structural_v1_intensity"] = asdict(run_v1_intensity_overlay())
    report["structural_overlay_reading"] = (
        "Perfect beta=1 overlay with zero main alpha: large monthly losses are offset and "
        "the hedge book still compounds below zero because of cost. "
        "v1 intensity is a much smaller ratio, so the same shocks leave a protection shortfall. "
        "Neither study is a market measurement."
    )
    out = ROOT / "evidence" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(out)
    forbidden = report["hypothesis"]["HEDGE_10_PERCENT_COMPOUND"]
    if forbidden == "PASS":
        raise SystemExit("refusing to write a PASS on an unmeasured hedge book")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
