"""Structural paired-shock study. The label forbids reading it as XAUUSD."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from adapters.monte_carlo.structural import run_overlay  # noqa: E402


def main() -> int:
    summary = run_overlay()
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
