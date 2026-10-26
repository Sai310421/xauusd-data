"""Market backtest entry. Refuses to invent ticks or native fills."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from adapters.nautilus.fills import native_fill_count  # noqa: E402


def main() -> int:
    catalog = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = {
        "status": "BLOCKED",
        "reason": "no raw bid/ask catalog was passed; OHLC stand-in is forbidden",
        "catalog": str(catalog) if catalog else None,
        "nautilus_native_fills": native_fill_count(),
        "candidates_not_executed": [
            "research/ae_allweather_nautilus_raw_ab_v9_drmpc.py",
            "research/armada_nautilus_raw_cell.py",
            "research/bigplayer_rawtick_5tf_21d_bt.py",
        ],
    }
    if catalog is None or not catalog.exists():
        print(json.dumps(payload))
        return 2
    payload["reason"] = "catalog path exists but NautilusTrader is not attached in this package"
    print(json.dumps(payload))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
