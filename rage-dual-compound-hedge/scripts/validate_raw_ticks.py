"""Validate a CSV of ts,bid,ask. Missing file is BLOCKED, not a synthetic pass."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from adapters.nautilus.catalog_gate import QuoteRow, validate_quote_ticks  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_raw_ticks.py quotes.csv", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(json.dumps({"status": "BLOCKED", "reason": "catalog file missing", "path": str(path)}))
        return 2
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"ts", "bid", "ask"} <= fields:
            print(json.dumps({"status": "FAIL", "reason": "ohlc or incomplete header rejected", "fields": sorted(fields)}))
            return 1
        for record in reader:
            rows.append(QuoteRow(int(record["ts"]), float(record["bid"]), float(record["ask"])))
    verdict = validate_quote_ticks(rows)
    print(json.dumps({"status": verdict.status, "count": verdict.count, "reasons": verdict.reasons}))
    return 0 if verdict.status == "RAW_BIDASK_CATALOG_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
