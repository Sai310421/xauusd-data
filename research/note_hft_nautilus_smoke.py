from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    import nautilus_trader

    data = Path("csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv")
    report = {
        "engine": "NautilusTrader",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "data_path": str(data),
        "data_exists": data.exists(),
        "data_bytes": data.stat().st_size if data.exists() else 0,
        "status": "SMOKE_ONLY",
        "note_hft_bt": False,
        "reason": "Public runner validates Nautilus + dataset only. NOTE-HFT Frozen adapter remains private and is not substituted.",
    }

    if data.exists():
        with data.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            first = next(reader, [])
            second = next(reader, [])
        report["csv_header"] = first
        report["csv_first_row"] = second

    out = Path("results/note-hft")
    out.mkdir(parents=True, exist_ok=True)
    (out / "nautilus_public_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
