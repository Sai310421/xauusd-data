from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

KEYS = ["symbol", "side", "entry_time", "exit_time"]
NUMERIC = ["entry_price", "exit_price", "qty", "pnl"]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"BLOCKED_MISSING_INPUT: {path}")
    df = pd.read_csv(path)
    missing = [c for c in KEYS + NUMERIC if c not in df.columns]
    if missing:
        raise SystemExit(f"BLOCKED_SCHEMA_MISMATCH {path}: {missing}")
    for c in ["entry_time", "exit_time"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df.sort_values(KEYS).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--nautilus", required=True)
    ap.add_argument("--out", default="results/parity/trade_level_parity.json")
    ap.add_argument("--max-mismatch-pct", type=float, default=3.0)
    args = ap.parse_args()

    mt5 = load(Path(args.mt5))
    nt = load(Path(args.nautilus))
    n = max(len(mt5), len(nt), 1)
    count_delta = abs(len(mt5) - len(nt))
    matched = min(len(mt5), len(nt))
    field_mismatches = 0
    details = []

    for i in range(matched):
        row_diff = {}
        for c in KEYS:
            a, b = mt5.iloc[i][c], nt.iloc[i][c]
            if str(a) != str(b):
                row_diff[c] = [str(a), str(b)]
        for c in NUMERIC:
            a, b = float(mt5.iloc[i][c]), float(nt.iloc[i][c])
            tol = max(1e-9, abs(a) * 1e-6)
            if abs(a - b) > tol:
                row_diff[c] = [a, b]
        if row_diff:
            field_mismatches += 1
            if len(details) < 50:
                details.append({"index": i, "diff": row_diff})

    mismatch_rows = count_delta + field_mismatches
    mismatch_pct = 100.0 * mismatch_rows / n
    status = "PASS" if mismatch_pct <= args.max_mismatch_pct else "FAIL"
    result = {
        "status": status,
        "mt5_trades": len(mt5),
        "nautilus_trades": len(nt),
        "count_delta": count_delta,
        "mismatch_rows": mismatch_rows,
        "mismatch_pct": mismatch_pct,
        "threshold_pct": args.max_mismatch_pct,
        "details": details,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
