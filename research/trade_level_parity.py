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
    return df.sort_values(["symbol", "side", "entry_time"]).reset_index(drop=True)


def _within_time(a, b, tol_ms: float) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    dt = abs((pd.Timestamp(a) - pd.Timestamp(b)).total_seconds() * 1000.0)
    return dt <= tol_ms


def compare(mt5: pd.DataFrame, nt: pd.DataFrame, *, max_mismatch_pct: float, time_tol_ms: float, sample_limit: int = 50) -> dict:
    used = set()
    field_mismatches = 0
    details = []

    for i, a in mt5.iterrows():
        best = None
        best_dt = None
        for j, b in nt.iterrows():
            if j in used:
                continue
            if str(a["symbol"]) != str(b["symbol"]) or str(a["side"]).upper() != str(b["side"]).upper():
                continue
            if not _within_time(a["entry_time"], b["entry_time"], time_tol_ms):
                continue
            if not _within_time(a["exit_time"], b["exit_time"], time_tol_ms):
                continue
            dt = abs((pd.Timestamp(a["entry_time"]) - pd.Timestamp(b["entry_time"])).total_seconds())
            if best is None or dt < best_dt:
                best, best_dt = j, dt
        if best is None:
            field_mismatches += 1
            if len(details) < sample_limit:
                details.append({"mt5_index": int(i), "reason": "unmatched_mt5"})
            continue
        used.add(best)
        b = nt.loc[best]
        row_diff = {}
        for c in NUMERIC:
            av, bv = float(a[c]), float(b[c])
            tol = max(1e-9, abs(av) * 1e-6)
            if abs(av - bv) > tol:
                row_diff[c] = [av, bv]
        if row_diff:
            field_mismatches += 1
            if len(details) < sample_limit:
                details.append({"mt5_index": int(i), "nautilus_index": int(best), "diff": row_diff})

    unmatched_nt = len(nt) - len(used)
    mismatch_rows = field_mismatches + unmatched_nt
    n = max(len(mt5), len(nt), 1)
    mismatch_pct = 100.0 * mismatch_rows / n
    status = "PASS" if mismatch_pct <= max_mismatch_pct else "FAIL"
    return {
        "status": status,
        "mt5_trades": int(len(mt5)),
        "nautilus_trades": int(len(nt)),
        "count_delta": int(abs(len(mt5) - len(nt))),
        "mismatch_rows": int(mismatch_rows),
        "mismatch_pct": float(mismatch_pct),
        "threshold_pct": float(max_mismatch_pct),
        "time_tolerance_ms": float(time_tol_ms),
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--nautilus", required=True)
    ap.add_argument("--out", default="results/parity/trade_level_parity.json")
    ap.add_argument("--max-mismatch-pct", type=float, default=3.0)
    ap.add_argument("--time-tolerance-ms", type=float, default=1000.0)
    args = ap.parse_args()

    result = compare(
        load(Path(args.mt5)),
        load(Path(args.nautilus)),
        max_mismatch_pct=args.max_mismatch_pct,
        time_tol_ms=args.time_tolerance_ms,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
