"""Sanity-check an OHLCV CSV.

Usage:
    python scripts/validate.py csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv
"""
import sys
from pathlib import Path

import pandas as pd


def validate(path: Path) -> int:
    df = pd.read_csv(path, parse_dates=["datetime"])
    issues = []

    if not df["datetime"].is_monotonic_increasing:
        issues.append("datetime not monotonic increasing")

    bad_ohlc = df[
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
    ]
    if len(bad_ohlc):
        issues.append(f"{len(bad_ohlc)} rows violate low <= O/C <= high")

    nans = df.isna().sum().sum()
    if nans:
        issues.append(f"{nans} NaN cells")

    # weekday gap detection (excluding Sat/Sun)
    df["dow"] = df["datetime"].dt.dayofweek
    weekday = df[df["dow"] < 5].copy()
    weekday["delta"] = weekday["datetime"].diff().dt.total_seconds()
    # heuristic: gap > 2 hours on a weekday is suspicious
    weekday_gaps = weekday[weekday["delta"] > 2 * 3600]
    if len(weekday_gaps):
        issues.append(f"{len(weekday_gaps)} weekday gaps > 2h")

    print(f"== {path.name} ==")
    print(f"  rows: {len(df):,}")
    print(f"  range: {df['datetime'].min()} → {df['datetime'].max()}")
    if issues:
        print("  ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        return 1
    print("  OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: validate.py <csv-path>")
        sys.exit(2)
    sys.exit(validate(Path(sys.argv[1])))
