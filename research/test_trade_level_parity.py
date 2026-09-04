from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

import trade_level_parity as tlp


def frame(n: int, *, offset_ms=0, price_shift=0.0, start=None):
    start = start or datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        entry = start + timedelta(seconds=i * 60, milliseconds=offset_ms)
        exit_ = entry + timedelta(seconds=30)
        rows.append({
            "symbol": "XAUUSD",
            "side": "BUY",
            "entry_time": entry.isoformat(),
            "exit_time": exit_.isoformat(),
            "entry_price": 2000.0 + price_shift,
            "exit_price": 2001.0 + price_shift,
            "qty": 0.1,
            "pnl": 1.0,
        })
    return pd.DataFrame(rows)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True)
    return out


class TradeLevelParityTests(unittest.TestCase):
    def test_timestamp_tolerance_allows_subsecond_skew(self):
        r = tlp.compare(_prep(frame(10)), _prep(frame(10, offset_ms=800)), max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["status"], "PASS")
        self.assertEqual(r["mismatch_rows"], 0)

    def test_mismatch_over_three_percent_fails(self):
        r = tlp.compare(_prep(frame(100)), _prep(frame(100, price_shift=10.0)), max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["status"], "FAIL")
        self.assertGreater(r["mismatch_pct"], 3.0)

    def test_details_cap_does_not_cap_count(self):
        r = tlp.compare(_prep(frame(200)), _prep(frame(200, price_shift=5.0)), max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["mismatch_rows"], 200)
        self.assertEqual(len(r["details"]), 50)
        self.assertEqual(r["mismatch_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
