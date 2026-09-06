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
        rows.append({"symbol":"XAUUSD","side":"BUY","entry_time":entry.isoformat(),"exit_time":exit_.isoformat(),
                     "entry_price":2000.0 + price_shift,"exit_price":2001.0 + price_shift,"qty":0.1,"pnl":1.0})
    return pd.DataFrame(rows)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    for c in tlp.NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["symbol", "side", "entry_time"], na_position="last").reset_index(drop=True)


class TradeLevelParityTests(unittest.TestCase):
    def test_timestamp_tolerance_allows_subsecond_skew(self):
        r = tlp.compare(_prep(frame(10)), _prep(frame(10, offset_ms=800)), max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["status"], "PASS"); self.assertEqual(r["mismatch_rows"], 0)

    def test_mismatch_over_three_percent_fails(self):
        r = tlp.compare(_prep(frame(100)), _prep(frame(100, price_shift=10.0)), max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["status"], "FAIL"); self.assertGreater(r["mismatch_pct"], 3.0)

    def test_details_cap_does_not_cap_count(self):
        r = tlp.compare(_prep(frame(200)), _prep(frame(200, price_shift=5.0)), max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["mismatch_rows"], 200); self.assertEqual(len(r["details"]), 50); self.assertEqual(r["mismatch_pct"], 100.0)

    def test_global_assignment_avoids_greedy_false_fail(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        def one(entry_ms: int):
            entry = base + timedelta(milliseconds=entry_ms)
            exit_ = entry + timedelta(seconds=10)
            return {"symbol":"XAUUSD","side":"BUY","entry_time":entry,"exit_time":exit_,"entry_price":2000.0,"exit_price":2001.0,"qty":0.1,"pnl":1.0}
        mt5 = pd.DataFrame([one(0), one(900)])
        nt = pd.DataFrame([one(-800), one(100)])
        r = tlp.compare(_prep(mt5), _prep(nt), max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["status"], "PASS", r); self.assertEqual(r["mismatch_rows"], 0)

    def test_invalid_timestamp_and_numeric_are_fail_closed(self):
        a = frame(1); b = frame(1)
        a.loc[0, "entry_time"] = "bad-time"
        b.loc[0, "qty"] = "not-a-number"
        r = tlp.compare(_prep(a), _prep(b), max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["status"], "FAIL")
        self.assertGreaterEqual(r["mismatch_rows"], 2)
        self.assertTrue(any(x["reason"] == "invalid_required_field" for x in r["details"]))

    def test_large_sparse_export_does_not_require_full_cross_scan(self):
        mt5 = _prep(frame(4000)); nt = _prep(frame(4000, offset_ms=250))
        r = tlp.compare(mt5, nt, max_mismatch_pct=3.0, time_tol_ms=1000)
        self.assertEqual(r["status"], "PASS"); self.assertEqual(r["mismatch_rows"], 0)


if __name__ == "__main__":
    unittest.main()
