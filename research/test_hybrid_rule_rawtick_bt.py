from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

import hybrid_rule_rawtick_bt as h


class HybridRuleRawTickTests(unittest.TestCase):
    def _ticks(self):
        base = pd.Timestamp(datetime(2026, 1, 1, tzinfo=timezone.utc))
        return pd.DataFrame([
            {"datetime": base + pd.Timedelta(seconds=0), "bid": 2000.0, "ask": 2000.2, "bid_size": 1.0, "ask_size": 1.0, "mid": 2000.1},
            {"datetime": base + pd.Timedelta(minutes=1), "bid": 2001.0, "ask": 2001.2, "bid_size": 1.0, "ask_size": 1.0, "mid": 2001.1},
            {"datetime": base + pd.Timedelta(minutes=2), "bid": 1999.0, "ask": 1999.2, "bid_size": 1.0, "ask_size": 1.0, "mid": 1999.1},
            {"datetime": base + pd.Timedelta(minutes=3), "bid": 1998.0, "ask": 1998.2, "bid_size": 1.0, "ask_size": 1.0, "mid": 1998.1},
        ])

    def test_buy_entry_uses_subsequent_ask_not_bar_price(self):
        ticks = self._ticks()
        signals = pd.DataFrame([{"datetime": ticks.iloc[0].datetime, "direction": "BUY", "score": 0.9}])
        trades = h.run_execution(ticks, signals)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].entry_time, ticks.iloc[1].datetime.isoformat())
        self.assertEqual(trades[0].entry_price, ticks.iloc[1].ask)
        self.assertEqual(trades[0].exit_price, ticks.iloc[-1].bid)

    def test_reversal_closes_long_at_bid_and_opens_short_at_bid(self):
        ticks = self._ticks()
        signals = pd.DataFrame([
            {"datetime": ticks.iloc[0].datetime, "direction": "BUY", "score": 0.9},
            {"datetime": ticks.iloc[1].datetime, "direction": "SELL", "score": 0.9},
        ])
        trades = h.run_execution(ticks, signals)
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].exit_price, ticks.iloc[2].bid)
        self.assertEqual(trades[1].entry_price, ticks.iloc[2].bid)
        self.assertEqual(trades[1].exit_price, ticks.iloc[-1].ask)

    def test_same_direction_signal_does_not_duplicate_position(self):
        ticks = self._ticks()
        signals = pd.DataFrame([
            {"datetime": ticks.iloc[0].datetime, "direction": "BUY", "score": 0.8},
            {"datetime": ticks.iloc[1].datetime, "direction": "BUY", "score": 0.9},
        ])
        trades = h.run_execution(ticks, signals)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].entry_price, ticks.iloc[1].ask)

    def test_metrics_are_strict_json_even_with_no_losses(self):
        t = h.Trade("BUY", "a", "b", 1.0, 2.0, 1.0, 0.9, "END")
        m = h.metrics([t])
        self.assertIsNone(m["PF"])
        self.assertTrue(m["PF_infinite"])
        json.dumps(m, allow_nan=False)

    def test_raw_tick_bar_builder_is_causal_minute_grouping(self):
        ticks = self._ticks()
        bars = h.raw_ticks_to_m1(ticks)
        self.assertEqual(len(bars), 4)
        self.assertTrue((bars.tick_count == 1).all())
        self.assertEqual(float(bars.iloc[0].open), ticks.iloc[0].mid)


if __name__ == "__main__":
    unittest.main()
