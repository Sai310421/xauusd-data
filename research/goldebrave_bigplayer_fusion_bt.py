from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import nautilus_trader
import numpy as np
import pandas as pd
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.goldebrave_nautilus_raw_bt import GoldeBraveConfig, GoldeBraveStrategy

SIM = Venue("SIM")
M15_NS = 900_000_000_000


class FusionStrategy(GoldeBraveStrategy):
    """GoldeBrave v4 + BigPlayerDetector v4.3 with MT5 parity strengthening.

    A/B/C remain GoldeBrave. D_BIGPLAYER is independent.
    BigPlayer is evaluated on the latest *closed* M15 bar, matching MQL5 bar[1].
    Tick volume is reconstructed as raw QuoteTick count for that exact closed M15 bucket.
    The rolling statistics include the current closed bar before z-score evaluation and use
    the same sample-standard-deviation correction as the uploaded mq5.
    """

    BP_LOOKBACK = 200
    BP_VOL_SIGMA = 2.0
    BP_RANGE_MULT = 1.5
    BP_WICK_RATIO = 1.2
    BP_SWING_LOOKBACK = 20

    def __init__(self, config):
        super().__init__(config)
        self.layer_entries["D_BIGPLAYER"] = 0
        self.layer_pnl["D_BIGPLAYER"] = 0.0
        self.bp_bucket = None
        self.bp_bucket_ticks = 0
        self.bp_completed_volume: dict[int, int] = {}
        self.bp_ring = deque(maxlen=self.BP_LOOKBACK)
        self.bp_last_processed_bar_ns = None
        self.bp_signals = {
            "z_pass": 0,
            "imb_buy": 0, "imb_sell": 0,
            "abs_buy": 0, "abs_sell": 0,
            "sweep_buy": 0, "sweep_sell": 0,
            "combo_buy": 0, "combo_sell": 0,
        }
        self.bp_entries_attempted = 0
        self.bp_closed_bars_seen = 0
        self.bp_volume_alignment_miss = 0
        self.reject_reason = {
            "max_pending_side": 0,
            "buy_not_above_ask": 0,
            "sell_not_below_bid": 0,
            "missing_quote": 0,
            "other": 0,
        }
        self.reject_by_layer: dict[str, int] = {}
        self.place_attempts_by_layer: dict[str, int] = {}
        self.operational_day_resets = 0

    def _day_key(self, ns: int):
        """Reproduce mq5: dayKey = iTime(H1, g_hour).

        g_hour is the shifted broker/session hour. Subtracting that many H1 bars from the
        current H1 open yields the operational day-start H1 timestamp, which stays constant
        through the day. The prior reconstruction incorrectly included hour in the key.
        """
        dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        return int((dt - timedelta(hours=self.shifted_hour(ns))).timestamp())

    def _session_rebuild(self, ns: int, changed_tf: str):
        dk = self._day_key(ns)
        if self.day_key != dk:
            # mq5 RefreshEntriesToday() counts only today's entry deals; forward simulation
            # is equivalent to resetting this counter at the operational day boundary.
            self.entries_today = 0
            self.operational_day_resets += 1
        return super()._session_rebuild(ns, changed_tf)

    def _day_range(self, ns: int):
        # mq5 BarsSinceDayStart = g_hour*3600 / PeriodSeconds(M15), minimum 1.
        n = max(self.shifted_hour(ns) * 4, 1)
        xs = list(self.bars["M15"])[-n:]
        if not xs:
            return None
        return max(x.h for x in xs), min(x.l for x in xs)

    def _place(self, side, price, layer, ns, track_placed=False):
        self.place_attempts_by_layer[layer] = self.place_attempts_by_layer.get(layer, 0) + 1
        reason = None
        if self.config.max_pending_per_side > 0 and self._pending_count(side) >= self.config.max_pending_per_side:
            reason = "max_pending_side"
        elif self.last_ask is None or self.last_bid is None:
            reason = "missing_quote"
        elif side > 0 and price <= self.last_ask:
            reason = "buy_not_above_ask"
        elif side < 0 and price >= self.last_bid:
            reason = "sell_not_below_bid"
        before = self.rejected_stops
        ok = super()._place(side, price, layer, ns, track_placed=track_placed)
        if not ok and self.rejected_stops > before:
            reason = reason or "other"
            self.reject_reason[reason] += 1
            self.reject_by_layer[layer] = self.reject_by_layer.get(layer, 0) + 1
        return ok

    def on_quote_tick(self, tick: QuoteTick):
        ns = int(tick.ts_event)
        bucket = ns // M15_NS
        if self.bp_bucket is None:
            self.bp_bucket = bucket
        elif bucket != self.bp_bucket:
            self.bp_completed_volume[self.bp_bucket] = self.bp_bucket_ticks
            cutoff = bucket - 260
            for k in [x for x in self.bp_completed_volume if x < cutoff]:
                del self.bp_completed_volume[k]
            self.bp_bucket = bucket
            self.bp_bucket_ticks = 0
        self.bp_bucket_ticks += 1
        super().on_quote_tick(tick)

    def on_bar(self, bar: Bar):
        super().on_bar(bar)
        bt = str(bar.bar_type)
        if "15-MINUTE" not in bt or self.last_bid is None:
            return
        b = self._bar(bar)
        if self.bp_last_processed_bar_ns == b.ts_ns:
            return
        self.bp_last_processed_bar_ns = b.ts_ns
        self._bigplayer_closed_bar(b)

    @staticmethod
    def _sample_mean_std(values):
        n = len(values)
        if n <= 1:
            return 0.0, 0.0
        s = float(sum(values))
        ss = float(sum(v * v for v in values))
        mean = s / n
        variance = max(0.0, (ss / n) - mean * mean)
        return mean, float(np.sqrt(variance * n / (n - 1)))

    def _closed_bar_tick_volume(self, b):
        close_bucket = b.ts_ns // M15_NS
        for key in (close_bucket - 1, close_bucket):
            if key in self.bp_completed_volume:
                return self.bp_completed_volume[key]
        self.bp_volume_alignment_miss += 1
        return None

    def _bigplayer_closed_bar(self, b):
        bars = list(self.bars["M15"])
        self.bp_closed_bars_seen += 1
        if len(bars) < self.BP_SWING_LOOKBACK + 2:
            return
        vol = self._closed_bar_tick_volume(b)
        if vol is None:
            return
        # mq5 AddToRolling(tick_volume[1]) occurs before AnalyzeBar(1).
        self.bp_ring.append(float(vol))
        if len(self.bp_ring) < max(2, self.BP_LOOKBACK // 2):
            return
        h = self.shifted_hour(b.ts_ns)
        if h not in self.config.trade_hours or self.spread_break:
            return
        mu, sd = self._sample_mean_std(self.bp_ring)
        if sd <= 0.0 or mu <= 0.0:
            return
        z = (float(vol) - mu) / sd
        if z < self.BP_VOL_SIGMA:
            return
        self.bp_signals["z_pass"] += 1
        atr = self._atr(self.bars["M15"], 14)
        if atr is None or atr <= 0:
            return
        rng = b.h - b.l
        if rng <= 0:
            return
        body = abs(b.c - b.o)
        body_ratio = body / rng
        upper = b.h - max(b.o, b.c)
        lower = min(b.o, b.c) - b.l
        bullish, bearish = b.c > b.o, b.c < b.o
        imb_buy = bullish and (rng / atr) >= self.BP_RANGE_MULT and body_ratio >= 0.60
        imb_sell = bearish and (rng / atr) >= self.BP_RANGE_MULT and body_ratio >= 0.60
        abs_buy = lower >= body * self.BP_WICK_RATIO and lower > upper
        abs_sell = upper >= body * self.BP_WICK_RATIO and upper > lower
        prev = bars[-(self.BP_SWING_LOOKBACK + 1):-1]
        if len(prev) < self.BP_SWING_LOOKBACK:
            return
        swing_hi = max(x.h for x in prev)
        swing_lo = min(x.l for x in prev)
        sweep_buy = b.l < swing_lo and b.c > swing_lo
        sweep_sell = b.h > swing_hi and b.c < swing_hi
        for name, flag in (
            ("imb_buy", imb_buy), ("imb_sell", imb_sell),
            ("abs_buy", abs_buy), ("abs_sell", abs_sell),
            ("sweep_buy", sweep_buy), ("sweep_sell", sweep_sell),
        ):
            if flag:
                self.bp_signals[name] += 1
        if sweep_buy and imb_buy:
            self.bp_signals["combo_buy"] += 1
        if sweep_sell and imb_sell:
            self.bp_signals["combo_sell"] += 1
        buy_score = 3 * int(sweep_buy and imb_buy) + 2 * int(sweep_buy) + 2 * int(imb_buy) + int(abs_buy)
        sell_score = 3 * int(sweep_sell and imb_sell) + 2 * int(sweep_sell) + 2 * int(imb_sell) + int(abs_sell)
        if buy_score == sell_score:
            return
        side = 1 if buy_score > sell_score else -1
        px = (b.h + self.off()) if side > 0 else (b.l - self.off())
        self.bp_entries_attempted += 1
        if not self._near(side, px):
            self._place(side, px, "D_BIGPLAYER", b.ts_ns)

    def summary(self):
        x = super().summary()
        x["strategy_id"] = "GoldeBrave_v4_plus_BigPlayer_v4_3_parity_v3"
        x["bigplayer"] = {
            "timeframe": "M15 latest closed bar / MT5 bar[1] semantics",
            "tick_volume_proxy": "exact Raw QuoteTick count for each closed M15 bucket",
            "rolling_semantics": "current closed bar added before z-score; sample std correction n/(n-1)",
            "defaults": {
                "LookbackBars": 200,
                "VolumeSigmaThreshold": 2.0,
                "RangeMultiplier": 1.5,
                "WickRatioThreshold": 1.2,
                "SwingLookback": 20,
            },
            "signals": self.bp_signals,
            "entry_attempts": self.bp_entries_attempted,
            "closed_bars_seen": self.bp_closed_bars_seen,
            "volume_alignment_miss": self.bp_volume_alignment_miss,
            "rolling_count_final": len(self.bp_ring),
        }
        x["parity_telemetry"] = {
            "operational_day_resets": self.operational_day_resets,
            "place_attempts_by_layer": self.place_attempts_by_layer,
            "reject_by_layer": self.reject_by_layer,
            "reject_reason": self.reject_reason,
            "reject_rate_pct": 100.0 * self.rejected_stops / max(sum(self.place_attempts_by_layer.values()), 1),
        }
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--raw-bidask-only", action="store_true")
    ap.add_argument("--initial-balance", type=float, default=100000.0)
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit("raw-bidask-only mandatory")
    catalog = ParquetDataCatalog(str(Path(args.catalog)))
    instrument = next((x for x in catalog.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD"), None)
    if instrument is None:
        raise SystemExit("XAUUSD missing")
    ticks = catalog.query(data_cls=QuoteTick, identifiers=[instrument.id.value])
    if not ticks:
        raise SystemExit("no raw XAUUSD QuoteTicks")
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"), risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=SIM, oms_type=OmsType.HEDGING, account_type=AccountType.MARGIN, base_currency=USD, starting_balances=[Money(args.initial_balance, USD)], default_leverage=Decimal("2000"))
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    iid = instrument.id.value
    m1 = BarType.from_str(f"{iid}-1-MINUTE-BID-INTERNAL")
    m15 = BarType.from_str(f"{iid}-15-MINUTE-BID-INTERNAL")
    h1 = BarType.from_str(f"{iid}-1-HOUR-BID-INTERNAL")
    st = FusionStrategy(GoldeBraveConfig(instrument_id=instrument.id, m1=m1, m15=m15, h1=h1, initial_balance=args.initial_balance))
    engine.add_strategy(st)
    engine.run()
    s = st.summary()
    s.update({
        "verification_level": "NAUTILUS_RAW_BIDASK_MT5_PARITY_V3",
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "raw_ticks": len(ticks),
        "ohlc_resample_used": False,
    })
    out = Path("results/goldebrave_bigplayer") / args.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([s]).to_json(out / "arena_result.json", orient="records", indent=2)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    engine.dispose()


if __name__ == "__main__":
    main()
