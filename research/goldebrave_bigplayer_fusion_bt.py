from __future__ import annotations

import argparse
import json
import math
from collections import deque
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


class FusionStrategy(GoldeBraveStrategy):
    """GoldeBrave v4 + BigPlayerDetector v4.3 independent parallel layer.

    A/B/C are preserved. D_BIGPLAYER is an additional causal M15 entry engine using
    the uploaded detector's default logic: LookbackBars=200, VolumeSigmaThreshold=2.0,
    RangeMultiplier=1.5, WickRatioThreshold=1.2, SwingLookback=20.

    Tick volume is reconstructed causally as raw QuoteTick count per M15 bucket.
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
        self.bp_volumes = deque(maxlen=400)
        self.bp_bucket = None
        self.bp_bucket_ticks = 0
        self.bp_signals = {"imb_buy": 0, "imb_sell": 0, "abs_buy": 0, "abs_sell": 0, "sweep_buy": 0, "sweep_sell": 0, "combo_buy": 0, "combo_sell": 0}
        self.bp_entries_attempted = 0

    def _day_range(self, ns: int):
        # Exact EA intent: BarsSinceDayStart = shifted_hour * 4 M15 bars, minimum 1.
        n = max(self.shifted_hour(ns) * 4, 1)
        xs = list(self.bars["M15"])[-n:]
        if not xs:
            return None
        return max(x.h for x in xs), min(x.l for x in xs)

    def on_quote_tick(self, tick: QuoteTick):
        ns = int(tick.ts_event)
        bucket = ns // 900_000_000_000
        if self.bp_bucket is None:
            self.bp_bucket = bucket
        elif bucket != self.bp_bucket:
            self.bp_volumes.append(self.bp_bucket_ticks)
            self.bp_bucket_ticks = 0
            self.bp_bucket = bucket
        self.bp_bucket_ticks += 1
        super().on_quote_tick(tick)

    def on_bar(self, bar: Bar):
        super().on_bar(bar)
        bt = str(bar.bar_type)
        if "15-MINUTE" in bt and self.last_bid is not None:
            self._bigplayer(self._bar(bar).ts_ns)

    def _bigplayer(self, ns: int):
        bars = list(self.bars["M15"])
        if len(bars) < max(self.BP_LOOKBACK, self.BP_SWING_LOOKBACK + 2, 16) or len(self.bp_volumes) < self.BP_LOOKBACK:
            return
        h = self.shifted_hour(ns)
        if h not in self.config.trade_hours or self.spread_break:
            return

        b = bars[-1]
        vols = np.asarray(list(self.bp_volumes)[-self.BP_LOOKBACK:], dtype=float)
        mu, sd = float(vols.mean()), float(vols.std(ddof=0))
        if sd <= 0:
            return
        z = (float(self.bp_bucket_ticks or vols[-1]) - mu) / sd
        if z < self.BP_VOL_SIGMA:
            return

        atr = self._atr(self.bars["M15"], 14)
        if atr is None or atr <= 0:
            return
        rng = b.h - b.l
        if rng <= 0:
            return
        body = abs(b.c - b.o)
        br = body / rng
        upper = b.h - max(b.o, b.c)
        lower = min(b.o, b.c) - b.l
        bullish, bearish = b.c > b.o, b.c < b.o

        imb_buy = bullish and rng / atr >= self.BP_RANGE_MULT and br >= 0.60
        imb_sell = bearish and rng / atr >= self.BP_RANGE_MULT and br >= 0.60
        abs_buy = lower >= body * self.BP_WICK_RATIO and lower > upper
        abs_sell = upper >= body * self.BP_WICK_RATIO and upper > lower

        prev = bars[-(self.BP_SWING_LOOKBACK + 1):-1]
        swing_hi = max(x.h for x in prev)
        swing_lo = min(x.l for x in prev)
        sweep_buy = b.l < swing_lo and b.c > swing_lo
        sweep_sell = b.h > swing_hi and b.c < swing_hi

        for name, flag in (("imb_buy", imb_buy), ("imb_sell", imb_sell), ("abs_buy", abs_buy), ("abs_sell", abs_sell), ("sweep_buy", sweep_buy), ("sweep_sell", sweep_sell)):
            if flag:
                self.bp_signals[name] += 1
        if sweep_buy and imb_buy:
            self.bp_signals["combo_buy"] += 1
        if sweep_sell and imb_sell:
            self.bp_signals["combo_sell"] += 1

        # Independent entry engine: strongest direction wins. Combo > sweep > imbalance > absorption.
        buy_score = 3 * int(sweep_buy and imb_buy) + 2 * int(sweep_buy) + 2 * int(imb_buy) + int(abs_buy)
        sell_score = 3 * int(sweep_sell and imb_sell) + 2 * int(sweep_sell) + 2 * int(imb_sell) + int(abs_sell)
        if buy_score == sell_score:
            return
        side = 1 if buy_score > sell_score else -1

        # Place just beyond the signal bar, retaining GoldeBrave dynamic SL/TP/BE/trail.
        px = (b.h + self.off()) if side > 0 else (b.l - self.off())
        self.bp_entries_attempted += 1
        if not self._near(side, px):
            self._place(side, px, "D_BIGPLAYER", ns)

    def summary(self):
        x = super().summary()
        x["strategy_id"] = "GoldeBrave_v4_plus_BigPlayer_v4_3"
        x["bigplayer"] = {
            "timeframe": "M15 causal",
            "tick_volume_proxy": "Raw QuoteTick count per M15 bucket",
            "defaults": {"LookbackBars": 200, "VolumeSigmaThreshold": 2.0, "RangeMultiplier": 1.5, "WickRatioThreshold": 1.2, "SwingLookback": 20},
            "signals": self.bp_signals,
            "entry_attempts": self.bp_entries_attempted,
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
    s.update({"verification_level": "NAUTILUS_RAW_BIDASK_CAUSAL_FUSION", "engine": "NautilusTrader BacktestEngine", "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"), "raw_ticks": len(ticks), "ohlc_resample_used": False})
    out = Path("results/goldebrave_bigplayer") / args.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([s]).to_json(out / "arena_result.json", orient="records", indent=2)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    engine.dispose()


if __name__ == "__main__":
    main()
