from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import nautilus_trader
import pandas as pd
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.goldebrave_bigplayer_fusion_bt import FusionStrategy, M15_NS
from research.goldebrave_nautilus_raw_bt import GoldeBraveConfig

SIM = Venue("SIM")


class ParityV4Strategy(FusionStrategy):
    """MT5 compatibility/parity adapter v6.

    v6 fixes the context around Layer C without reversing its original direction:
      day high + offset -> BuyStop
      day low  - offset -> SellStop

    Critical parity fixes:
    - Raw Dukascopy UTC is first mapped to MT5 broker-server time.
    - ShiftedHour then applies the EA's GMT/GMTS correction to server time.
    - iTime(H1, g_hour) is represented as the shifted trading-day midnight key.
    - DayRange uses exactly max(g_hour * 4, 1) latest M15 bars (BarrierTF=M15),
      matching BarsSinceDayStart() semantics rather than UTC calendar-day slicing.
    - entriesToday is refreshed from fills belonging to the current broker D1 day,
      matching RefreshEntriesToday(), and is not tied to the A/B dayKey.

    Raw QuoteTicks remain the only execution source. No OHLC-resample execution.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bp_bucket_counts: dict[int, int] = {}
        self.bp_latest_quote_bucket: int | None = None
        self.bp_volume_alignment_hit = 0
        self.bp_volume_key_offset = Counter()
        self.bp_bar_to_latest_offset = Counter()
        self.bp_volume_bucket_reuse = 0
        self.bp_last_volume_bucket: int | None = None

        self.pending_duplicate_attempts = 0
        self.pending_capacity_attempts = 0
        self.pending_price_invalid_attempts = 0
        self.c_daily_attempt_prices: list[float] = []

        self.entry_fill_times: list[int] = []
        self.c_refresh_calls = 0
        self.c_dayrange_bar_counts = Counter()
        self.c_direction_attempts = Counter()

    # ------------------------------------------------------------------
    # MT5 clock parity
    # ------------------------------------------------------------------
    def _dst_for_clock(self, dt):
        return self._eu_dst(dt) if self.config.dst_area == 1 else self._us_dst(dt)

    def _server_dt(self, ns: int):
        """Convert raw UTC timestamp to the broker server clock configured by GMT/GMTS."""
        utc = self._dt(ns)
        # Determine the seasonal server offset. Around the transition hour the original EA
        # itself uses a simplified DST formula, so retaining that formula is intentional.
        dst = self._dst_for_clock(utc)
        off = self.config.gmts if dst else self.config.gmt
        return utc + timedelta(hours=off)

    def _trading_dt(self, ns: int):
        """Clock seen by g_hour after ShiftedHour(server_time)."""
        server = self._server_dt(ns)
        dst = self._dst_for_clock(server)
        correction = (3 - self.config.gmts) if dst else (2 - self.config.gmt)
        return server + timedelta(hours=correction)

    def shifted_hour(self, ns: int) -> int:
        return self._trading_dt(ns).hour

    def _day_key(self, ns: int):
        # MQL: iTime(_Symbol, PERIOD_H1, g_hour). With hourly-aligned bars this is
        # the current shifted trading day's midnight anchor, NOT (date, hour).
        d = self._trading_dt(ns)
        return d.year, d.month, d.day

    def _broker_d1_key(self, ns: int):
        # HistorySelect(iTime(PERIOD_D1,0), TimeCurrent()) follows broker D1.
        d = self._server_dt(ns)
        return d.year, d.month, d.day

    def _refresh_entries_today(self, ns: int):
        key = self._broker_d1_key(ns)
        self.entries_today = sum(1 for t in self.entry_fill_times if self._broker_d1_key(t) == key)
        self.c_refresh_calls += 1

    # ------------------------------------------------------------------
    # Layer C / DayRange exact structural parity
    # ------------------------------------------------------------------
    def _day_range(self, ns: int):
        # MQL BarsSinceDayStart(): int((g_hour*3600)/PeriodSeconds(M15)); max(n,1)
        n = max(int((self.shifted_hour(ns) * 3600) / (15 * 60)), 1)
        xs = list(self.bars["M15"])[-n:]
        if not xs:
            return None
        self.c_dayrange_bar_counts[n] += 1
        return max(x.h for x in xs), min(x.l for x in xs)

    def _open(self, pend, fill: float, ns: int):
        super()._open(pend, fill, ns)
        self.entry_fill_times.append(ns)
        # Parent increments a running counter; reset it to MT5 D1-history semantics.
        self._refresh_entries_today(ns)

    def _boost(self, ns):
        # Original OnTick refreshes HistorySelect count before the M15-new-bar BoostLayer call.
        self._refresh_entries_today(ns)
        h = self.shifted_hour(ns)
        if h not in self.config.trade_hours or h < self.config.boost_hour:
            return
        if self.config.min_entries_per_day <= 0 or self.entries_today >= self.config.min_entries_per_day:
            return
        if self.config.adapt and self.config.range_disable_c and self.regime == -1:
            return
        dr = self._day_range(ns)
        if dr is None:
            return
        dhi, dlo = dr
        off = self.config.boost_off_units * self.config.unit * self.g_vol
        if self.config.adapt and self.regime == -1:
            off *= self.config.range_boost_mult

        # Preserve ORIGINAL C direction exactly: upper breakout BUY, lower breakout SELL.
        p, q = dhi + off, dlo - off
        if not self._near(+1, p):
            self.c_direction_attempts["upper_buy"] += 1
            self._place(+1, p, "C_DAILY", ns)
        if not self._near(-1, q):
            self.c_direction_attempts["lower_sell"] += 1
            self._place(-1, q, "C_DAILY", ns)

    # ------------------------------------------------------------------
    # BigPlayer Raw QuoteTick -> MT5-like tick_volume[1]
    # ------------------------------------------------------------------
    def on_quote_tick(self, tick: QuoteTick):
        ns = int(tick.ts_event)
        bucket = ns // M15_NS
        self.bp_latest_quote_bucket = bucket
        self.bp_bucket_counts[bucket] = self.bp_bucket_counts.get(bucket, 0) + 1
        cutoff = bucket - 420
        if len(self.bp_bucket_counts) > 460:
            for k in [x for x in self.bp_bucket_counts if x < cutoff]:
                del self.bp_bucket_counts[k]
        super().on_quote_tick(tick)

    def _place(self, side, price, layer, ns, track_placed=False):
        if any(p.side == side and abs(p.price - price) <= self.config.unit * 0.25 for p in self.pending):
            self.pending_duplicate_attempts += 1
        if self.config.max_pending_per_side > 0 and self._pending_count(side) >= self.config.max_pending_per_side:
            self.pending_capacity_attempts += 1
        if self.last_ask is not None and self.last_bid is not None:
            if (side > 0 and price <= self.last_ask) or (side < 0 and price >= self.last_bid):
                self.pending_price_invalid_attempts += 1
        if layer == "C_DAILY":
            self.c_daily_attempt_prices.append(float(price))
            if len(self.c_daily_attempt_prices) > 2000:
                self.c_daily_attempt_prices = self.c_daily_attempt_prices[-2000:]
        return super()._place(side, price, layer, ns, track_placed=track_placed)

    def _closed_bar_tick_volume(self, b):
        bar_bucket = b.ts_ns // M15_NS
        if self.bp_latest_quote_bucket is not None:
            self.bp_bar_to_latest_offset[self.bp_latest_quote_bucket - bar_bucket] += 1

        candidates: list[tuple[str, int]] = []
        if self.bp_latest_quote_bucket is not None:
            candidates.append(("latest_minus_1", self.bp_latest_quote_bucket - 1))
        candidates.extend([
            ("bar_bucket_minus_1", bar_bucket - 1),
            ("bar_bucket", bar_bucket),
        ])

        seen = set()
        for label, key in candidates:
            if key in seen:
                continue
            seen.add(key)
            vol = self.bp_bucket_counts.get(key, 0)
            if vol <= 0:
                continue
            if self.bp_last_volume_bucket == key:
                self.bp_volume_bucket_reuse += 1
                continue
            self.bp_last_volume_bucket = key
            self.bp_volume_alignment_hit += 1
            self.bp_volume_key_offset[label] += 1
            return vol

        self.bp_volume_alignment_miss += 1
        return None

    def summary(self):
        x = super().summary()
        x["strategy_id"] = "GoldeBrave_v4_plus_BigPlayer_v4_3_parity_v6"
        x["verification_level"] = "NAUTILUS_RAW_BIDASK_MT5_PARITY_V6_CLOCK_D1_DAYRANGE"
        x["bigplayer"]["volume_alignment_hit"] = self.bp_volume_alignment_hit
        x["bigplayer"]["volume_key_offset"] = dict(self.bp_volume_key_offset)
        x["bigplayer"]["bar_to_latest_quote_bucket_offset"] = dict(self.bp_bar_to_latest_offset)
        x["bigplayer"]["volume_bucket_reuse_blocked"] = self.bp_volume_bucket_reuse
        total = self.bp_volume_alignment_hit + self.bp_volume_alignment_miss
        x["bigplayer"]["volume_alignment_rate_pct"] = 100.0 * self.bp_volume_alignment_hit / max(total, 1)

        t = x["parity_telemetry"]
        t["pending_duplicate_attempts"] = self.pending_duplicate_attempts
        t["pending_capacity_attempts"] = self.pending_capacity_attempts
        t["pending_price_invalid_attempts"] = self.pending_price_invalid_attempts
        t["c_entries_refresh_calls"] = self.c_refresh_calls
        t["c_direction_attempts"] = dict(self.c_direction_attempts)
        t["c_dayrange_bar_counts"] = dict(self.c_dayrange_bar_counts)
        t["clock_model"] = "UTC -> broker GMT/GMTS server -> EA ShiftedHour correction"
        t["day_key_model"] = "iTime(H1,g_hour) ~= shifted trading-day midnight"
        t["entries_today_model"] = "broker D1 fill-history recount"
        if self.c_daily_attempt_prices:
            unique = len({round(p, 2) for p in self.c_daily_attempt_prices})
            t["c_daily_attempts"] = len(self.c_daily_attempt_prices)
            t["c_daily_unique_prices_2dp"] = unique
            t["c_daily_repeat_ratio_pct"] = 100.0 * (len(self.c_daily_attempt_prices) - unique) / max(len(self.c_daily_attempt_prices), 1)
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
    st = ParityV4Strategy(GoldeBraveConfig(
        instrument_id=instrument.id,
        m1=BarType.from_str(f"{iid}-1-MINUTE-BID-INTERNAL"),
        m15=BarType.from_str(f"{iid}-15-MINUTE-BID-INTERNAL"),
        h1=BarType.from_str(f"{iid}-1-HOUR-BID-INTERNAL"),
        initial_balance=args.initial_balance,
    ))
    engine.add_strategy(st)
    engine.run()
    s = st.summary()
    s.update({
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
