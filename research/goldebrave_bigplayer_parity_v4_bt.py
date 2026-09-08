from __future__ import annotations

import argparse
import json
from collections import Counter
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
    """MT5 compatibility/parity adapter v5.

    Raw QuoteTicks remain the only execution source. For BigPlayer's M15 tick_volume[1],
    v5 resolves the just-closed bucket from the latest causal quote bucket at the instant
    Nautilus emits the M15 bar. The primary mapping is latest_quote_bucket - 1, with the
    bar timestamp mappings retained only as causal fallbacks/diagnostics.
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

        # Extra pending telemetry: distinguish repeated candidate generation from true
        # broker-side invalid stop prices / side-cap pressure.
        self.pending_duplicate_attempts = 0
        self.pending_capacity_attempts = 0
        self.pending_price_invalid_attempts = 0
        self.c_daily_attempt_prices: list[float] = []

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
        # Telemetry only; the parent retains original execution behavior.
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

        # Primary causal rule: at bar notification, bar[1] is the bucket immediately before
        # the latest observed raw quote bucket. This avoids dependence on Nautilus bar stamp.
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
                # Do not feed the same closed M15 volume twice into the rolling window.
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
        x["strategy_id"] = "GoldeBrave_v4_plus_BigPlayer_v4_3_parity_v5"
        x["verification_level"] = "NAUTILUS_RAW_BIDASK_MT5_PARITY_V5_CAUSAL_BAR1"
        x["bigplayer"]["volume_alignment_hit"] = self.bp_volume_alignment_hit
        x["bigplayer"]["volume_key_offset"] = dict(self.bp_volume_key_offset)
        x["bigplayer"]["bar_to_latest_quote_bucket_offset"] = dict(self.bp_bar_to_latest_offset)
        x["bigplayer"]["volume_bucket_reuse_blocked"] = self.bp_volume_bucket_reuse
        total = self.bp_volume_alignment_hit + self.bp_volume_alignment_miss
        x["bigplayer"]["volume_alignment_rate_pct"] = 100.0 * self.bp_volume_alignment_hit / max(total, 1)
        x["parity_telemetry"]["pending_duplicate_attempts"] = self.pending_duplicate_attempts
        x["parity_telemetry"]["pending_capacity_attempts"] = self.pending_capacity_attempts
        x["parity_telemetry"]["pending_price_invalid_attempts"] = self.pending_price_invalid_attempts
        if self.c_daily_attempt_prices:
            unique = len({round(p, 2) for p in self.c_daily_attempt_prices})
            x["parity_telemetry"]["c_daily_attempts"] = len(self.c_daily_attempt_prices)
            x["parity_telemetry"]["c_daily_unique_prices_2dp"] = unique
            x["parity_telemetry"]["c_daily_repeat_ratio_pct"] = 100.0 * (len(self.c_daily_attempt_prices) - unique) / max(len(self.c_daily_attempt_prices), 1)
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
