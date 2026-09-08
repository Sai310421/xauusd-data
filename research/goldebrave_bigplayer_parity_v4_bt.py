from __future__ import annotations

import argparse
import json
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
    """Event-order-safe MT5 parity layer.

    Nautilus may emit an INTERNAL time bar before the strategy receives the first quote of
    the next bucket, so a 'freeze on bucket transition' design misses nearly all closed-bar
    volumes. V4 continuously counts every raw quote by M15 bucket and resolves bar volume
    directly from the bar timestamp. This removes the event-order dependency.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bp_bucket_counts: dict[int, int] = {}
        self.bp_volume_alignment_hit = 0
        self.bp_volume_key_offset = {"bar_bucket": 0, "bar_bucket_minus_1": 0}

    def on_quote_tick(self, tick: QuoteTick):
        ns = int(tick.ts_event)
        bucket = ns // M15_NS
        self.bp_bucket_counts[bucket] = self.bp_bucket_counts.get(bucket, 0) + 1
        cutoff = bucket - 300
        if len(self.bp_bucket_counts) > 340:
            for k in [x for x in self.bp_bucket_counts if x < cutoff]:
                del self.bp_bucket_counts[k]
        super().on_quote_tick(tick)

    def _closed_bar_tick_volume(self, b):
        bar_bucket = b.ts_ns // M15_NS
        # INTERNAL bars commonly carry a timestamp inside the bar's own bucket. Prefer that.
        # If the engine stamps at the close boundary, previous bucket is the correct bar.
        if bar_bucket in self.bp_bucket_counts and self.bp_bucket_counts[bar_bucket] > 0:
            self.bp_volume_alignment_hit += 1
            self.bp_volume_key_offset["bar_bucket"] += 1
            return self.bp_bucket_counts[bar_bucket]
        prev = bar_bucket - 1
        if prev in self.bp_bucket_counts and self.bp_bucket_counts[prev] > 0:
            self.bp_volume_alignment_hit += 1
            self.bp_volume_key_offset["bar_bucket_minus_1"] += 1
            return self.bp_bucket_counts[prev]
        self.bp_volume_alignment_miss += 1
        return None

    def summary(self):
        x = super().summary()
        x["strategy_id"] = "GoldeBrave_v4_plus_BigPlayer_v4_3_parity_v4"
        x["verification_level"] = "NAUTILUS_RAW_BIDASK_MT5_PARITY_V4_EVENT_ORDER_SAFE"
        x["bigplayer"]["volume_alignment_hit"] = self.bp_volume_alignment_hit
        x["bigplayer"]["volume_key_offset"] = self.bp_volume_key_offset
        total = self.bp_volume_alignment_hit + self.bp_volume_alignment_miss
        x["bigplayer"]["volume_alignment_rate_pct"] = 100.0 * self.bp_volume_alignment_hit / max(total, 1)
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
