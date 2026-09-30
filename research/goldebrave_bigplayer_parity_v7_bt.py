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

from research.goldebrave_bigplayer_parity_v4_bt import ParityV4Strategy
from research.goldebrave_nautilus_raw_bt import GoldeBraveConfig

SIM = Venue("SIM")
M15_NS = 900_000_000_000


class ParityV7Strategy(ParityV4Strategy):
    """Parity v7: faithful BigPlayer detector integration.

    The uploaded BigPlayerDetector is an indicator/detector and does not define an order
    entry/exit engine. Earlier fusion revisions converted detector scores into an invented
    independent D_BIGPLAYER stop-entry rule. v7 removes that synthetic execution.

    GoldeBrave A/B/C execution is unchanged from parity v6. BigPlayer continues to run on
    causal closed M15 bars, but its directional candidates are telemetry only. We record
    whether later A/B/C pending candidates agree with a recent BigPlayer direction, so a
    future filter/boost can be validated before changing execution.
    """

    BP_CONFLUENCE_WINDOW_NS = 4 * M15_NS  # 60 minutes; telemetry only, no trade filtering.

    def __init__(self, config):
        super().__init__(config)
        self.bp_detector_only_candidates = 0
        self.bp_last_candidate_side: int | None = None
        self.bp_last_candidate_ns: int | None = None
        self.bp_candidate_side = Counter()
        self.bp_confluence_by_layer = Counter()
        self.bp_conflict_by_layer = Counter()
        self.bp_no_recent_by_layer = Counter()
        self.bp_confluence_lag_buckets = Counter()
        self.synthetic_d_orders_blocked = 0

    def _place(self, side, price, layer, ns, track_placed=False):
        if layer == "D_BIGPLAYER":
            # Detector signal only. The source mq5 does not define an execution engine.
            self.synthetic_d_orders_blocked += 1
            self.bp_detector_only_candidates += 1
            self.bp_last_candidate_side = int(side)
            self.bp_last_candidate_ns = int(ns)
            self.bp_candidate_side["buy" if side > 0 else "sell"] += 1
            return False

        # Telemetry only: do not filter or size A/B/C yet.
        if layer in ("A_H1", "B_M15", "C_DAILY"):
            if self.bp_last_candidate_ns is None or ns < self.bp_last_candidate_ns:
                self.bp_no_recent_by_layer[layer] += 1
            else:
                lag = ns - self.bp_last_candidate_ns
                if lag <= self.BP_CONFLUENCE_WINDOW_NS:
                    self.bp_confluence_lag_buckets[str(int(lag // M15_NS))] += 1
                    if side == self.bp_last_candidate_side:
                        self.bp_confluence_by_layer[layer] += 1
                    else:
                        self.bp_conflict_by_layer[layer] += 1
                else:
                    self.bp_no_recent_by_layer[layer] += 1
        return super()._place(side, price, layer, ns, track_placed=track_placed)

    def summary(self):
        x = super().summary()
        x["strategy_id"] = "GoldeBrave_v4_plus_BigPlayerDetector_parity_v7_detector_only"
        x["verification_level"] = "NAUTILUS_RAW_BIDASK_MT5_PARITY_V7_DETECTOR_ONLY"
        x["bigplayer"]["integration_mode"] = "DETECTOR_ONLY_NO_SYNTHETIC_D_EXECUTION"
        x["bigplayer"]["source_semantics"] = (
            "Uploaded BigPlayerDetector defines detection/markers/notifications, not trade execution"
        )
        x["bigplayer"]["detector_only_candidates"] = self.bp_detector_only_candidates
        x["bigplayer"]["candidate_side"] = dict(self.bp_candidate_side)
        x["bigplayer"]["synthetic_d_orders_blocked"] = self.synthetic_d_orders_blocked
        x["bigplayer"]["confluence_window_minutes"] = 60
        x["bigplayer"]["confluence_by_layer"] = dict(self.bp_confluence_by_layer)
        x["bigplayer"]["conflict_by_layer"] = dict(self.bp_conflict_by_layer)
        x["bigplayer"]["no_recent_signal_by_layer"] = dict(self.bp_no_recent_by_layer)
        x["bigplayer"]["confluence_lag_m15_buckets"] = dict(self.bp_confluence_lag_buckets)
        # Explicitly state that D PnL is not a valid concept for the source indicator.
        x["layer_entries"]["D_BIGPLAYER"] = 0
        x["layer_pnl"]["D_BIGPLAYER"] = 0.0
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

    engine = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="ERROR"),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(args.initial_balance, USD)],
        default_leverage=Decimal("2000"),
    )
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    iid = instrument.id.value
    st = ParityV7Strategy(GoldeBraveConfig(
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
