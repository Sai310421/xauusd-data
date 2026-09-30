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

from research.goldebrave_bigplayer_parity_v8_bt import ParityV8Strategy
from research.goldebrave_nautilus_raw_bt import GoldeBraveConfig, Pending

SIM = Venue("SIM")


class ParityV9Strategy(ParityV8Strategy):
    """Parity v9: audit accepted pending -> fill/delete lifecycle without changing execution.

    No strategy optimization is introduced here. Every accepted virtual MT5 pending order is
    tracked until fill or deletion, including lifetime and closest executable-price distance.
    This distinguishes legitimate far-away GTC ZigZag levels from premature reconstruction
    deletion caused by session/day/spread state mismatch.
    """

    def __init__(self, config):
        super().__init__(config)
        self._pending_meta: dict[int, dict] = {}
        self._delete_reason_hint = "unspecified"
        self.pending_terminal = Counter()
        self.pending_terminal_by_layer = Counter()
        self.pending_lifetime_s_by_layer: dict[str, list[float]] = {"A_H1": [], "B_M15": [], "C_DAILY": []}
        self.pending_closest_gap_by_layer: dict[str, list[float]] = {"A_H1": [], "B_M15": [], "C_DAILY": []}
        self.pending_created_by_layer = Counter()
        self.pending_fill_by_layer = Counter()

    def _place(self, side, price, layer, ns, track_placed=False):
        before = len(self.pending)
        ok = super()._place(side, price, layer, ns, track_placed=track_placed)
        if ok and len(self.pending) > before:
            p = self.pending[-1]
            self._pending_meta[id(p)] = {
                "layer": layer,
                "side": int(side),
                "price": float(price),
                "created_ns": int(ns),
                "closest_gap": float("inf"),
            }
            self.pending_created_by_layer[layer] += 1
        return ok

    def _update_pending_gap(self):
        if self.last_bid is None or self.last_ask is None:
            return
        for p in self.pending:
            m = self._pending_meta.get(id(p))
            if m is None:
                continue
            # Distance still required before trigger; negative/zero means crossed.
            gap = (p.price - self.last_ask) if p.side > 0 else (self.last_bid - p.price)
            m["closest_gap"] = min(float(m["closest_gap"]), float(gap))

    def _finalize_pending(self, p: Pending, ns: int, terminal: str):
        m = self._pending_meta.pop(id(p), None)
        if m is None:
            return
        layer = m["layer"]
        life_s = max(0.0, (int(ns) - int(m["created_ns"])) / 1e9)
        gap = m["closest_gap"]
        if gap == float("inf"):
            gap = None
        self.pending_terminal[terminal] += 1
        self.pending_terminal_by_layer[f"{layer}:{terminal}"] += 1
        self.pending_lifetime_s_by_layer.setdefault(layer, []).append(life_s)
        if gap is not None:
            self.pending_closest_gap_by_layer.setdefault(layer, []).append(float(gap))

    def _delete_pending(self):
        ns = int(self.last_ns or 0)
        reason = self._delete_reason_hint or "unspecified"
        for p in list(self.pending):
            self._finalize_pending(p, ns, f"deleted_{reason}")
        return super()._delete_pending()

    def _session_rebuild(self, ns: int, changed_tf: str):
        h = self.shifted_hour(ns)
        dk = self._day_key(ns)
        old = self._delete_reason_hint
        if h not in self.config.trade_hours:
            self._delete_reason_hint = "out_of_session"
        elif self.last_build_day != dk:
            self._delete_reason_hint = "daily_rebuild"
        else:
            self._delete_reason_hint = old
        try:
            return super()._session_rebuild(ns, changed_tf)
        finally:
            self._delete_reason_hint = old

    def _open(self, pend: Pending, fill: float, ns: int):
        self._finalize_pending(pend, ns, "filled")
        self.pending_fill_by_layer[pend.layer] += 1
        return super()._open(pend, fill, ns)

    def on_quote_tick(self, tick: QuoteTick):
        # Update closest approach using this raw executable quote before base crossing logic.
        bid = self.f(tick.bid_price)
        ask = self.f(tick.ask_price)
        for p in self.pending:
            m = self._pending_meta.get(id(p))
            if m is None:
                continue
            gap = (p.price - ask) if p.side > 0 else (bid - p.price)
            m["closest_gap"] = min(float(m["closest_gap"]), float(gap))

        spread_units = (ask - bid) / self.config.unit
        h = self.shifted_hour(int(tick.ts_event))
        old = self._delete_reason_hint
        if h in self.config.trade_hours and spread_units > self.config.spread_break_units and not self.spread_break:
            self._delete_reason_hint = "spread_break"
        try:
            return super().on_quote_tick(tick)
        finally:
            self._delete_reason_hint = old

    @staticmethod
    def _stats(values):
        xs = sorted(float(x) for x in values)
        if not xs:
            return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
        n = len(xs)
        def q(frac):
            return xs[min(n - 1, max(0, int(round((n - 1) * frac))))]
        return {
            "n": n,
            "mean": sum(xs) / n,
            "median": q(0.5),
            "p90": q(0.9),
            "max": xs[-1],
        }

    def on_stop(self):
        # Base closes positions and deletes remaining pendings. Label those as end-of-test.
        old = self._delete_reason_hint
        self._delete_reason_hint = "end_of_test"
        try:
            return super().on_stop()
        finally:
            self._delete_reason_hint = old

    def summary(self):
        x = super().summary()
        x["strategy_id"] = "GoldeBrave_v4_plus_BigPlayerDetector_parity_v9_pending_lifecycle_audit"
        x["verification_level"] = "NAUTILUS_RAW_BIDASK_MT5_PARITY_V9_PENDING_LIFECYCLE_AUDIT"
        x["pending_lifecycle"] = {
            "execution_changed": False,
            "created_by_layer": dict(self.pending_created_by_layer),
            "fills_by_layer": dict(self.pending_fill_by_layer),
            "terminal_counts": dict(self.pending_terminal),
            "terminal_by_layer": dict(self.pending_terminal_by_layer),
            "lifetime_seconds_by_layer": {
                k: self._stats(v) for k, v in self.pending_lifetime_s_by_layer.items()
            },
            "closest_gap_price_by_layer": {
                k: self._stats(v) for k, v in self.pending_closest_gap_by_layer.items()
            },
            "open_meta_remaining": len(self._pending_meta),
            "interpretation": (
                "closest_gap is executable-side distance remaining before stop trigger; "
                "<=0 indicates the order price was crossed on a raw quote."
            ),
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
    st = ParityV9Strategy(GoldeBraveConfig(
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
