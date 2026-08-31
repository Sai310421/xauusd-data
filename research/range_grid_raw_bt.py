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
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from research.hft_boost_raw_xau_bt import metrics

SIM = Venue("SIM")
VARIANTS = ("equal_grid", "linear_nanpin", "atr_grid", "sqrt_nanpin", "protected_recovery")


def _money_float(v):
    if v is None:
        return 0.0
    if hasattr(v, "as_double"):
        return float(v.as_double())
    if hasattr(v, "as_decimal"):
        return float(v.as_decimal())
    try:
        return float(str(v).replace(",", "").split()[0])
    except Exception:
        return 0.0


def _px(v):
    if hasattr(v, "as_double"):
        return float(v.as_double())
    if hasattr(v, "as_decimal"):
        return float(v.as_decimal())
    return float(v)


def _qty(v):
    if hasattr(v, "as_double"):
        return float(v.as_double())
    if hasattr(v, "as_decimal"):
        return float(v.as_decimal())
    try:
        return float(v)
    except Exception:
        return float(str(v))


def query_ticks(catalog, instrument_id, start, end):
    try:
        return catalog.query(QuoteTick, identifiers=[instrument_id], start=start, end=end)
    except TypeError:
        return catalog.query(QuoteTick, instrument_ids=[instrument_id], start=start, end=end)


class RangeGridConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    point: float
    variant: str
    base_qty: Decimal = Decimal("0.01")
    max_layers: int = 5
    min_range_score: float = 62.0
    entry_z: float = 1.15
    basket_tp_points: float = 5.0
    max_hold_ms: int = 120000
    cooldown_ms: int = 500
    hard_tail_points: float = 55.0


class RangeGridStrategy(Strategy):
    def __init__(self, config: RangeGridConfig):
        super().__init__(config)
        self.mid = deque(maxlen=240)
        self.spreads = deque(maxlen=240)
        self.prev_mid = None
        self.path = deque(maxlen=239)
        self.basket_side = None
        self.avg_entry = None
        self.net_qty = 0.0
        self.layers = 0
        self.last_add_price = None
        self.basket_start_ms = 0
        self.last_exit_ms = -10**18
        self.order_pending = False
        self.pending_kind = None
        self.pending_side = None
        self.pending_qty = 0.0
        self.exit_pending = False
        self.closed_trades = []
        self.order_fills = 0
        self.order_rejects = 0
        self.order_denials = 0
        self.entries_submitted = 0
        self.adds_submitted = 0
        self.range_ticks = 0
        self.transition_ticks = 0
        self.max_layers_seen = 0
        self.max_adverse_points = 0.0
        self.range_score_sum = 0.0
        self.range_score_n = 0

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def _state(self, tick):
        bid = _px(tick.bid_price)
        ask = _px(tick.ask_price)
        mid = (bid + ask) * 0.5
        ts_ms = int(tick.ts_event // 1_000_000)
        if self.prev_mid is not None:
            self.path.append(abs(mid - self.prev_mid))
        self.prev_mid = mid
        self.mid.append(mid)
        self.spreads.append(max(0.0, (ask - bid) / self.config.point))
        if len(self.mid) < 80:
            return None
        a = np.asarray(self.mid, dtype=float)
        center = float(a.mean())
        std = float(a.std())
        span_pts = float((a.max() - a.min()) / self.config.point)
        path_sum = float(sum(self.path))
        drift = abs(float(a[-1] - a[0]))
        efficiency = drift / max(path_sum, self.config.point)
        z = (mid - center) / max(std, self.config.point * 0.5)
        vol_pts = float(np.mean(np.asarray(self.path, dtype=float)) / self.config.point) if self.path else 0.0
        spread_pts = float(np.mean(self.spreads))
        half = max(1, len(a) // 2)
        slope_pts = abs(float(a[-1] - a[-half])) / self.config.point
        score = 100.0
        score -= min(55.0, efficiency * 110.0)
        score -= min(20.0, max(0.0, spread_pts - 4.0) * 2.0)
        score -= min(20.0, max(0.0, slope_pts - 10.0) * 0.7)
        if span_pts < 8.0:
            score -= 18.0
        elif span_pts <= 80.0:
            score += 5.0
        score = max(0.0, min(100.0, score))
        self.range_score_sum += score
        self.range_score_n += 1
        if score >= self.config.min_range_score:
            self.range_ticks += 1
        else:
            self.transition_ticks += 1
        return ts_ms, bid, ask, mid, z, vol_pts, spread_pts, score, span_pts

    def _layer_mult(self, next_layer):
        v = self.config.variant
        if v == "linear_nanpin":
            return 1.0 + 0.25 * (next_layer - 1)
        if v == "sqrt_nanpin":
            return math.sqrt(next_layer)
        if v == "protected_recovery":
            return 1.0 + 0.12 * max(0, next_layer - 2)
        return 1.0

    def _spacing(self, vol_pts, next_layer):
        v = self.config.variant
        if v in ("equal_grid", "linear_nanpin"):
            return 8.0
        if v == "atr_grid":
            return max(6.0, min(22.0, vol_pts * 7.0))
        if v == "sqrt_nanpin":
            return max(6.0, min(18.0, vol_pts * 5.0))
        return max(7.0, min(28.0, (vol_pts * 6.0 + 4.0) * (1.0 + 0.18 * (next_layer - 1))))

    def _submit(self, side, qty, kind):
        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            quantity=instrument.make_qty(Decimal(str(qty))),
        )
        self.order_pending = True
        self.pending_kind = kind
        self.pending_side = side
        self.pending_qty = qty
        self.entries_submitted += 1
        if kind == "add":
            self.adds_submitted += 1
        self.submit_order(order)

    def _close(self):
        if not self.exit_pending:
            self.exit_pending = True
            self.close_all_positions(self.config.instrument_id)

    def on_quote_tick(self, tick: QuoteTick):
        s = self._state(tick)
        if s is None:
            return
        ts_ms, bid, ask, mid, z, vol_pts, spread_pts, score, _ = s
        if self.basket_side is not None and self.avg_entry is not None:
            signed = 1.0 if self.basket_side == "buy" else -1.0
            mark = bid if self.basket_side == "buy" else ask
            basket_pts = signed * (mark - self.avg_entry) / self.config.point
            adverse = max(0.0, -basket_pts)
            self.max_adverse_points = max(self.max_adverse_points, adverse)
            held = ts_ms - self.basket_start_ms
            if basket_pts >= self.config.basket_tp_points or adverse >= self.config.hard_tail_points or held >= self.config.max_hold_ms:
                self._close(); return
            if score < 35.0 and adverse >= 16.0:
                self._close(); return
            if self.order_pending or self.exit_pending or self.layers >= self.config.max_layers:
                return
            if score < self.config.min_range_score:
                return
            next_layer = self.layers + 1
            spacing = self._spacing(vol_pts, next_layer)
            if self.basket_side == "buy":
                trigger = self.last_add_price - spacing * self.config.point
                if ask <= trigger:
                    qty = float(self.config.base_qty) * self._layer_mult(next_layer)
                    self._submit("buy", qty, "add")
            else:
                trigger = self.last_add_price + spacing * self.config.point
                if bid >= trigger:
                    qty = float(self.config.base_qty) * self._layer_mult(next_layer)
                    self._submit("sell", qty, "add")
            return

        if self.order_pending or self.exit_pending or ts_ms - self.last_exit_ms < self.config.cooldown_ms:
            return
        if score < self.config.min_range_score or spread_pts > 12.0:
            return
        if z <= -self.config.entry_z:
            self._submit("buy", float(self.config.base_qty), "seed")
        elif z >= self.config.entry_z:
            self._submit("sell", float(self.config.base_qty), "seed")

    def on_order_filled(self, event):
        self.order_fills += 1
        if not self.order_pending or self.pending_kind not in ("seed", "add"):
            return
        fill_px = _px(getattr(event, "last_px", getattr(event, "price", 0)))
        fill_qty = _qty(getattr(event, "last_qty", self.pending_qty))
        if self.basket_side is None:
            self.basket_side = self.pending_side
            self.avg_entry = fill_px
            self.net_qty = fill_qty
            self.layers = 1
            self.basket_start_ms = int(getattr(event, "ts_event", 0) // 1_000_000)
        else:
            total = self.net_qty + fill_qty
            self.avg_entry = (self.avg_entry * self.net_qty + fill_px * fill_qty) / max(total, 1e-12)
            self.net_qty = total
            self.layers += 1
        self.last_add_price = fill_px
        self.max_layers_seen = max(self.max_layers_seen, self.layers)
        self.order_pending = False
        self.pending_kind = None

    def on_order_rejected(self, event):
        self.order_rejects += 1
        self.order_pending = False
        self.pending_kind = None

    def on_order_denied(self, event):
        self.order_denials += 1
        self.order_pending = False
        self.pending_kind = None

    def on_position_closed(self, event):
        pnl = _money_float(getattr(event, "realized_pnl", None))
        ts = int(getattr(event, "ts_closed", None) or getattr(event, "ts_event", 0) or 0)
        self.closed_trades.append({"pnl": pnl, "ts_closed": ts})
        self.last_exit_ms = int(ts // 1_000_000) if ts > 10**12 else ts
        self.basket_side = None
        self.avg_entry = None
        self.net_qty = 0.0
        self.layers = 0
        self.last_add_price = None
        self.basket_start_ms = 0
        self.order_pending = False
        self.pending_kind = None
        self.exit_pending = False

    def on_stop(self):
        if self.basket_side is not None:
            self.close_all_positions(self.config.instrument_id)


def run_one(args):
    cp = Path(args.catalog)
    catalog = ParquetDataCatalog(str(cp))
    instrument = {x.id.symbol.value.replace("/", ""): x for x in catalog.instruments()}.get("XAUUSD")
    if instrument is None:
        raise SystemExit("XAUUSD missing")
    ticks = query_ticks(catalog, instrument.id.value, args.start, args.end)
    outdir = Path("results/ae-bt") / args.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    if not ticks:
        pd.DataFrame(columns=["pnl", "ts_closed"]).to_csv(outdir / "trades.csv", index=False)
        summary = {"verification_level":"NAUTILUS_BT_RAW_BIDASK_RANGE_GRID","variant":args.variant,"empty_shard":True,"raw_ticks":0,"kpi":metrics([], args.initial, 1)}
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2)); return
    point = _px(instrument.price_increment)
    eng = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"), risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN, base_currency=USD, starting_balances=[Money(args.initial, USD)], default_leverage=Decimal("2000"))
    eng.add_instrument(instrument); eng.add_data(ticks)
    strat = RangeGridStrategy(RangeGridConfig(instrument_id=instrument.id, point=point, variant=args.variant, base_qty=Decimal(args.base_qty), max_layers=args.max_layers))
    eng.add_strategy(strat); eng.run()
    k = metrics(strat.closed_trades, args.initial, 1)
    pd.DataFrame(strat.closed_trades).to_csv(outdir / "trades.csv", index=False)
    summary = {"verification_level":"NAUTILUS_BT_RAW_BIDASK_RANGE_GRID","engine":"NautilusTrader BacktestEngine","nautilus_version":getattr(nautilus_trader,"__version__","unknown"),"data_kind":"RAW_BIDASK QuoteTick","ohlc_resample_used":False,"variant":args.variant,"empty_shard":False,"period":{"start":args.start,"end":args.end},"raw_ticks":len(ticks),"order_fills":strat.order_fills,"order_rejects":strat.order_rejects,"order_denials":strat.order_denials,"entries_submitted":strat.entries_submitted,"adds_submitted":strat.adds_submitted,"closed_baskets":len(strat.closed_trades),"max_layers_seen":strat.max_layers_seen,"max_adverse_points":strat.max_adverse_points,"range_tick_ratio":strat.range_ticks/max(strat.range_ticks+strat.transition_ticks,1),"avg_range_score":strat.range_score_sum/max(strat.range_score_n,1),"kpi":k,"params":{"base_qty":args.base_qty,"max_layers":args.max_layers,"min_range_score":62,"entry_z":1.15,"basket_tp_points":5,"hard_tail_points":55}}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True)); print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True)); eng.dispose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True); ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True); ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--base-qty", default="0.01"); ap.add_argument("--max-layers", type=int, default=5)
    args = ap.parse_args(); run_one(args)

if __name__ == "__main__": main()
