from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.hft_boost_raw_xau_bt import HFTBaseConfig, metrics
from research.hft_boost_raw_xau_bt_event import HFTBaseEventStrategy, _price_float, _money_float

SIM = Venue("SIM")


class HFTBaseV08(HFTBaseEventStrategy):
    """Fill-driven HFT base with executable-cost normalized exits.

    Modes:
      fixed:      legacy TP/SL points.
      costnorm:   TP/SL floors scale from entry spread.
      peak:       costnorm + no-red-after-green / peak retrace exit.
    """

    def __init__(self, config, mode: str, target_spread_mult: float, stop_spread_mult: float,
                 green_arm_mult: float, retrace_mult: float):
        super().__init__(config)
        self.mode = mode
        self.target_spread_mult = target_spread_mult
        self.stop_spread_mult = stop_spread_mult
        self.green_arm_mult = green_arm_mult
        self.retrace_mult = retrace_mult
        self.entry_spread_points = 0.0
        self.dynamic_tp_points = config.tp_points
        self.dynamic_sl_points = config.sl_points
        self.max_profit_points = 0.0
        self.cost_gate_rejects = 0
        self.exit_tp = 0
        self.exit_sl = 0
        self.exit_time = 0
        self.exit_peak = 0
        self.exit_red_after_green = 0

    def _reset_trade_state(self):
        self.entry_spread_points = 0.0
        self.dynamic_tp_points = self.config.tp_points
        self.dynamic_sl_points = self.config.sl_points
        self.max_profit_points = 0.0

    def on_quote_tick(self, tick: QuoteTick):
        m = self._micro(tick)
        if m is None:
            return
        ts_ms, bid, ask, velocity, imbalance, spread, exhaustion = m

        if self.entry_price is not None and not self.exit_pending:
            signed = 1 if self.entry_side == "buy" else -1
            mark = bid if self.entry_side == "buy" else ask
            dpts = signed * (mark - self.entry_price) / self.config.point
            held = ts_ms - self.entry_ts_ms
            self.max_profit_points = max(self.max_profit_points, dpts)

            reason = None
            if dpts >= self.dynamic_tp_points:
                reason = "tp"; self.exit_tp += 1
            elif dpts <= -self.dynamic_sl_points:
                reason = "sl"; self.exit_sl += 1
            elif self.mode == "peak":
                green_arm = max(self.entry_spread_points * self.green_arm_mult, self.config.tp_points * 0.50)
                retrace = max(self.entry_spread_points * self.retrace_mult, self.config.tp_points * 0.35)
                if self.max_profit_points >= green_arm and dpts < 0:
                    reason = "red_after_green"; self.exit_red_after_green += 1
                elif self.max_profit_points >= green_arm and dpts <= self.max_profit_points - retrace:
                    reason = "peak"; self.exit_peak += 1
            if reason is None and held >= self.config.max_hold_ms:
                reason = "time"; self.exit_time += 1
            if reason is not None:
                self.close_all_positions(self.config.instrument_id)
                self.exit_pending = True
                return

        if self.entry_price is not None or self.entry_pending or self.exit_pending:
            return
        if ts_ms - self.last_exit_ts_ms < self.config.cooldown_ms:
            return

        s = self._signal(m)
        if s is None:
            return
        side, score = s

        # Hard executable-cost gate. A target smaller than spread is structurally unattractive.
        if self.mode != "fixed" and spread <= 0:
            self.cost_gate_rejects += 1
            return

        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            quantity=instrument.make_qty(self.config.trade_size),
        )
        self.entry_spread_points = spread
        if self.mode == "fixed":
            self.dynamic_tp_points = self.config.tp_points
            self.dynamic_sl_points = self.config.sl_points
        else:
            self.dynamic_tp_points = max(self.config.tp_points, spread * self.target_spread_mult)
            self.dynamic_sl_points = max(self.config.sl_points, spread * self.stop_spread_mult)
        self.max_profit_points = 0.0
        self.pending_side = side
        self.entry_pending = True
        self.entries += 1
        self.submit_order(order)

    def on_order_rejected(self, event):
        self.order_rejects += 1
        self._clear_entry_pending()
        self.exit_pending = False
        self._reset_trade_state()

    def on_order_denied(self, event):
        super().on_order_denied(event)
        self._reset_trade_state()

    def on_order_canceled(self, event):
        super().on_order_canceled(event)
        self._reset_trade_state()

    def on_position_closed(self, event):
        pnl_value = _money_float(getattr(event, "realized_pnl", None))
        ts = getattr(event, "ts_closed", None) or getattr(event, "ts_event", 0)
        self.closed_trades.append({"pnl": pnl_value, "ts_closed": int(ts or 0)})
        self._clear_entry_pending()
        # Call grandparent position-close reset once; avoid duplicate ledger append.
        try:
            ts_ms = int(event.ts_event // 1_000_000)
        except Exception:
            ts_ms = self.entry_ts_ms or 0
        self.last_exit_ts_ms = ts_ms
        self.entry_price = None
        self.entry_side = None
        self.entry_ts_ms = None
        self.exit_pending = False
        self._reset_trade_state()


def query_ticks(catalog, instrument_id):
    try:
        return catalog.query(QuoteTick, identifiers=[instrument_id])
    except TypeError:
        return catalog.query(QuoteTick, instrument_ids=[instrument_id])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--raw-bidask-only", action="store_true")
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--trade-size", default="1")
    ap.add_argument("--min-score", type=float, default=62.0)
    ap.add_argument("--tp-points", type=float, default=8.0)
    ap.add_argument("--sl-points", type=float, default=10.0)
    ap.add_argument("--mode", choices=["fixed", "costnorm", "peak"], default="costnorm")
    ap.add_argument("--target-spread-mult", type=float, default=2.0)
    ap.add_argument("--stop-spread-mult", type=float, default=1.5)
    ap.add_argument("--green-arm-mult", type=float, default=1.0)
    ap.add_argument("--retrace-mult", type=float, default=0.75)
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit("raw-bidask-only mandatory")

    cp = Path(args.catalog)
    manifest = json.loads((cp / "catalog_manifest.json").read_text())
    days = int(manifest["days"])
    catalog = ParquetDataCatalog(str(cp))
    instrument = {x.id.symbol.value.replace("/", ""): x for x in catalog.instruments()}.get("XAUUSD")
    if instrument is None:
        raise SystemExit("XAUUSD missing")
    ticks = query_ticks(catalog, instrument.id.value)
    if not ticks:
        raise SystemExit("no XAUUSD raw QuoteTicks")
    try:
        point = float(instrument.price_increment.as_double())
    except Exception:
        point = float(str(instrument.price_increment))

    # Empirical raw spread diagnostics, in instrument points.
    sample = ticks[:min(len(ticks), 200000)]
    spreads = [max(0.0, (_price_float(t.ask_price) - _price_float(t.bid_price)) / point) for t in sample]
    spreads_sorted = sorted(spreads)
    def q(frac):
        if not spreads_sorted: return 0.0
        return spreads_sorted[min(len(spreads_sorted)-1, int(frac*(len(spreads_sorted)-1)))]

    eng = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="ERROR"),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    eng.add_venue(
        venue=SIM,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(args.initial, USD)],
        default_leverage=Decimal("2000"),
    )
    eng.add_instrument(instrument)
    eng.add_data(ticks)
    strat = HFTBaseV08(
        HFTBaseConfig(
            instrument_id=instrument.id,
            trade_size=Decimal(args.trade_size),
            point=point,
            min_score=args.min_score,
            tp_points=args.tp_points,
            sl_points=args.sl_points,
        ),
        mode=args.mode,
        target_spread_mult=args.target_spread_mult,
        stop_spread_mult=args.stop_spread_mult,
        green_arm_mult=args.green_arm_mult,
        retrace_mult=args.retrace_mult,
    )
    eng.add_strategy(strat)
    eng.run()

    trades = strat.closed_trades
    k = metrics(trades, args.initial, days)
    outdir = Path("results/ae-bt") / args.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(outdir / "trades.csv", index=False)
    summary = {
        "verification_level": "NAUTILUS_BT_RAW_BIDASK",
        "edge": "HFT_BOOST_BASE_v0.8-cost-normalized-ablation",
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "data_kind": "RAW_BIDASK QuoteTick",
        "ohlc_resample_used": False,
        "period": {"start": manifest["start"], "days": days, "end_exclusive": manifest["end_exclusive"]},
        "raw_ticks": len(ticks),
        "point": point,
        "spread_points_sample": {"n": len(spreads), "p50": q(.50), "p90": q(.90), "p99": q(.99)},
        "signals": strat.signal_count,
        "entries_submitted": strat.entries,
        "order_fills": strat.order_fills,
        "order_rejects": strat.order_rejects,
        "closed_positions": len(trades),
        "cost_gate_rejects": strat.cost_gate_rejects,
        "exit_counts": {
            "tp": strat.exit_tp,
            "sl": strat.exit_sl,
            "time": strat.exit_time,
            "peak": strat.exit_peak,
            "red_after_green": strat.exit_red_after_green,
        },
        "params": {
            "mode": args.mode,
            "min_score": args.min_score,
            "base_tp_points": args.tp_points,
            "base_sl_points": args.sl_points,
            "target_spread_mult": args.target_spread_mult,
            "stop_spread_mult": args.stop_spread_mult,
            "green_arm_mult": args.green_arm_mult,
            "retrace_mult": args.retrace_mult,
            "trade_size": args.trade_size,
        },
        "kpi": k,
        "limitations": [
            "Closed-equity DD only; floating MTM DD/MAE/MFE remain Reality Gate work.",
            "No explicit commission/slippage yet; native Bid/Ask spread is present.",
            "This is a cost-normalization/exit ablation, not final production validation.",
        ],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True)
    (outdir / "summary.json").write_text(text)
    (outdir / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(text)
    eng.dispose()


if __name__ == "__main__":
    main()
