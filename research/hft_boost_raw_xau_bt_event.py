from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.hft_boost_raw_xau_bt import HFTBaseConfig, HFTBaseStrategy, metrics

SIM = Venue("SIM")


def _query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
    try:
        return self.query(QuoteTick, identifiers=identifiers, start=start, end=end, **kwargs)
    except TypeError:
        return self.query(QuoteTick, instrument_ids=identifiers, start=start, end=end, **kwargs)


if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks


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


def _price_float(v):
    if v is None:
        return None
    if hasattr(v, "as_double"):
        return float(v.as_double())
    if hasattr(v, "as_decimal"):
        return float(v.as_decimal())
    try:
        return float(v)
    except Exception:
        try:
            return float(str(v))
        except Exception:
            return None


def _event_reason(event):
    for name in ("reason", "message", "error", "info"):
        value = getattr(event, name, None)
        if value is not None:
            return str(value)
    return repr(event)


class HFTBaseEventStrategy(HFTBaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.closed_trades = []
        self.entry_pending = False
        self.pending_side = None
        self.order_fills = 0
        self.order_rejects = 0
        self.order_denials = 0
        self.order_cancels = 0

    def _clear_entry_pending(self):
        self.entry_pending = False
        self.pending_side = None

    def on_quote_tick(self, tick: QuoteTick):
        m = self._micro(tick)
        if m is None:
            return
        ts_ms, bid, ask, *_ = m
        if self.entry_price is not None and not self.exit_pending:
            signed = 1 if self.entry_side == "buy" else -1
            mark = bid if self.entry_side == "buy" else ask
            dpts = signed * (mark - self.entry_price) / self.config.point
            held = ts_ms - self.entry_ts_ms
            if dpts >= self.config.tp_points or dpts <= -self.config.sl_points or held >= self.config.max_hold_ms:
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
        side, _ = s
        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            quantity=instrument.make_qty(self.config.trade_size),
        )
        self.pending_side = side
        self.entry_pending = True
        self.entries += 1
        self.submit_order(order)

    def on_order_filled(self, event):
        self.order_fills += 1
        if self.entry_pending and self.entry_price is None:
            px = _price_float(getattr(event, "last_px", None))
            if px is None:
                px = _price_float(getattr(event, "avg_px", None))
            if px is None:
                px = _price_float(getattr(event, "price", None))
            self.entry_price = px
            self.entry_side = self.pending_side
            self.entry_ts_ms = int(getattr(event, "ts_event", 0) // 1_000_000)
            self._clear_entry_pending()

    def on_order_rejected(self, event):
        self.order_rejects += 1
        reason = _event_reason(event)
        print("FIRST_ORDER_REJECT_REASON=" + reason, flush=True)
        print("FIRST_ORDER_REJECT_EVENT=" + repr(event), flush=True)
        raise RuntimeError("HFT_ORDER_REJECT_DIAGNOSTIC: " + reason)

    def on_order_denied(self, event):
        self.order_denials += 1
        self._clear_entry_pending()
        self.exit_pending = False

    def on_order_canceled(self, event):
        self.order_cancels += 1
        self._clear_entry_pending()
        self.exit_pending = False

    def on_position_closed(self, event):
        pnl_value = _money_float(getattr(event, "realized_pnl", None))
        ts = getattr(event, "ts_closed", None) or getattr(event, "ts_event", 0)
        self.closed_trades.append({"pnl": pnl_value, "ts_closed": int(ts or 0)})
        self._clear_entry_pending()
        super().on_position_closed(event)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--raw-bidask-only", action="store_true")
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--min-score", type=float, default=62.0)
    ap.add_argument("--tp-points", type=float, default=8.0)
    ap.add_argument("--sl-points", type=float, default=10.0)
    ap.add_argument("--trade-size", default="1")
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
    ticks = catalog.query_quote_ticks(identifiers=[instrument.id.value])
    if not ticks:
        raise SystemExit("no XAUUSD raw QuoteTicks")
    try:
        point = float(instrument.price_increment.as_double())
    except Exception:
        point = float(str(instrument.price_increment))

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
    strat = HFTBaseEventStrategy(HFTBaseConfig(
        instrument_id=instrument.id,
        trade_size=Decimal(args.trade_size),
        point=point,
        min_score=args.min_score,
        tp_points=args.tp_points,
        sl_points=args.sl_points,
    ))
    eng.add_strategy(strat)
    eng.run()

    trades = strat.closed_trades
    k = metrics(trades, args.initial, days)
    outdir = Path("results/ae-bt") / args.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(outdir / "trades.csv", index=False)
    summary = {
        "verification_level": "NAUTILUS_BT_RAW_BIDASK",
        "edge": "HFT_BOOST_BASE_v0.6-reject-diagnostic",
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "data_kind": "RAW_BIDASK QuoteTick",
        "ohlc_resample_used": False,
        "period": {"start": manifest["start"], "days": days, "end_exclusive": manifest["end_exclusive"]},
        "raw_ticks": len(ticks),
        "signals": strat.signal_count,
        "entries_submitted": strat.entries,
        "order_fills": strat.order_fills,
        "order_rejects": strat.order_rejects,
        "closed_positions": len(trades),
        "kpi": k,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True)
    (outdir / "summary.json").write_text(text)
    print(text)
    eng.dispose()


if __name__ == "__main__":
    main()
