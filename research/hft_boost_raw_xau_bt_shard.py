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
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.hft_boost_raw_xau_bt import HFTBaseConfig, metrics
from research.hft_boost_raw_xau_bt_event import HFTBaseEventStrategy

SIM = Venue("SIM")


def _query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
    try:
        return self.query(QuoteTick, identifiers=identifiers, start=start, end=end, **kwargs)
    except TypeError:
        return self.query(QuoteTick, instrument_ids=identifiers, start=start, end=end, **kwargs)


if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks


def _normal_reject(self, event):
    self.order_rejects += 1
    self._clear_entry_pending()
    self.exit_pending = False


HFTBaseEventStrategy.on_order_rejected = _normal_reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--start", required=True, help="Inclusive UTC date/time")
    ap.add_argument("--end", required=True, help="Exclusive UTC date/time")
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--min-score", type=float, default=62.0)
    ap.add_argument("--tp-points", type=float, default=8.0)
    ap.add_argument("--sl-points", type=float, default=10.0)
    ap.add_argument("--trade-size", default="1")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if end <= start:
        raise SystemExit("end must be after start")
    days = (end - start).total_seconds() / 86400.0

    cp = Path(args.catalog)
    manifest = json.loads((cp / "catalog_manifest.json").read_text())
    catalog = ParquetDataCatalog(str(cp))
    instrument = {x.id.symbol.value.replace("/", ""): x for x in catalog.instruments()}.get("XAUUSD")
    if instrument is None:
        raise SystemExit("XAUUSD missing")

    ticks = catalog.query_quote_ticks(
        identifiers=[instrument.id.value],
        start=start.isoformat(),
        end=end.isoformat(),
    )
    if not ticks:
        raise SystemExit(f"no XAUUSD raw QuoteTicks in {start}..{end}")

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
    pd.DataFrame(trades, columns=["pnl", "ts_closed"]).to_csv(outdir / "trades.csv", index=False)
    summary = {
        "verification_level": "NAUTILUS_BT_RAW_BIDASK_SHARD",
        "edge": "HFT_BOOST_BASE_v0.7",
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "data_kind": "RAW_BIDASK QuoteTick",
        "ohlc_resample_used": False,
        "period": {"start": start.isoformat(), "days": days, "end_exclusive": end.isoformat()},
        "catalog_period": {"start": manifest["start"], "days": manifest["days"], "end_exclusive": manifest["end_exclusive"]},
        "raw_ticks": len(ticks),
        "signals": strat.signal_count,
        "entries_submitted": strat.entries,
        "order_fills": strat.order_fills,
        "order_rejects": strat.order_rejects,
        "order_denials": strat.order_denials,
        "order_cancels": strat.order_cancels,
        "closed_positions": len(trades),
        "kpi": k,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True)
    (outdir / "summary.json").write_text(text)
    print(text)
    eng.dispose()


if __name__ == "__main__":
    main()
