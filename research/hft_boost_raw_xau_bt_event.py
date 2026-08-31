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

from research.hft_boost_raw_xau_bt import HFTBaseConfig, HFTBaseStrategy, metrics

SIM = Venue("SIM")


def _query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
    try:
        return self.query(QuoteTick, identifiers=identifiers, start=start, end=end, **kwargs)
    except TypeError:
        return self.query(QuoteTick, instrument_ids=identifiers, start=start, end=end, **kwargs)


if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks


class HFTBaseEventStrategy(HFTBaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.closed_trades = []

    def on_position_closed(self, event):
        pnl = getattr(event, "realized_pnl", None)
        if pnl is None:
            pnl_value = 0.0
        elif hasattr(pnl, "as_double"):
            pnl_value = float(pnl.as_double())
        elif hasattr(pnl, "as_decimal"):
            pnl_value = float(pnl.as_decimal())
        else:
            try:
                pnl_value = float(str(pnl).replace(",", "").split()[0])
            except Exception:
                pnl_value = 0.0
        ts = getattr(event, "ts_closed", None) or getattr(event, "ts_event", 0)
        self.closed_trades.append({"pnl": pnl_value, "ts_closed": int(ts or 0)})
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
        "edge": "HFT_BOOST_BASE_v0.4-event-ledger",
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "data_kind": "RAW_BIDASK QuoteTick",
        "ohlc_resample_used": False,
        "execution": "Nautilus native MARKET orders; native Bid/Ask spread; PnL captured from PositionClosed.realized_pnl; no explicit fee/slippage yet",
        "period": {"start": manifest["start"], "days": days, "end_exclusive": manifest["end_exclusive"]},
        "raw_ticks": len(ticks),
        "point": point,
        "signals": strat.signal_count,
        "entries_submitted": strat.entries,
        "closed_positions": len(trades),
        "avg_signal_score": strat.score_sum / strat.signal_count if strat.signal_count else 0.0,
        "params": {"min_score": args.min_score, "tp_points": args.tp_points, "sl_points": args.sl_points, "trade_size": args.trade_size},
        "kpi": k,
        "limitations": ["Floating mark-to-market DD/MAE/MFE and explicit commission/slippage are next Reality Gate."],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True)
    (outdir / "summary.json").write_text(text)
    (outdir / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(text)
    eng.dispose()


if __name__ == "__main__":
    main()
