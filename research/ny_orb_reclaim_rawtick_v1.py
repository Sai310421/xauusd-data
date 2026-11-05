#!/usr/bin/env python3
"""Raw Bid/Ask NY compressed opening-range break/pullback/reclaim research."""
from __future__ import annotations
import argparse, json
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

NS = 1_000_000_000
NY = ZoneInfo("America/New_York")

def f(x):
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)

def ema(values, period):
    if len(values) < period:
        return None
    v = float(np.mean(values[:period]))
    k = 2.0 / (period + 1.0)
    for x in values[period:]:
        v = k * float(x) + (1.0 - k) * v
    return v

class Config(StrategyConfig, frozen=True):
    instrument_id: object

class NyOrbReclaim(Strategy):
    """One maximum trade per NY day; all decisions use completed bars/ticks only."""
    def __init__(self, config):
        super().__init__(config)
        self.m1_bucket = self.m1_work = None
        self.m15_bucket = self.m15_work = None
        self.m15_close = deque(maxlen=300)
        self.trend = 0
        self.day = None
        self.or_high = self.or_low = None
        self.or_final = False
        self.compressed = False
        self.or_history = deque(maxlen=20)
        self.state = "idle"
        self.side = 0
        self.pulled_back = False
        self.traded = False
        self.active = None
        self.last_bid = self.last_ask = None
        self.ticks = 0
        self.signals = 0
        self.trades = []

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def _local(self, ts):
        return datetime.fromtimestamp(ts // NS, tz=timezone.utc).astimezone(NY)

    def _trend(self):
        c = list(self.m15_close)
        fast, slow, prev_fast = ema(c, 20), ema(c, 50), ema(c[:-1], 20)
        if fast is None or slow is None or prev_fast is None:
            self.trend = 0
        elif fast > slow and fast > prev_fast:
            self.trend = 1
        elif fast < slow and fast < prev_fast:
            self.trend = -1
        else:
            self.trend = 0

    def _new_day(self, local_day):
        if local_day == self.day:
            return
        self.day = local_day
        self.or_high = self.or_low = None
        self.or_final = self.compressed = False
        self.state, self.side, self.pulled_back = "idle", 0, False
        self.traded = False

    def _on_m1_close(self, bar):
        dt = self._local(bar["ts"])
        self._new_day(dt.date().isoformat())
        minute = dt.hour * 60 + dt.minute
        # Fixed 08:00-08:30 New York opening range.  The decision at 08:30 uses
        # only the now-completed range, never the current/future candle.
        if 8 * 60 <= minute < 8 * 60 + 30:
            self.or_high = bar["h"] if self.or_high is None else max(self.or_high, bar["h"])
            self.or_low = bar["l"] if self.or_low is None else min(self.or_low, bar["l"])
        elif minute == 8 * 60 + 30 and self.or_high is not None and not self.or_final:
            width = self.or_high - self.or_low
            base = float(np.median(self.or_history)) if len(self.or_history) >= 20 else None
            self.compressed = base is not None and width > 0 and width <= 0.85 * base
            self.or_history.append(width)
            self.or_final = True
            self.state = "wait_break" if self.compressed and self.trend != 0 else "idle"

    def _finish(self, mark, why):
        x = self.active
        move = (mark - x["entry"]) * x["side"]
        risk = abs(x["entry"] - x["stop"])
        self.trades.append({
            "ny_day": self.day, "side": "BUY" if x["side"] > 0 else "SELL",
            "pnl_usd": move * 2.0, "r": move / risk if risk else None, "why": why,
            "entry": x["entry"], "stop": x["stop"], "take": x["take"],
            "risk_price": risk, "mae_price": x["mae"], "mfe_price": x["mfe"],
            "mae_R": x["mae"] / risk if risk else None,
            "mfe_R": x["mfe"] / risk if risk else None,
        })
        self.active = None

    def _enter(self, side, bid, ask):
        width = self.or_high - self.or_low
        entry = ask if side > 0 else bid
        stop = self.or_low - 0.15 * width if side > 0 else self.or_high + 0.15 * width
        risk = abs(entry - stop)
        if risk <= 0:
            return
        self.active = {"side": side, "entry": entry, "stop": stop,
                       "take": entry + side * 2.0 * risk, "mae": 0.0, "mfe": 0.0}
        self.traded = True
        self.state = "done"
        self.signals += 1

    def _evaluate(self, bid, ask, ts):
        if self.active:
            x = self.active
            mark = bid if x["side"] > 0 else ask
            move = (mark - x["entry"]) * x["side"]
            x["mae"], x["mfe"] = min(x["mae"], move), max(x["mfe"], move)
            stopped = mark <= x["stop"] if x["side"] > 0 else mark >= x["stop"]
            taken = mark >= x["take"] if x["side"] > 0 else mark <= x["take"]
            if stopped or taken:
                self._finish(mark, "TP" if taken else "SL")
            return
        if not self.or_final or not self.compressed or self.traded or self.state == "idle":
            return
        dt = self._local(ts)
        minute = dt.hour * 60 + dt.minute
        if minute >= 11 * 60:
            self.state = "done"
            return
        width = self.or_high - self.or_low
        if self.state == "wait_break":
            if self.trend > 0 and bid >= self.or_high + 0.10 * width:
                self.side, self.state = 1, "wait_pullback"
            elif self.trend < 0 and ask <= self.or_low - 0.10 * width:
                self.side, self.state = -1, "wait_pullback"
        elif self.state == "wait_pullback":
            if self.side > 0:
                if bid <= self.or_high + 0.15 * width:
                    self.pulled_back = True
                if self.pulled_back and bid >= self.or_high + 0.10 * width:
                    self._enter(1, bid, ask)
            else:
                if ask >= self.or_low - 0.15 * width:
                    self.pulled_back = True
                if self.pulled_back and ask <= self.or_low - 0.10 * width:
                    self._enter(-1, bid, ask)

    def on_quote_tick(self, tick):
        self.ticks += 1
        bid, ask, ts = f(tick.bid_price), f(tick.ask_price), tick.ts_event
        self.last_bid, self.last_ask = bid, ask
        for minutes, prefix in ((1, "m1"), (15, "m15")):
            bucket = ts // (minutes * 60 * NS)
            battr, wattr = prefix + "_bucket", prefix + "_work"
            old, work = getattr(self, battr), getattr(self, wattr)
            if old is None:
                setattr(self, battr, bucket)
                setattr(self, wattr, {"o": bid, "h": bid, "l": bid, "c": bid, "ts": ts})
            elif bucket != old:
                if prefix == "m1":
                    self._on_m1_close(work)
                else:
                    self.m15_close.append(work["c"])
                    self._trend()
                setattr(self, battr, bucket)
                setattr(self, wattr, {"o": bid, "h": bid, "l": bid, "c": bid, "ts": ts})
            else:
                work["h"], work["l"], work["c"] = max(work["h"], bid), min(work["l"], bid), bid
        self._evaluate(bid, ask, ts)

    def on_stop(self):
        if self.active:
            self._finish(self.last_bid if self.active["side"] > 0 else self.last_ask, "END")

    def result(self):
        pnl = np.asarray([x["pnl_usd"] for x in self.trades], dtype=float)
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        equity = peak = 1000.0
        dd = 0.0
        for x in pnl:
            equity += x
            peak = max(peak, equity)
            dd = max(dd, peak - equity)
        mae = np.asarray([-x["mae_R"] for x in self.trades], dtype=float)
        mfe = np.asarray([x["mfe_R"] for x in self.trades], dtype=float)
        return {
            "signals": self.signals, "N": len(pnl),
            "WR_pct": float((pnl > 0).mean() * 100) if len(pnl) else 0.0,
            "PF": float(wins.sum() / abs(losses.sum())) if len(losses) else None,
            "EV_usd": float(pnl.mean()) if len(pnl) else 0.0,
            "Net_usd": float(pnl.sum()), "MaxDD_pct": float(dd / peak * 100),
            "MAE_MFE_R": {
                "avg_MAE_R": float(mae.mean()) if len(mae) else 0.0,
                "median_MAE_R": float(np.median(mae)) if len(mae) else 0.0,
                "avg_MFE_R": float(mfe.mean()) if len(mfe) else 0.0,
                "median_MFE_R": float(np.median(mfe)) if len(mfe) else 0.0,
                "MFE_ge_1R_pct": float((mfe >= 1.0).mean() * 100) if len(mfe) else 0.0,
            },
            "trades": self.trades,
        }

def l1(ticks):
    one = Quantity.from_int(1)
    out = []
    for t in ticks:
        if f(t.bid_size) <= 0 or f(t.ask_size) <= 0:
            out.append(QuoteTick(instrument_id=t.instrument_id, bid_price=t.bid_price, ask_price=t.ask_price,
                bid_size=one if f(t.bid_size) <= 0 else t.bid_size,
                ask_size=one if f(t.ask_size) <= 0 else t.ask_size,
                ts_event=t.ts_event, ts_init=t.ts_init))
        else:
            out.append(t)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", action="append", required=True)
    ap.add_argument("--experiment-id", required=True)
    args = ap.parse_args()
    raw, inst = [], None
    for root in args.catalog:
        cat = ParquetDataCatalog(root)
        i = next(x for x in cat.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD")
        inst = inst or i
        raw += cat.query(data_cls=QuoteTick, identifiers=[i.id.value])
    raw.sort(key=lambda x: x.ts_event)
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),
        risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=inst.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        book_type=BookType.L1_MBP, base_currency=USD, starting_balances=[Money(1000, USD)],
        default_leverage=Decimal("2000"))
    engine.add_instrument(inst)
    engine.add_data(l1(raw))
    strategy = NyOrbReclaim(Config(instrument_id=inst.id))
    engine.add_strategy(strategy)
    engine.run()
    engine.end()
    result = {"status": "COMPLETE", "engine": "NautilusTrader BacktestEngine",
        "raw_ticks": len(raw), "ohlc_resample_used": False,
        "execution": "Raw QuoteTick BUY Ask / SELL Bid",
        "hypothesis": "NY 08:00-08:30 compressed opening range; M15 EMA20/EMA50 trend; break, pullback, reclaim",
        "anti_lookahead": "Opening range and M15 trend are read only after their bars close; entry is the qualifying QuoteTick.",
        "parameters": {"compression": "opening range <= 85% median of previous 20 NY opening ranges",
            "break": "0.10 x opening-range width", "pullback": "within 0.15 x range of broken boundary",
            "stop": "opposite boundary +/- 0.15 x range", "take_profit": "2R", "max_trades_per_day": 1},
        "results": {"ny_orb_reclaim_v1": strategy.result()}}
    out = Path("results/ny-orb-reclaim") / args.experiment_id / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
