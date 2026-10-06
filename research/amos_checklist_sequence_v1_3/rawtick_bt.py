from __future__ import annotations

import argparse
import json
import math
from collections import deque
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from research.minimumspike_raw6x3_bt import extract_trades, metrics
import research.minimumspike_raw6x3_bt_compat as _raw_catalog_compat  # patches Nautilus 1.230 QuoteTick reader

SIM = Venue("SIM")
ASIA = (0, 6 * 60)
LONDON = (7 * 60, 10 * 60)
NY = (13 * 60 + 30, 16 * 60)

def in_window(minute: int, window: tuple[int, int]) -> bool:
    return window[0] <= minute < window[1]

class SequenceRawConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    model: str
    target_r: float
    context_min: int
    trade_size: Decimal
    lookback: int = 8
    displacement_atr: float = 0.35
    fib_lo: float = 0.50
    fib_hi: float = 0.62
    require_clear_target: bool = True
    max_setup_bars: int = 30
    max_entries_day: int = 3

class SequenceRawStrategy(Strategy):
    def __init__(self, config: SequenceRawConfig):
        super().__init__(config)
        self.bars = deque(maxlen=180)
        self.day = None
        self.asia_hi = self.asia_lo = None
        self.london_hi = self.london_lo = None
        self.entries_today = 0
        self.seq_state = "IDLE"
        self.age = 0
        self.direction = 0
        self.sweep_extreme = None
        self.protected = None
        self.ref_hi = self.ref_lo = None
        self.sweep_ts = None
        self.mss_ts = None
        self.disp_extreme = None
        self.eq_lo = self.eq_hi = None
        self.eq_touched = False
        self.gaps = []
        self.armed = None
        self.entry_ref = self.stop_ref = self.tp_ref = None
        self.exit_pending = False
        self.diag = {
            "bars": 0, "sweeps": 0, "mss": 0, "eq_touches": 0,
            "eq_rebalance": 0, "ifvg_created": 0, "ifvg_inverted": 0,
            "context_pass": 0, "clear_target_pass": 0, "armed": 0,
            "orders_submitted": 0,
        }

    def on_start(self) -> None:
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    @staticmethod
    def _f(x) -> float:
        return float(x.as_double()) if hasattr(x, "as_double") else float(x)

    @staticmethod
    def _ts(ns: int) -> pd.Timestamp:
        return pd.Timestamp(ns, unit="ns", tz="UTC")

    def _atr14(self, prior=None) -> float | None:
        xs = list(self.bars) if prior is None else prior
        if len(xs) < 15:
            return None
        trs = []
        for i in range(len(xs) - 14, len(xs)):
            cur, prev = xs[i], xs[i - 1]
            trs.append(max(cur["h"] - cur["l"], abs(cur["h"] - prev["c"]), abs(cur["l"] - prev["c"])))
        a = float(np.mean(trs))
        return a if math.isfinite(a) and a > 0 else None

    def _reset_sequence(self) -> None:
        self.seq_state = "IDLE"
        self.age = 0
        self.direction = 0
        self.sweep_extreme = self.protected = None
        self.ref_hi = self.ref_lo = None
        self.sweep_ts = self.mss_ts = None
        self.disp_extreme = None
        self.eq_lo = self.eq_hi = None
        self.eq_touched = False
        self.gaps = []

    def _new_day(self, d) -> None:
        if self.day == d:
            return
        self.day = d
        self.asia_hi = self.asia_lo = None
        self.london_hi = self.london_lo = None
        self.entries_today = 0
        if self.entry_ref is None:
            self._reset_sequence()

    @staticmethod
    def _upd(hi, lo, h, l):
        hi = h if hi is None else max(hi, h)
        lo = l if lo is None else min(lo, l)
        return hi, lo

    def _session_reference(self, minute: int):
        if in_window(minute, LONDON) and self.asia_hi is not None:
            return self.asia_hi, self.asia_lo
        if in_window(minute, NY) and self.london_hi is not None:
            return self.london_hi, self.london_lo
        return None

    def _contexts(self, b) -> tuple[int, dict]:
        xs = list(self.bars)
        htf = False
        if len(xs) >= 60:
            w = xs[-60:]
            hi = max(x["h"] for x in w[:-1])
            lo = min(x["l"] for x in w[:-1])
            eq = (hi + lo) / 2.0
            htf = b["c"] <= eq if self.direction > 0 else b["c"] >= eq
        t = self._ts(b["ts"])
        minute = t.hour * 60 + t.minute
        macro = in_window(minute, (7 * 60 + 30, 9 * 60 + 30)) or in_window(minute, (14 * 60, 15 * 60 + 30))
        volume = False
        if len(xs) >= 21:
            prev = [x["v"] for x in xs[-21:-1] if x["v"] > 0]
            if prev:
                volume = b["v"] >= float(np.mean(prev)) * 1.05
        flags = {"htf_pda": int(htf), "macro": int(macro), "volume": int(volume)}
        return sum(flags.values()), flags

    def _target_price(self):
        return self.ref_hi if self.direction > 0 else self.ref_lo

    def _runway_ok(self, entry_est: float, stop: float) -> bool:
        target = self._target_price()
        if target is None:
            return False
        risk = abs(entry_est - stop)
        if risk <= 0:
            return False
        reward = (target - entry_est) * self.direction
        return reward >= self.config.target_r * risk

    def _arm(self, b, stop: float, trigger: str) -> None:
        if self.entries_today >= self.config.max_entries_day:
            self._reset_sequence()
            return
        context_count, flags = self._contexts(b)
        if context_count < self.config.context_min:
            return
        self.diag["context_pass"] += 1
        if self.config.require_clear_target and not self._runway_ok(b["c"], stop):
            return
        self.diag["clear_target_pass"] += 1
        self.armed = {
            "direction": self.direction, "stop": float(stop), "trigger": trigger,
            "context": flags, "signal_ts": int(b["ts"]),
            "ref_hi": self.ref_hi, "ref_lo": self.ref_lo,
        }
        self.diag["armed"] += 1
        self._reset_sequence()

    def _track_ifvg(self, b) -> None:
        xs = list(self.bars)
        if len(xs) < 3 or self.sweep_ts is None:
            return
        cur, two = xs[-1], xs[-3]
        if cur["ts"] > self.sweep_ts:
            if cur["l"] > two["h"]:
                self.gaps.append({"kind": "BULL", "lo": two["h"], "hi": cur["l"], "ts": cur["ts"]})
                self.diag["ifvg_created"] += 1
            if cur["h"] < two["l"]:
                self.gaps.append({"kind": "BEAR", "lo": cur["h"], "hi": two["l"], "ts": cur["ts"]})
                self.diag["ifvg_created"] += 1
        self.gaps = self.gaps[-16:]
        for g in self.gaps:
            if b["ts"] <= g["ts"]:
                continue
            if self.direction > 0 and g["kind"] == "BEAR" and b["c"] > g["hi"]:
                self.diag["ifvg_inverted"] += 1
                atr = self._atr14()
                if atr:
                    self._arm(b, self.sweep_extreme - 0.08 * atr, "IFVG")
                return
            if self.direction < 0 and g["kind"] == "BULL" and b["c"] < g["lo"]:
                self.diag["ifvg_inverted"] += 1
                atr = self._atr14()
                if atr:
                    self._arm(b, self.sweep_extreme + 0.08 * atr, "IFVG")
                return

    def on_bar(self, bar: Bar) -> None:
        prior = list(self.bars)
        b = {
            "o": self._f(bar.open), "h": self._f(bar.high),
            "l": self._f(bar.low), "c": self._f(bar.close),
            "v": self._f(bar.volume), "ts": int(bar.ts_event),
        }
        self.bars.append(b)
        self.diag["bars"] += 1
        t = self._ts(b["ts"])
        self._new_day(t.date())
        minute = t.hour * 60 + t.minute
        if in_window(minute, ASIA):
            self.asia_hi, self.asia_lo = self._upd(self.asia_hi, self.asia_lo, b["h"], b["l"])
        if in_window(minute, LONDON):
            self.london_hi, self.london_lo = self._upd(self.london_hi, self.london_lo, b["h"], b["l"])
        if self.entry_ref is not None or self.armed is not None:
            return
        atr = self._atr14()
        if atr is None:
            return

        if self.seq_state == "IDLE":
            ref = self._session_reference(minute)
            if ref is None or len(prior) < self.config.lookback:
                return
            H, L = ref
            min_s, max_s = 0.02 * atr, 1.20 * atr
            high_sweep = b["h"] > H + min_s and b["c"] < H and b["h"] - H <= max_s
            low_sweep = b["l"] < L - min_s and b["c"] > L and L - b["l"] <= max_s
            if not (high_sweep or low_sweep):
                return
            self.diag["sweeps"] += 1
            self.seq_state = "WAIT_MSS" if self.config.model == "MSS_EQ" else "WAIT_IFVG"
            self.age = 0
            self.ref_hi, self.ref_lo = float(H), float(L)
            self.sweep_ts = b["ts"]
            if high_sweep:
                self.direction = -1
                self.sweep_extreme = b["h"]
                self.protected = min(x["l"] for x in prior[-self.config.lookback:])
            else:
                self.direction = 1
                self.sweep_extreme = b["l"]
                self.protected = max(x["h"] for x in prior[-self.config.lookback:])
            return

        self.age += 1
        if self.age > self.config.max_setup_bars:
            self._reset_sequence()
            return

        if self.seq_state == "WAIT_IFVG":
            self.sweep_extreme = max(self.sweep_extreme, b["h"]) if self.direction < 0 else min(self.sweep_extreme, b["l"])
            self._track_ifvg(b)
            return

        if self.seq_state == "WAIT_MSS":
            if self.direction < 0:
                self.sweep_extreme = max(self.sweep_extreme, b["h"])
                mss = b["c"] < self.protected
            else:
                self.sweep_extreme = min(self.sweep_extreme, b["l"])
                mss = b["c"] > self.protected
            disp = abs(b["c"] - b["o"]) >= self.config.displacement_atr * atr
            if not (mss and disp):
                return
            self.diag["mss"] += 1
            self.seq_state = "WAIT_EQ"
            self.age = 0
            self.mss_ts = b["ts"]
            self.disp_extreme = b["l"] if self.direction < 0 else b["h"]
            return

        if self.seq_state == "WAIT_EQ":
            if self.direction < 0:
                self.disp_extreme = min(self.disp_extreme, b["l"])
                rng = self.sweep_extreme - self.disp_extreme
                if rng <= 0: return
                lo = self.disp_extreme + self.config.fib_lo * rng
                hi = self.disp_extreme + self.config.fib_hi * rng
                touch = b["h"] >= lo and b["l"] <= hi
                rejection = touch and b["c"] < lo and b["c"] < b["o"]
                stop = self.sweep_extreme + 0.08 * atr
            else:
                self.disp_extreme = max(self.disp_extreme, b["h"])
                rng = self.disp_extreme - self.sweep_extreme
                if rng <= 0: return
                lo = self.disp_extreme - self.config.fib_hi * rng
                hi = self.disp_extreme - self.config.fib_lo * rng
                touch = b["l"] <= hi and b["h"] >= lo
                rejection = touch and b["c"] > hi and b["c"] > b["o"]
                stop = self.sweep_extreme - 0.08 * atr
            if touch:
                self.diag["eq_touches"] += 1
                self.eq_lo, self.eq_hi = float(lo), float(hi)
                self.eq_touched = True
            if rejection:
                self.diag["eq_rebalance"] += 1
                self._arm(b, stop, "EQ_REBALANCE")

    def on_quote_tick(self, tick: QuoteTick) -> None:
        bid, ask = self._f(tick.bid_price), self._f(tick.ask_price)
        flat = not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
        if self.armed is not None and self.entry_ref is None and flat:
            p = self.armed
            side = int(p["direction"])
            entry = ask if side > 0 else bid
            stop = float(p["stop"])
            if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
                self.armed = None
                return
            target_liq = p["ref_hi"] if side > 0 else p["ref_lo"]
            risk = abs(entry - stop)
            if self.config.require_clear_target and (target_liq - entry) * side < self.config.target_r * risk:
                self.armed = None
                return
            instrument = self.cache.instrument(self.config.instrument_id)
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY if side > 0 else OrderSide.SELL,
                quantity=instrument.make_qty(self.config.trade_size),
            )
            self.submit_order(order)
            self.entry_ref, self.stop_ref = float(entry), float(stop)
            self.tp_ref = float(entry + side * self.config.target_r * risk)
            self.exit_pending = False
            self.entries_today += 1
            self.diag["orders_submitted"] += 1
            self.armed = None
            return
        if self.entry_ref is None or self.exit_pending:
            return
        side = 1 if self.portfolio.is_net_long(self.config.instrument_id) else -1 if self.portfolio.is_net_short(self.config.instrument_id) else 0
        if side == 0: return
        px = bid if side > 0 else ask
        hit = (px <= self.stop_ref or px >= self.tp_ref) if side > 0 else (px >= self.stop_ref or px <= self.tp_ref)
        if hit:
            self.close_all_positions(self.config.instrument_id)
            self.exit_pending = True

    def on_position_closed(self, event) -> None:
        self.entry_ref = self.stop_ref = self.tp_ref = None
        self.exit_pending = False

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--model", choices=["MSS_EQ", "IFVG"], required=True)
    ap.add_argument("--target-r", type=float, required=True)
    ap.add_argument("--context-min", type=int, default=1)
    ap.add_argument("--raw-bidask-only", action="store_true")
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit("RAW_BIDASK_ONLY_REQUIRED")

    catalog_path = Path(args.catalog)
    manifest = json.loads((catalog_path / "catalog_manifest.json").read_text(encoding="utf-8"))
    catalog = ParquetDataCatalog(str(catalog_path))
    instruments = {x.id.symbol.value.replace("/", ""): x for x in catalog.instruments()}
    instrument = instruments.get("XAUUSD")
    if instrument is None: raise SystemExit("XAUUSD instrument missing")
    ticks = catalog.query_quote_ticks(identifiers=[instrument.id.value])
    if not ticks: raise SystemExit("XAUUSD QuoteTicks missing")

    config = BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"), risk_engine=RiskEngineConfig(bypass=True))
    engine = BacktestEngine(config=config)
    engine.add_venue(
        venue=SIM, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        base_currency=USD, starting_balances=[Money(1000, USD)], default_leverage=Decimal("2000"),
    )
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    bar_type = BarType.from_str(f"{instrument.id.value}-1-MINUTE-BID-INTERNAL")
    strat = SequenceRawStrategy(SequenceRawConfig(
        instrument_id=instrument.id, bar_type=bar_type, model=args.model,
        target_r=args.target_r, context_min=args.context_min, trade_size=Decimal("1"),
    ))
    engine.add_strategy(strat)
    engine.run()

    report = engine.trader.generate_positions_report()
    trades = extract_trades(report, "XAUUSD", "M1")
    days = int(manifest["days"])
    kpi = metrics(trades, initial=1000.0, days=days)
    outdir = Path("results/ae-bt") / args.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(outdir / "trades.csv", index=False)
    summary = {
        "verification_level": "NAUTILUS_BT_RAW_BIDASK",
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "data_kind": "RAW_BIDASK QuoteTick", "ohlc_resample_used": False,
        "signal_bars": "Nautilus INTERNAL 1-MINUTE BID bars built from raw QuoteTicks",
        "execution": "Raw QuoteTick market entry/exit; native observed Bid/Ask spread included; no added commission/slippage",
        "instrument": "XAUUSD", "timeframe": "M1", "model": args.model,
        "ordered_core": (
            ["Liquidity Sweep", "Displacement MSS", "EQ/OTE pullback", "EQ Rebalance", "Clear Target", "Entry"]
            if args.model == "MSS_EQ"
            else ["Liquidity Sweep", "post-sweep IFVG inversion", "Clear Target", "Entry"]
        ),
        "target_R": args.target_r, "context_min": args.context_min,
        "period": {"start": manifest["start"], "days": days, "end_exclusive": manifest["end_exclusive"]},
        "raw_ticks": len(ticks), "kpi_fixed_1unit": kpi, "diagnostics": strat.diag,
        "limitations": [
            "First raw parity gate uses fixed 1 XAU unit; PF/WR/N are primary comparison metrics.",
            "Observed raw Bid/Ask spread is native; no added commission or probabilistic slippage.",
            "HTF PDA is reconstructed as 60-minute premium/discount context from raw-built M1 BID bars.",
            "Volume influx uses raw-built bar volume proxy.",
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    engine.dispose()

if __name__ == "__main__":
    main()
