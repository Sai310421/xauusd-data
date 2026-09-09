from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy

from g75_event_parity import (
    EPS, START_BALANCE, LOT, XAU_USD_PER_1USD_MOVE_PER_LOT,
    SPREAD_PRICE, COMMISSION_USD_PER_LOT_RT, PRIMARY_BUSINESS_DAYS,
    load_df,
)

BASE_TRIGGER = 0.12
BASE_ADD = 0.025
BASE_REVERSAL = 0.20
MAX_LAYERS = 10
WARMUP_BARS = 1440
ATR_ALPHA = 2.0 / (60.0 + 1.0)
SCALE_MIN = 0.50
SCALE_MAX = 2.00


@dataclass
class DistanceState:
    trigger: float = BASE_TRIGGER
    add: float = BASE_ADD
    reversal: float = BASE_REVERSAL
    scale: float = 1.0


class AdaptiveDistanceCore:
    """G75 core with causal volatility scaling.

    The G75 DNA ratios are preserved exactly:
      add/trigger = 0.025/0.12
      reversal/trigger = 0.20/0.12
      reversal/add = 8

    Scale is estimated causally from 1-minute true range. During FLAT the scale
    follows ATR(EMA60) relative to the first 1440-bar reference median TR.
    At ENTRY the scale is frozen for the whole cycle so Add/Reversal geometry
    cannot jump mid-cycle. No future bars are used.
    """
    def __init__(self, adaptive: bool):
        self.adaptive = adaptive
        self.anchor = None
        self.active = False
        self.side = 0
        self.last_add = None
        self.extreme = None
        self.layers = 0
        self.cycle_id = 0
        self.counts = {"ENTRY": 0, "ADD": 0, "REVERSAL": 0, "EXIT": 0, "CYCLE": 0}
        self.prev_close = None
        self.atr_fast = None
        self.warmup_tr = []
        self.ref_tr = None
        self.live_scale = 1.0
        self.cycle_scale = 1.0
        self.scale_samples = []

    def _update_scale(self, h: float, l: float, c: float):
        if self.prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.prev_close = c
        if self.atr_fast is None:
            self.atr_fast = tr
        else:
            self.atr_fast = ATR_ALPHA * tr + (1.0 - ATR_ALPHA) * self.atr_fast
        if len(self.warmup_tr) < WARMUP_BARS:
            self.warmup_tr.append(tr)
            if len(self.warmup_tr) == WARMUP_BARS:
                s = pd.Series(self.warmup_tr)
                self.ref_tr = float(s.median())
            self.live_scale = 1.0
        elif self.ref_tr and self.ref_tr > EPS:
            self.live_scale = min(SCALE_MAX, max(SCALE_MIN, self.atr_fast / self.ref_tr))
        else:
            self.live_scale = 1.0
        self.scale_samples.append(self.live_scale)

    def _distances(self):
        s = self.cycle_scale if self.active else self.live_scale
        if not self.adaptive:
            s = 1.0
        return DistanceState(BASE_TRIGGER*s, BASE_ADD*s, BASE_REVERSAL*s, s)

    def on_bar(self, o: float, h: float, l: float, c: float, ts=None):
        self._update_scale(h, l, c)
        ev = []
        d = self._distances()
        if self.anchor is None:
            self.anchor = float(c)
            return ev
        if not self.active:
            up = h >= self.anchor + d.trigger
            dn = l <= self.anchor - d.trigger
            if not (up or dn):
                self.anchor = float(c)
                return ev
            self.side = 1 if c >= self.anchor else -1
            self.cycle_scale = self.live_scale if self.adaptive else 1.0
            d = DistanceState(BASE_TRIGGER*self.cycle_scale, BASE_ADD*self.cycle_scale,
                              BASE_REVERSAL*self.cycle_scale, self.cycle_scale)
            self.cycle_id += 1
            self.active = True
            entry = self.anchor + self.side * d.trigger
            self.last_add = entry
            self.extreme = entry
            self.layers = 1
            self.counts["ENTRY"] += 1
            self.counts["CYCLE"] += 1
            ev.append(("ENTRY", entry, ts, self.side, self.cycle_id))
        else:
            d = self._distances()

        if self.side == 1:
            while self.layers < MAX_LAYERS and self.last_add + d.add <= h + EPS:
                self.last_add += d.add
                self.layers += 1
                self.counts["ADD"] += 1
                ev.append(("ADD", self.last_add, ts, self.side, self.cycle_id))
            self.extreme = max(float(self.extreme), float(h))
            reversal_hit = c <= self.extreme - d.reversal
        else:
            while self.layers < MAX_LAYERS and self.last_add - d.add >= l - EPS:
                self.last_add -= d.add
                self.layers += 1
                self.counts["ADD"] += 1
                ev.append(("ADD", self.last_add, ts, self.side, self.cycle_id))
            self.extreme = min(float(self.extreme), float(l))
            reversal_hit = c >= self.extreme + d.reversal

        if reversal_hit:
            self.counts["REVERSAL"] += 1
            self.counts["EXIT"] += 1
            ev.append(("REVERSAL", c, ts, self.side, self.cycle_id))
            ev.append(("EXIT", c, ts, self.side, self.cycle_id))
            self.active = False
            self.anchor = float(c)
            self.cycle_scale = 1.0
        return ev


class ABConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    adaptive: bool


class ABStrategy(Strategy):
    def __init__(self, config: ABConfig):
        super().__init__(config)
        self.core = AdaptiveDistanceCore(config.adaptive)
        self.balance = START_BALANCE
        self.peak_equity = START_BALANCE
        self.max_dd_pct = 0.0
        self.open_prices = []
        self.open_side = 0
        self.cycle_pnls = []
        self.bar_count = 0

    def on_start(self):
        self.subscribe_bars(self.config.bar_type)

    @staticmethod
    def _cost():
        return LOT * (SPREAD_PRICE * XAU_USD_PER_1USD_MOVE_PER_LOT + COMMISSION_USD_PER_LOT_RT)

    def _mark(self, close: float):
        unreal = sum(self.open_side*(close-p)*LOT*XAU_USD_PER_1USD_MOVE_PER_LOT for p in self.open_prices)
        unreal -= len(self.open_prices)*self._cost()
        eq = self.balance + unreal
        self.peak_equity = max(self.peak_equity, eq)
        dd = 100.0*(self.peak_equity-eq)/self.peak_equity if self.peak_equity else 0.0
        self.max_dd_pct = max(self.max_dd_pct, dd)

    def on_bar(self, bar: Bar):
        self.bar_count += 1
        close = float(bar.close)
        events = self.core.on_bar(float(bar.open), float(bar.high), float(bar.low), close, bar.ts_event)
        for action, price, ts, side, cid in events:
            if action == "ENTRY":
                self.open_side = side
                self.open_prices = [float(price)]
            elif action == "ADD":
                self.open_prices.append(float(price))
            elif action == "EXIT":
                gross = sum(self.open_side*(float(price)-p)*LOT*XAU_USD_PER_1USD_MOVE_PER_LOT for p in self.open_prices)
                pnl = gross - len(self.open_prices)*self._cost()
                self.balance += pnl
                self.cycle_pnls.append(pnl)
                self.open_prices = []
                self.open_side = 0
        self._mark(close)


def summarize(name: str, s: ABStrategy):
    pnls = s.cycle_pnls
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    net = s.balance - START_BALANCE
    ret90 = 100.0*net/START_BALANCE
    monthly21 = ((s.balance/START_BALANCE)**(21.0/PRIMARY_BUSINESS_DAYS)-1.0)*100.0
    pf = sum(wins)/abs(sum(losses)) if losses else float("inf")
    wr = 100.0*len(wins)/len(pnls) if pnls else 0.0
    rf = ret90/s.max_dd_pct if s.max_dd_pct else float("inf")
    scales = pd.Series(s.core.scale_samples[WARMUP_BARS:]) if len(s.core.scale_samples) > WARMUP_BARS else pd.Series([1.0])
    return {
        "variant": name,
        "adaptive": s.config.adaptive,
        "bars": s.bar_count,
        "entries_N": s.core.counts["ENTRY"],
        "adds": s.core.counts["ADD"],
        "cycles": s.core.counts["CYCLE"],
        "wr_pct": wr,
        "pf": pf,
        "max_dd_pct": s.max_dd_pct,
        "return90_pct": ret90,
        "monthly21_pct": monthly21,
        "rf": rf,
        "end_balance": s.balance,
        "scale_mean": float(scales.mean()),
        "scale_median": float(scales.median()),
        "scale_p05": float(scales.quantile(.05)),
        "scale_p95": float(scales.quantile(.95)),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
    }


def run_variant(df: pd.DataFrame, adaptive: bool):
    sim = Venue("SIM")
    instrument = TestInstrumentProvider.default_fx_ccy("XAU/USD", sim)
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(df)
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(venue=sim, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                     starting_balances=[Money(1_000_000, USD)], base_currency=USD, default_leverage=Decimal(1))
    engine.add_instrument(instrument)
    engine.add_data(bars)
    strategy = ABStrategy(ABConfig(instrument_id=instrument.id, bar_type=bar_type, adaptive=adaptive))
    engine.add_strategy(strategy)
    engine.run()
    result = summarize("B_ADAPTIVE_ATR60_FROZEN_CYCLE" if adaptive else "A_FROZEN_DISTANCE", strategy)
    engine.dispose()
    return result


def main():
    df = load_df(Path("csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv"))
    a = run_variant(df, False)
    b = run_variant(df, True)
    result = {
        "engine": "NautilusTrader BacktestEngine",
        "data_start": str(df.index.min()),
        "data_end": str(df.index.max()),
        "adaptive_formula": "s=clip(EMA60(TrueRange)/median(first 1440 causal TR),0.50,2.00); T=0.12s; A=0.025s; R=0.20s; scale frozen per cycle",
        "ratio_invariants": {"A/T": BASE_ADD/BASE_TRIGGER, "R/T": BASE_REVERSAL/BASE_TRIGGER, "R/A": BASE_REVERSAL/BASE_ADD},
        "A": a,
        "B": b,
        "delta_B_minus_A": {k: b[k]-a[k] for k in ["entries_N","adds","wr_pct","pf","max_dd_pct","return90_pct","monthly21_pct","rf"]},
        "gates": {
            "N_retention_pct": 100.0*b["entries_N"]/a["entries_N"] if a["entries_N"] else 0.0,
            "N_retention_ge_95": b["entries_N"] >= .95*a["entries_N"],
            "DD_le_3": b["max_dd_pct"] <= 3.0,
            "PF_ge_6": b["pf"] >= 6.0,
            "monthly_not_below_A": b["monthly21_pct"] >= a["monthly21_pct"],
        },
    }
    out = Path("research/g75_nautilus/results")
    out.mkdir(parents=True, exist_ok=True)
    (out/"g75_adaptive_distance_ab.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([a,b]).to_csv(out/"g75_adaptive_distance_ab.csv", index=False)
    print("G75_ADAPTIVE_AB=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
