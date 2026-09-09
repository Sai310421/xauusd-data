from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import nautilus_trader
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue("SIM")


@dataclass(frozen=True)
class RiskBudget:
    budget: float
    saturation_target: float
    utilization: float
    dd_headroom: float
    tail_penalty: float
    recovery_prob_proxy: float


class Cfg(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    gate: str = "G1"
    initial_balance: float = 1000.0
    trigger: float = 0.12
    add: float = 0.025
    reversal: float = 0.20
    max_layers: int = 10
    dd_limit: float = 4.5
    saturation_target: float = 0.80
    min_layers: int = 1
    risk_floor: float = 0.05
    tail_window: int = 240


class RiskSaturationController:
    """Continuous risk-budget controller.

    The controller does not decide direction. It only transforms current
    state into an admissible risk budget. The alpha module can be replaced
    without changing this class.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def budget(self, dd_pct: float, tail_z: float, trend_conf: float) -> RiskBudget:
        headroom = max(0.0, 1.0 - dd_pct / max(self.cfg.dd_limit, 1e-9))
        # Smoothly reduce risk before the hard boundary. Squared headroom gives
        # stronger protection near the limit without binary entry shutdown.
        dd_factor = headroom * headroom
        tail_penalty = 1.0 / (1.0 + max(0.0, tail_z) ** 2)
        # First-passage proxy: more headroom + stronger directional confidence
        # implies a higher probability of natural recovery before the DD wall.
        p_recovery = min(1.0, max(0.0, 0.60 * headroom + 0.40 * trend_conf))
        recovery_factor = 0.25 + 0.75 * p_recovery
        raw_budget = self.cfg.saturation_target * dd_factor * tail_penalty * recovery_factor
        b = 0.0 if headroom <= 0 else max(self.cfg.risk_floor, min(1.0, raw_budget))
        return RiskBudget(
            budget=b,
            saturation_target=self.cfg.saturation_target,
            utilization=b / max(self.cfg.saturation_target, 1e-9),
            dd_headroom=headroom,
            tail_penalty=tail_penalty,
            recovery_prob_proxy=p_recovery,
        )


class S(Strategy):
    def __init__(self, cfg: Cfg):
        super().__init__(cfg)
        self.ctrl = RiskSaturationController(cfg)
        self.buckets = {60: None, 300: None, 900: None}
        self.cur = {60: None, 300: None, 900: None}
        self.cl = {60: deque(maxlen=64), 300: deque(maxlen=64), 900: deque(maxlen=64)}
        self.ret = deque(maxlen=cfg.tail_window)
        self.anchor = None
        self.active = False
        self.side = 0
        self.entries: list[tuple[float, float]] = []  # (price, risk weight)
        self.last_add = None
        self.extreme = None
        self.started_m1 = False
        self.real = 0.0
        self.peak = cfg.initial_balance
        self.maxdd = 0.0
        self.cycles = self.wins = self.losses = self.adds = 0
        self.gw = self.gl = 0.0
        self.maxlayers = 0
        self.last_mid = None
        self.last_bid = self.last_ask = None
        self.budget_sum = 0.0
        self.budget_n = 0
        self.saturation_sum = 0.0
        self.tail_max = 0.0
        self.hard_events = 0
        self.entry_candidates = 0
        self.direction_rejects = 0

    @staticmethod
    def f(x):
        return float(x.as_double()) if hasattr(x, "as_double") else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def ema9(self, tf):
        x = list(self.cl[tf])
        if len(x) < 10:
            return None, None
        a = 2 / 10
        e = x[0]
        prev = e
        for v in x:
            prev = e
            e = a * v + (1 - a) * e
        return e, e - prev

    def bias_and_conf(self):
        e5, s5 = self.ema9(300)
        if e5 is None:
            return 0, 0.0
        c5 = self.cl[300][-1]
        dist5 = abs(c5 - e5)
        b5 = 1 if c5 > e5 and s5 > 0 else (-1 if c5 < e5 and s5 < 0 else 0)
        conf5 = min(1.0, (dist5 + abs(s5)) / max(self.config.trigger, 1e-9))
        if self.config.gate == "G1":
            return b5, conf5
        e15, s15 = self.ema9(900)
        if e15 is None:
            return 0, 0.0
        c15 = self.cl[900][-1]
        b15 = 1 if c15 > e15 and s15 > 0 else (-1 if c15 < e15 and s15 < 0 else 0)
        conf15 = min(1.0, (abs(c15 - e15) + abs(s15)) / max(self.config.trigger, 1e-9))
        if b5 == 0 or b5 != b15:
            return 0, 0.0
        conf = min(conf5, conf15)
        if self.config.gate == "G2":
            return b5, conf
        if len(self.cl[60]) < 4:
            return 0, 0.0
        mom = self.cl[60][-1] - self.cl[60][-4]
        if mom * b5 <= 0:
            return 0, 0.0
        return b5, min(1.0, conf + abs(mom) / max(self.config.trigger * 3, 1e-9))

    def tail_z(self):
        if len(self.ret) < 30:
            return 0.0
        x = list(self.ret)
        mu = sum(x) / len(x)
        var = sum((v - mu) ** 2 for v in x) / max(1, len(x) - 1)
        sd = math.sqrt(max(var, 1e-12))
        z = abs(x[-1] - mu) / sd
        self.tail_max = max(self.tail_max, z)
        return z

    def mark(self, bid, ask):
        if not self.active:
            return 0.0
        px = bid if self.side > 0 else ask
        return sum((px - e) * self.side * w for e, w in self.entries)

    def dd(self, bid, ask):
        eq = self.config.initial_balance + self.real + self.mark(bid, ask)
        self.peak = max(self.peak, eq)
        d = max(0.0, (self.peak - eq) / max(self.peak, 1e-9) * 100)
        self.maxdd = max(self.maxdd, d)
        return d

    def close(self, bid, ask, reason):
        if not self.active:
            return
        px = bid if self.side > 0 else ask
        p = sum((px - e) * self.side * w for e, w in self.entries)
        self.real += p
        self.cycles += 1
        if p > 0:
            self.wins += 1
            self.gw += p
        elif p < 0:
            self.losses += 1
            self.gl += abs(p)
        self.active = False
        self.side = 0
        self.entries = []
        self.last_add = None
        self.extreme = None

    def finish(self, tf, bid, ask):
        c = self.cur[tf]
        if c is None:
            return
        self.cl[tf].append(c)
        self.cur[tf] = None
        if tf == 60:
            if self.active:
                px = bid if self.side > 0 else ask
                rev = px <= self.extreme - self.config.reversal if self.side > 0 else px >= self.extreme + self.config.reversal
                if rev:
                    self.close(bid, ask, "REV")
            self.anchor = c
            self.started_m1 = False

    def on_quote_tick(self, t):
        bid = self.f(t.bid_price)
        ask = self.f(t.ask_price)
        mid = (bid + ask) / 2
        if self.last_mid is not None:
            self.ret.append(mid - self.last_mid)
        self.last_mid = mid
        self.last_bid, self.last_ask = bid, ask
        sec = int(t.ts_event) // 1_000_000_000
        for tf in (60, 300, 900):
            b = sec // tf
            if self.buckets[tf] is None:
                self.buckets[tf] = b
            elif b != self.buckets[tf]:
                self.finish(tf, bid, ask)
                self.buckets[tf] = b
            self.cur[tf] = mid
        if self.anchor is None:
            self.anchor = mid
            return

        d = self.dd(bid, ask)
        bias, conf = self.bias_and_conf()
        rb = self.ctrl.budget(d, self.tail_z(), conf)
        self.budget_sum += rb.budget
        self.budget_n += 1
        if d >= self.config.dd_limit:
            self.hard_events += 1
            if self.active:
                self.close(bid, ask, "DD_LIMIT")
            return

        # Risk budget maps continuously into admissible layers and position weight.
        cap = max(self.config.min_layers, min(self.config.max_layers, int(math.ceil(self.config.max_layers * rb.budget))))
        unit_w = max(self.config.risk_floor, rb.budget)
        if not self.active and not self.started_m1:
            # More available budget permits slightly denser entry sampling, while
            # low budget raises the trigger threshold rather than hard-disabling alpha.
            adaptive_trigger = self.config.trigger * (1.20 - 0.40 * rb.budget)
            up = mid >= self.anchor + adaptive_trigger
            dn = mid <= self.anchor - adaptive_trigger
            if up or dn:
                self.entry_candidates += 1
                sig = 1 if up else -1
                if bias != sig:
                    self.direction_rejects += 1
                    self.started_m1 = True
                    return
                self.side = sig
                entry = ask if sig > 0 else bid
                self.active = True
                self.entries = [(entry, unit_w)]
                self.last_add = entry
                self.extreme = bid if sig > 0 else ask
                self.started_m1 = True
                self.maxlayers = max(self.maxlayers, 1)

        if not self.active:
            return
        px = bid if self.side > 0 else ask
        self.extreme = max(self.extreme, px) if self.side > 0 else min(self.extreme, px)
        while len(self.entries) < cap:
            # Lower budget widens add spacing; high budget approaches Frozen G75 add.
            add_step = self.config.add * (1.60 - 0.60 * rb.budget)
            target = self.last_add + self.side * add_step
            cross = px >= target if self.side > 0 else px <= target
            if not cross:
                break
            fill = ask if self.side > 0 else bid
            self.entries.append((fill, unit_w))
            self.last_add = target
            self.adds += 1
            self.maxlayers = max(self.maxlayers, len(self.entries))
        used = sum(w for _, w in self.entries) / max(1, self.config.max_layers)
        self.saturation_sum += min(1.0, used / max(rb.budget, 1e-9))
        self.dd(bid, ask)

    def on_stop(self):
        if self.active and self.last_bid is not None:
            self.close(self.last_bid, self.last_ask, "EOD")

    def summary(self):
        pf = self.gw / self.gl if self.gl else (math.inf if self.gw else 0.0)
        avg_budget = self.budget_sum / max(1, self.budget_n)
        return {
            "gate": self.config.gate,
            "dd_limit_pct": self.config.dd_limit,
            "saturation_target": self.config.saturation_target,
            "cycles": self.cycles,
            "WR_pct": 100 * self.wins / max(1, self.cycles),
            "PF": pf,
            "realized_usd_0p01lot_equiv": self.real,
            "return_pct_on_1000": self.real / 10,
            "max_DD_pct": self.maxdd,
            "growth_per_DD": (self.real / 10) / max(self.maxdd, 1e-9),
            "adds": self.adds,
            "max_layers": self.maxlayers,
            "avg_risk_budget": avg_budget,
            "avg_budget_utilization": self.saturation_sum / max(1, self.budget_n),
            "tail_z_max": self.tail_max,
            "hard_events": self.hard_events,
            "entry_candidates": self.entry_candidates,
            "direction_rejects": self.direction_rejects,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--gate", choices=["G1", "G2", "G3"], default="G1")
    p.add_argument("--dd-limit", type=float, required=True)
    p.add_argument("--saturation", type=float, required=True)
    p.add_argument("--raw-bidask-only", action="store_true")
    a = p.parse_args()
    if not a.raw_bidask_only:
        raise SystemExit("Raw BidAsk mandatory")
    if not (0 < a.saturation <= 1):
        raise SystemExit("saturation must be in (0,1]")

    cp = Path(a.catalog)
    man = json.loads((cp / "catalog_manifest.json").read_text())
    cat = ParquetDataCatalog(str(cp))
    inst = next(x for x in cat.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD")
    ticks = cat.query(data_cls=QuoteTick, identifiers=[inst.id.value])

    eng = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"), risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN, base_currency=USD, starting_balances=[Money(1000, USD)], default_leverage=Decimal("2000"))
    eng.add_instrument(inst)
    eng.add_data(ticks)
    s = S(Cfg(instrument_id=inst.id, gate=a.gate, dd_limit=a.dd_limit, saturation_target=a.saturation))
    eng.add_strategy(s)
    eng.run()
    r = {
        **s.summary(),
        "raw_ticks": len(ticks),
        "chronology": "AE_RISK_SATURATION_V1_RAW_BIDASK",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "period_start": man.get("start"),
        "period_days": man.get("days"),
        "period_end_exclusive": man.get("end_exclusive"),
    }
    out = Path("results/ae-risk-saturation") / a.experiment_id / "cells"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.gate}_DD{a.dd_limit:g}_S{int(round(a.saturation*100)):02d}.json"
    (out / name).write_text(json.dumps(r, indent=2))
    print(json.dumps(r))
    eng.dispose()


if __name__ == "__main__":
    main()
