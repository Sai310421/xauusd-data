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

EPS = 1e-12
PRIMARY_N = 67010
PRIMARY_ADDS = 599223
PRIMARY_MONTHLY21 = 574.315
PRIMARY_DD = 2.15813
PRIMARY_PF = 6.98731
PRIMARY_WR = 66.7721
PRIMARY_RF = 16992.9
PRIMARY_BUSINESS_DAYS = 65

START_BALANCE = 10_000.0
LOT = 0.05
XAU_USD_PER_1USD_MOVE_PER_LOT = 100.0
SPREAD_PRICE = 0.30
COMMISSION_USD_PER_LOT_RT = 7.0


@dataclass
class CoreConfig:
    trigger: float = 0.12
    add: float = 0.025
    reversal: float = 0.20
    max_layers: int = 10


class G75Core:
    def __init__(self, cfg: CoreConfig | None = None):
        self.cfg = cfg or CoreConfig()
        self.anchor = None
        self.active = False
        self.side = 0
        self.last_add = None
        self.extreme = None
        self.layers = 0
        self.cycle_id = 0
        self.counts = {"ENTRY": 0, "ADD": 0, "REVERSAL": 0, "EXIT": 0, "CYCLE": 0}

    def on_bar(self, o: float, h: float, l: float, c: float, ts=None):
        ev = []
        cfg = self.cfg
        if self.anchor is None:
            self.anchor = float(c)
            return ev
        if not self.active:
            up = h >= self.anchor + cfg.trigger
            dn = l <= self.anchor - cfg.trigger
            if not (up or dn):
                self.anchor = float(c)
                return ev
            self.side = 1 if c >= self.anchor else -1
            self.cycle_id += 1
            self.active = True
            entry = self.anchor + self.side * cfg.trigger
            self.last_add = entry
            self.extreme = entry
            self.layers = 1
            self.counts["ENTRY"] += 1
            self.counts["CYCLE"] += 1
            ev.append(("ENTRY", entry, ts, self.side, self.cycle_id))
            if self.side == 1:
                while self.layers < cfg.max_layers and self.last_add + cfg.add <= h + EPS:
                    self.last_add += cfg.add
                    self.layers += 1
                    self.counts["ADD"] += 1
                    ev.append(("ADD", self.last_add, ts, self.side, self.cycle_id))
                self.extreme = max(float(self.extreme), float(h))
                reversal_hit = c <= self.extreme - cfg.reversal
            else:
                while self.layers < cfg.max_layers and self.last_add - cfg.add >= l - EPS:
                    self.last_add -= cfg.add
                    self.layers += 1
                    self.counts["ADD"] += 1
                    ev.append(("ADD", self.last_add, ts, self.side, self.cycle_id))
                self.extreme = min(float(self.extreme), float(l))
                reversal_hit = c >= self.extreme + cfg.reversal
            if reversal_hit:
                self.counts["REVERSAL"] += 1
                self.counts["EXIT"] += 1
                ev.append(("REVERSAL", c, ts, self.side, self.cycle_id))
                ev.append(("EXIT", c, ts, self.side, self.cycle_id))
                self.active = False
                self.anchor = float(c)
            return ev
        if self.side == 1:
            while self.layers < cfg.max_layers and self.last_add + cfg.add <= h + EPS:
                self.last_add += cfg.add
                self.layers += 1
                self.counts["ADD"] += 1
                ev.append(("ADD", self.last_add, ts, self.side, self.cycle_id))
            self.extreme = max(float(self.extreme), float(h))
            reversal_hit = c <= self.extreme - cfg.reversal
        else:
            while self.layers < cfg.max_layers and self.last_add - cfg.add >= l - EPS:
                self.last_add -= cfg.add
                self.layers += 1
                self.counts["ADD"] += 1
                ev.append(("ADD", self.last_add, ts, self.side, self.cycle_id))
            self.extreme = min(float(self.extreme), float(l))
            reversal_hit = c >= self.extreme + cfg.reversal
        if reversal_hit:
            self.counts["REVERSAL"] += 1
            self.counts["EXIT"] += 1
            ev.append(("REVERSAL", c, ts, self.side, self.cycle_id))
            ev.append(("EXIT", c, ts, self.side, self.cycle_id))
            self.active = False
            self.anchor = float(c)
        return ev


class G75ParityConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType


class G75ParityStrategy(Strategy):
    def __init__(self, config: G75ParityConfig):
        super().__init__(config)
        self.core = G75Core()
        self.events = []
        self.bar_count = 0
        self.balance = START_BALANCE
        self.peak_equity = START_BALANCE
        self.max_dd_usd = 0.0
        self.max_dd_pct = 0.0
        self.open_prices = []
        self.open_side = 0
        self.cycle_pnls = []

    def on_start(self):
        self.subscribe_bars(self.config.bar_type)

    def _trade_cost_per_layer(self):
        return LOT * (SPREAD_PRICE * XAU_USD_PER_1USD_MOVE_PER_LOT + COMMISSION_USD_PER_LOT_RT)

    def _mark_equity(self, close: float):
        unreal = 0.0
        if self.open_prices:
            unreal = sum(self.open_side * (close - p) * LOT * XAU_USD_PER_1USD_MOVE_PER_LOT for p in self.open_prices)
            unreal -= len(self.open_prices) * self._trade_cost_per_layer()
        eq = self.balance + unreal
        self.peak_equity = max(self.peak_equity, eq)
        dd_usd = self.peak_equity - eq
        dd_pct = 100.0 * dd_usd / self.peak_equity if self.peak_equity > 0 else 0.0
        self.max_dd_usd = max(self.max_dd_usd, dd_usd)
        self.max_dd_pct = max(self.max_dd_pct, dd_pct)

    def on_bar(self, bar: Bar):
        self.bar_count += 1
        close = float(bar.close)
        events = self.core.on_bar(float(bar.open), float(bar.high), float(bar.low), close, bar.ts_event)
        self.events.extend(events)
        for action, price, ts, side, cycle_id in events:
            if action == "ENTRY":
                self.open_side = side
                self.open_prices = [float(price)]
            elif action == "ADD":
                self.open_prices.append(float(price))
            elif action == "EXIT":
                gross = sum(self.open_side * (float(price) - p) * LOT * XAU_USD_PER_1USD_MOVE_PER_LOT for p in self.open_prices)
                cost = len(self.open_prices) * self._trade_cost_per_layer()
                pnl = gross - cost
                self.balance += pnl
                self.cycle_pnls.append(pnl)
                self.open_prices = []
                self.open_side = 0
        self._mark_equity(close)


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in df.columns}
    tcol = lower.get("datetime") or lower.get("timestamp") or lower.get("time")
    if tcol is None:
        raise RuntimeError(f"No datetime column: {list(df.columns)}")
    out = pd.DataFrame({
        "open": pd.to_numeric(df[lower["open"]], errors="coerce"),
        "high": pd.to_numeric(df[lower["high"]], errors="coerce"),
        "low": pd.to_numeric(df[lower["low"]], errors="coerce"),
        "close": pd.to_numeric(df[lower["close"]], errors="coerce"),
    })
    out["volume"] = pd.to_numeric(df[lower["volume"]], errors="coerce").fillna(0.0) if "volume" in lower else 0.0
    idx = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    out.index = idx
    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"]).sort_index()
    out.index = out.index + pd.Timedelta(minutes=1)
    return out


def main():
    df = load_df(Path("csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv"))
    sim = Venue("SIM")
    instrument = TestInstrumentProvider.default_fx_ccy("XAU/USD", sim)
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(df)
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(venue=sim, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                     starting_balances=[Money(1_000_000, USD)], base_currency=USD, default_leverage=Decimal(1))
    engine.add_instrument(instrument)
    engine.add_data(bars)
    strategy = G75ParityStrategy(G75ParityConfig(instrument_id=instrument.id, bar_type=bar_type))
    engine.add_strategy(strategy)
    engine.run()

    counts = strategy.core.counts.copy()
    n, adds = counts["ENTRY"], counts["ADD"]
    pnls = strategy.cycle_pnls
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    net_profit = strategy.balance - START_BALANCE
    return90 = 100.0 * net_profit / START_BALANCE
    monthly21 = ((strategy.balance / START_BALANCE) ** (21.0 / PRIMARY_BUSINESS_DAYS) - 1.0) * 100.0
    wr = 100.0 * len(wins) / len(pnls)
    pf = sum(wins) / abs(sum(losses))
    rf = return90 / strategy.max_dd_pct

    result = {
        "engine": "NautilusTrader BacktestEngine",
        "profitability_mode": "Nautilus event-driven theoretical-fill ledger; fixed 0.05 lot; spread 0.30; commission $7/lot RT",
        "bars": strategy.bar_count,
        "entries_N": n, "adds": adds, "reversals": counts["REVERSAL"], "exits": counts["EXIT"], "cycles": counts["CYCLE"],
        "primary_N": PRIMARY_N, "delta_N": n - PRIMARY_N, "N_match_pct": (1.0 - abs(n - PRIMARY_N) / PRIMARY_N) * 100.0,
        "primary_adds": PRIMARY_ADDS, "delta_adds": adds - PRIMARY_ADDS,
        "start_balance": START_BALANCE, "end_balance": strategy.balance, "net_profit_usd": net_profit,
        "return90_pct": return90, "monthly21_pct": monthly21, "wr_pct": wr, "pf": pf,
        "max_dd_pct": strategy.max_dd_pct, "max_dd_usd": strategy.max_dd_usd, "rf": rf,
        "business_days_for_monthly": PRIMARY_BUSINESS_DAYS,
        "primary_monthly21_pct": PRIMARY_MONTHLY21, "primary_wr_pct": PRIMARY_WR,
        "primary_pf": PRIMARY_PF, "primary_dd_pct": PRIMARY_DD, "primary_rf": PRIMARY_RF,
        "nautilus_version": __import__("nautilus_trader").__version__, "instrument_envelope": str(instrument.id),
        "data_start": str(df.index.min()), "data_end": str(df.index.max()),
    }
    out = Path("research/g75_nautilus/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "g75_nautilus_profitability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(out / "g75_nautilus_profitability.csv", index=False)
    print("G75_NAUTILUS_PROFITABILITY=" + json.dumps(result, ensure_ascii=False))
    engine.dispose()


if __name__ == "__main__":
    main()
