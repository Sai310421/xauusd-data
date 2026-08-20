from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy

EPS = 1e-12
PRIMARY_N = 67010
PRIMARY_ADDS = 599223


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

            # Recovered G75 General Measurement rule: choose direction from bar close
            # once the anchor threshold has been crossed; execute theoretical trigger level.
            self.side = 1 if c >= self.anchor else -1
            self.cycle_id += 1
            self.active = True
            entry = self.anchor + self.side * cfg.trigger
            self.last_add = entry
            self.extreme = entry
            self.layers = 1
            self.counts["ENTRY"] += 1
            self.counts["CYCLE"] += 1
            ev.append(("ENTRY", entry, ts))

            if self.side == 1:
                while self.layers < cfg.max_layers and self.last_add + cfg.add <= h + EPS:
                    self.last_add += cfg.add
                    self.layers += 1
                    self.counts["ADD"] += 1
                    ev.append(("ADD", self.last_add, ts))
                self.extreme = max(float(self.extreme), float(h))
                reversal_hit = c <= self.extreme - cfg.reversal
            else:
                while self.layers < cfg.max_layers and self.last_add - cfg.add >= l - EPS:
                    self.last_add -= cfg.add
                    self.layers += 1
                    self.counts["ADD"] += 1
                    ev.append(("ADD", self.last_add, ts))
                self.extreme = min(float(self.extreme), float(l))
                reversal_hit = c >= self.extreme + cfg.reversal

            if reversal_hit:
                self.counts["REVERSAL"] += 1
                self.counts["EXIT"] += 1
                ev.append(("REVERSAL", c, ts))
                ev.append(("EXIT", c, ts))
                self.active = False
                self.anchor = float(c)
            return ev

        if self.side == 1:
            while self.layers < cfg.max_layers and self.last_add + cfg.add <= h + EPS:
                self.last_add += cfg.add
                self.layers += 1
                self.counts["ADD"] += 1
                ev.append(("ADD", self.last_add, ts))
            self.extreme = max(float(self.extreme), float(h))
            reversal_hit = c <= self.extreme - cfg.reversal
        else:
            while self.layers < cfg.max_layers and self.last_add - cfg.add >= l - EPS:
                self.last_add -= cfg.add
                self.layers += 1
                self.counts["ADD"] += 1
                ev.append(("ADD", self.last_add, ts))
            self.extreme = min(float(self.extreme), float(l))
            reversal_hit = c >= self.extreme + cfg.reversal

        if reversal_hit:
            self.counts["REVERSAL"] += 1
            self.counts["EXIT"] += 1
            ev.append(("REVERSAL", c, ts))
            ev.append(("EXIT", c, ts))
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

    def on_start(self):
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar):
        self.bar_count += 1
        o = float(bar.open)
        h = float(bar.high)
        l = float(bar.low)
        c = float(bar.close)
        self.events.extend(self.core.on_bar(o, h, l, c, bar.ts_event))


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
    if "volume" in lower:
        out["volume"] = pd.to_numeric(df[lower["volume"]], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0
    idx = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    out.index = idx
    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"]).sort_index()
    # Nautilus bar ts_init represents bar close. Source timestamps are bar-open M1 timestamps.
    out.index = out.index + pd.Timedelta(minutes=1)
    return out


def main():
    data_path = Path("csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv")
    df = load_df(data_path)

    sim = Venue("SIM")
    try:
        instrument = TestInstrumentProvider.default_fx_ccy("XAU/USD", sim)
    except Exception:
        # Event parity does not submit orders; instrument is only a Nautilus data envelope.
        instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", sim)

    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(df)

    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    engine.add_instrument(instrument)
    engine.add_data(bars)
    strategy = G75ParityStrategy(G75ParityConfig(instrument_id=instrument.id, bar_type=bar_type))
    engine.add_strategy(strategy)
    engine.run()

    counts = strategy.core.counts.copy()
    n = counts["ENTRY"]
    adds = counts["ADD"]
    result = {
        "engine": "NautilusTrader BacktestEngine",
        "bars": strategy.bar_count,
        "entries_N": n,
        "adds": adds,
        "reversals": counts["REVERSAL"],
        "exits": counts["EXIT"],
        "cycles": counts["CYCLE"],
        "primary_N": PRIMARY_N,
        "delta_N": n - PRIMARY_N,
        "N_match_pct": (1.0 - abs(n - PRIMARY_N) / PRIMARY_N) * 100.0,
        "primary_adds": PRIMARY_ADDS,
        "delta_adds": adds - PRIMARY_ADDS,
        "gate_95": abs(n - PRIMARY_N) <= PRIMARY_N * 0.05,
        "gate_99": abs(n - PRIMARY_N) <= PRIMARY_N * 0.01,
        "nautilus_version": __import__("nautilus_trader").__version__,
        "instrument_envelope": str(instrument.id),
        "data_start": str(df.index.min()),
        "data_end": str(df.index.max()),
    }
    out = Path("research/g75_nautilus/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "g75_nautilus_event_parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(out / "g75_nautilus_event_parity.csv", index=False)
    print("G75_NAUTILUS_RESULT=" + json.dumps(result, ensure_ascii=False))
    engine.dispose()


if __name__ == "__main__":
    main()
