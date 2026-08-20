"""
G75 Adaptive Distance DD3 v1.0

Purpose
-------
Preserve the G75 price-state-transition EDGE while adapting the price distances
and reallocating the Adaptive A/B drawdown reduction to a 3.0% DD budget.

Measured Adaptive baseline used for initial calibration:
    Max DD = 1.7700%

Initial risk multiplier:
    k0 = 3.0 / 1.7700 = 1.6949152542

This is a first-order calibration, not a guarantee of exactly 3.000% DD.
After a fresh Nautilus run, recalibrate:
    k_next = k_current * 3.0 / DD_measured

Core geometry
-------------
scale_t = clip(EMA60(TrueRange_t) / Median1440(TrueRange up to t), 0.5, 2.0)
Trigger_t  = 0.12  * scale_cycle
Add_t      = 0.025 * scale_cycle
Reversal_t = 0.20  * scale_cycle

The scale is frozen at cycle entry until Basket Exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_TRIGGER = 0.12
BASE_ADD = 0.025
BASE_REVERSAL = 0.20
MAX_LAYERS = 10

TARGET_DD_PCT = 3.0
ADAPTIVE_BASELINE_DD_PCT = 1.7700

BASE_LOT = 0.05
RISK_MULTIPLIER = TARGET_DD_PCT / ADAPTIVE_BASELINE_DD_PCT
SCALED_LOT = BASE_LOT * RISK_MULTIPLIER

SCALE_MIN = 0.50
SCALE_MAX = 2.00
TR_EMA_SPAN = 60
TR_MEDIAN_WINDOW = 1440
EPS = 1e-12


@dataclass
class G75AdaptiveDD3Config:
    trigger: float = BASE_TRIGGER
    add: float = BASE_ADD
    reversal: float = BASE_REVERSAL
    max_layers: int = MAX_LAYERS
    base_lot: float = BASE_LOT
    risk_multiplier: float = RISK_MULTIPLIER
    target_dd_pct: float = TARGET_DD_PCT


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def build_adaptive_scale(df: pd.DataFrame) -> pd.Series:
    """No-lookahead volatility scale."""
    tr = true_range(df)
    fast = tr.ewm(span=TR_EMA_SPAN, adjust=False, min_periods=1).mean()
    ref = tr.rolling(
        TR_MEDIAN_WINDOW,
        min_periods=max(60, TR_MEDIAN_WINDOW // 10),
    ).median()
    ref = ref.fillna(tr.expanding(min_periods=1).median()).replace(0.0, np.nan)
    scale = (fast / ref).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return scale.clip(SCALE_MIN, SCALE_MAX)


class G75AdaptiveDD3Core:
    """
    State machine:
        FLAT -> TRIGGER -> ENTRY -> ADD* -> RUNNING_EXTREME
             -> REVERSAL -> BASKET_EXIT -> IMMEDIATE_REARM
    """

    def __init__(self, cfg: G75AdaptiveDD3Config | None = None):
        self.cfg = cfg or G75AdaptiveDD3Config()
        self.anchor = None
        self.active = False
        self.side = 0
        self.last_add = None
        self.extreme = None
        self.layers = 0
        self.cycle_scale = 1.0
        self.entries = []
        self.events = []

    @property
    def lot(self) -> float:
        return self.cfg.base_lot * self.cfg.risk_multiplier

    def distances(self):
        s = self.cycle_scale
        return (
            self.cfg.trigger * s,
            self.cfg.add * s,
            self.cfg.reversal * s,
        )

    def on_bar(self, ts, o, h, l, c, adaptive_scale):
        o, h, l, c = map(float, (o, h, l, c))
        adaptive_scale = float(adaptive_scale)

        if self.anchor is None:
            self.anchor = c
            return []

        out = []

        if not self.active:
            trigger_now = self.cfg.trigger * adaptive_scale
            up = h >= self.anchor + trigger_now
            dn = l <= self.anchor - trigger_now

            if not (up or dn):
                self.anchor = c
                return []

            # Freeze the market geometry for the entire basket/cycle.
            self.cycle_scale = adaptive_scale
            trigger, add, reversal = self.distances()

            self.side = 1 if c >= self.anchor else -1
            entry = self.anchor + self.side * trigger

            self.active = True
            self.layers = 1
            self.last_add = entry
            self.extreme = entry
            self.entries = [entry]

            out.append(
                ("ENTRY", ts, entry, self.side, self.layers, self.cycle_scale, self.lot)
            )

        trigger, add, reversal = self.distances()

        if self.side == 1:
            while (
                self.layers < self.cfg.max_layers
                and self.last_add + add <= h + EPS
            ):
                self.last_add += add
                self.layers += 1
                self.entries.append(self.last_add)
                out.append(
                    ("ADD", ts, self.last_add, self.side, self.layers, self.cycle_scale, self.lot)
                )

            # Preserve G75 ordering: update the running extreme before reversal check.
            self.extreme = max(float(self.extreme), h)
            reversal_hit = c <= self.extreme - reversal
        else:
            while (
                self.layers < self.cfg.max_layers
                and self.last_add - add >= l - EPS
            ):
                self.last_add -= add
                self.layers += 1
                self.entries.append(self.last_add)
                out.append(
                    ("ADD", ts, self.last_add, self.side, self.layers, self.cycle_scale, self.lot)
                )

            self.extreme = min(float(self.extreme), l)
            reversal_hit = c >= self.extreme + reversal

        if reversal_hit:
            out.append(
                ("REVERSAL", ts, c, self.side, self.layers, self.cycle_scale, self.lot)
            )
            out.append(
                ("BASKET_EXIT", ts, c, self.side, self.layers, self.cycle_scale, self.lot)
            )

            # Immediate re-arm.
            self.active = False
            self.anchor = c
            self.entries = []
            self.layers = 0
            self.last_add = None
            self.extreme = None

        self.events.extend(out)
        return out


def next_risk_multiplier(
    current_multiplier: float,
    measured_dd_pct: float,
    target_dd_pct: float = TARGET_DD_PCT,
) -> float:
    """First-order DD re-calibration for the next Nautilus run."""
    if measured_dd_pct <= 0:
        raise ValueError("measured_dd_pct must be > 0")
    return current_multiplier * target_dd_pct / measured_dd_pct


def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in df.columns}
    tcol = next((lower[x] for x in ("datetime", "timestamp", "time") if x in lower), None)
    if tcol is None:
        raise ValueError("CSV requires datetime/timestamp/time column")

    out = pd.DataFrame(index=pd.to_datetime(df[tcol], utc=True, errors="coerce"))
    for k in ("open", "high", "low", "close"):
        if k not in lower:
            raise ValueError(f"CSV missing column: {k}")
        out[k] = pd.to_numeric(df[lower[k]], errors="coerce").to_numpy()

    out = out[~out.index.isna()].dropna().sort_index()
    out["adaptive_scale"] = build_adaptive_scale(out)
    return out


def dry_event_run(csv_path: str | Path):
    df = load_ohlc_csv(csv_path)
    core = G75AdaptiveDD3Core()

    for ts, r in df.iterrows():
        core.on_bar(
            ts,
            r["open"],
            r["high"],
            r["low"],
            r["close"],
            r["adaptive_scale"],
        )

    return {
        "version": "G75_Adaptive_DD3_v1.0",
        "target_dd_pct": TARGET_DD_PCT,
        "adaptive_baseline_dd_pct": ADAPTIVE_BASELINE_DD_PCT,
        "risk_multiplier_initial": RISK_MULTIPLIER,
        "base_lot": BASE_LOT,
        "theoretical_scaled_lot": SCALED_LOT,
        "event_count": len(core.events),
        "trigger_base": BASE_TRIGGER,
        "add_base": BASE_ADD,
        "reversal_base": BASE_REVERSAL,
        "max_layers": MAX_LAYERS,
        "note": "Initial linear DD calibration; re-run Nautilus and recalibrate from measured DD.",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="XAUUSD M1 OHLC CSV")
    args = parser.parse_args()
    print(json.dumps(dry_event_run(args.csv), indent=2, ensure_ascii=False, default=str))
