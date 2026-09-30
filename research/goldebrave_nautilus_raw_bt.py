from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import nautilus_trader
import numpy as np
import pandas as pd
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue("SIM")


@dataclass
class B:
    ts_ns: int
    o: float
    h: float
    l: float
    c: float


@dataclass
class Pending:
    side: int
    price: float
    sl: float
    tp: float
    layer: str
    created_ns: int


@dataclass
class Pos:
    side: int
    entry: float
    sl: float
    tp: float
    layer: str
    opened_ns: int


class GoldeBraveConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    m1: BarType
    m15: BarType
    h1: BarType
    initial_balance: float = 100000.0
    lot: float = 0.10
    contract_size: float = 100.0
    trade_hours: tuple[int, ...] = (11, 15, 16, 17, 18)
    gmt: int = 2
    gmts: int = 3
    dst_area: int = 0
    zz_depth: int = 12
    zz_deviation: int = 5
    zz_backstep: int = 3
    lookback: int = 600
    levels_a: int = 5
    levels_b: int = 3
    fs_units: float = 3.0
    min_dist_add_units: float = 1.0
    unit: float = 0.10
    use_atr_sltp: bool = True
    sl_atr_mult: float = 1.2
    tp_atr_mult: float = 2.4
    sl_units: float = 40.0
    tp_units: float = 90.0
    sl_max_units: float = 120.0
    tp_max_units: float = 300.0
    min_entries_per_day: int = 3
    boost_hour: int = 9
    boost_off_units: float = 2.0
    spread_break_units: float = 25.0
    max_pending_per_side: int = 4
    be_trigger_units: float = 12.0
    be_lock_units: float = 2.0
    tr_trigger_units: float = 20.0
    tr_offset_units: float = 30.0
    tr_band: float = 0.3
    adapt: bool = True
    vol_short: int = 14
    vol_long: int = 480
    vol_min_ratio: float = 0.6
    vol_max_ratio: float = 2.5
    adx_period: int = 14
    adx_trend: float = 25.0
    adx_range: float = 18.0
    trend_extra_levels: int = 2
    trend_tp_mult: float = 1.25
    range_tp_mult: float = 0.8
    range_boost_mult: float = 2.0
    range_disable_c: bool = False


class GoldeBraveStrategy(Strategy):
    """Causal Raw-BidAsk recreation of GoldeBrave_v4.mq5.

    Raw QuoteTicks drive execution. Nautilus INTERNAL M1/M15/H1 bars are built from the
    same raw stream and drive the MT5-equivalent indicator/state layer. Stop orders are
    modeled explicitly and fill against observed raw ask/bid on crossing. This avoids
    OHLC-resample execution and keeps spread path dependence.
    """

    def __init__(self, config: GoldeBraveConfig):
        super().__init__(config)
        self.bars = {"M1": deque(maxlen=2400), "M15": deque(maxlen=2400), "H1": deque(maxlen=2400)}
        self.pending: list[Pending] = []
        self.positions: list[Pos] = []
        self.placed: list[float] = []
        self.day_key = None
        self.entries_today = 0
        self.last_boost_bar = None
        self.last_build_day = None
        self.last_trail_m1 = None
        self.g_vol = 1.0
        self.g_adx = 0.0
        self.regime = 0
        self.spread_break = False
        self.break_started_ns = None
        self.last_bid = None
        self.last_ask = None
        self.realized = 0.0
        self.equity_peak = config.initial_balance
        self.max_dd_pct = 0.0
        self.gross_win = 0.0
        self.gross_loss = 0.0
        self.wins = 0
        self.losses = 0
        self.closed = 0
        self.layer_entries = {"A_H1": 0, "B_M15": 0, "C_DAILY": 0}
        self.layer_pnl = {"A_H1": 0.0, "B_M15": 0.0, "C_DAILY": 0.0}
        self.max_positions = 0
        self.max_pending = 0
        self.spread_break_events = 0
        self.be_moves = 0
        self.trail_moves = 0
        self.stop_fills = 0
        self.rejected_stops = 0
        self.first_ns = None
        self.last_ns = None

    @staticmethod
    def f(x):
        return float(x.as_double()) if hasattr(x, "as_double") else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.m1)
        self.subscribe_bars(self.config.m15)
        self.subscribe_bars(self.config.h1)

    def _dt(self, ns: int) -> datetime:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)

    @staticmethod
    def _us_dst(dt: datetime) -> bool:
        # EA's deliberately simple day-of-week formulation, reproduced rather than corrected.
        mo, dd = dt.month, dt.day
        # Python Monday=0; MQL Sunday=0.
        dw = (dt.weekday() + 1) % 7
        if 3 < mo < 11:
            return True
        if mo == 3:
            return dd >= 14 - dw
        if mo == 11:
            return dd < 7 - dw
        return False

    @staticmethod
    def _eu_dst(dt: datetime) -> bool:
        mo, dd = dt.month, dt.day
        dw = (dt.weekday() + 1) % 7
        if 3 < mo < 10:
            return True
        if mo == 3:
            return dd >= 31 - dw
        if mo == 10:
            return dd < 31 - dw
        return False

    def shifted_hour(self, ns: int) -> int:
        dt = self._dt(ns)
        dst = self._eu_dst(dt) if self.config.dst_area == 1 else self._us_dst(dt)
        h = dt.hour + ((3 - self.config.gmts) if dst else (2 - self.config.gmt))
        return h % 24

    def _day_key(self, ns: int):
        dt = self._dt(ns)
        # Mirror EA's H1(g_hour) key behavior: operational day anchored to shifted hour.
        h = self.shifted_hour(ns)
        return (dt.year, dt.month, dt.day, h)

    def _calendar_day(self, ns: int):
        dt = self._dt(ns)
        return dt.year, dt.month, dt.day

    def _bar(self, bar: Bar) -> B:
        return B(int(bar.ts_event), self.f(bar.open), self.f(bar.high), self.f(bar.low), self.f(bar.close))

    def on_bar(self, bar: Bar):
        bt = str(bar.bar_type)
        key = "M1" if "1-MINUTE" in bt else "M15" if "15-MINUTE" in bt else "H1"
        b = self._bar(bar)
        self.bars[key].append(b)
        if key == "H1":
            self._update_adapt()
            if self.last_bid is not None:
                self._session_rebuild(b.ts_ns, "H1")
        elif key == "M15":
            if self.last_bid is not None:
                self._session_rebuild(b.ts_ns, "M15")
                self._boost(b.ts_ns)
        elif key == "M1":
            if self.last_bid is not None:
                self._trail_once(b.ts_ns)

    def _atr(self, bars, n):
        if len(bars) < n + 1:
            return None
        xs = list(bars)[-(n + 1):]
        trs = []
        for i in range(1, len(xs)):
            prev = xs[i - 1].c
            z = xs[i]
            trs.append(max(z.h - z.l, abs(z.h - prev), abs(z.l - prev)))
        return float(np.mean(trs[-n:]))

    def _adx(self, bars, n):
        if len(bars) < 2 * n + 2:
            return None
        xs = list(bars)[-(2 * n + 2):]
        tr, pdm, mdm = [], [], []
        for i in range(1, len(xs)):
            a, b = xs[i - 1], xs[i]
            up, dn = b.h - a.h, a.l - b.l
            pdm.append(up if up > dn and up > 0 else 0.0)
            mdm.append(dn if dn > up and dn > 0 else 0.0)
            tr.append(max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c)))
        dx = []
        for i in range(n - 1, len(tr)):
            atr = sum(tr[i - n + 1:i + 1])
            if atr <= 0:
                continue
            pdi = 100 * sum(pdm[i - n + 1:i + 1]) / atr
            mdi = 100 * sum(mdm[i - n + 1:i + 1]) / atr
            den = pdi + mdi
            dx.append(0.0 if den <= 0 else 100 * abs(pdi - mdi) / den)
        return float(np.mean(dx[-n:])) if len(dx) >= n else None

    def _update_adapt(self):
        self.g_vol, self.g_adx, self.regime = 1.0, 0.0, 0
        if not self.config.adapt:
            return
        a = self._atr(self.bars["H1"], self.config.vol_short)
        b = self._atr(self.bars["H1"], self.config.vol_long)
        if a is not None and b is not None and b > 0:
            self.g_vol = min(max(a / b, self.config.vol_min_ratio), self.config.vol_max_ratio)
        x = self._adx(self.bars["H1"], self.config.adx_period)
        if x is not None:
            self.g_adx = x
            if x >= self.config.adx_trend:
                self.regime = 1
            elif x <= self.config.adx_range:
                self.regime = -1

    def off(self):
        return self.config.fs_units * self.config.unit * self.g_vol

    def min_dist(self):
        return (self.config.fs_units + self.config.min_dist_add_units) * self.config.unit * self.g_vol

    def sl_dist(self):
        base = self.config.sl_units * self.config.unit
        if not self.config.use_atr_sltp:
            return base
        a = self._atr(self.bars["H1"], self.config.vol_short)
        if a is None:
            return base
        return min(max(a * self.config.sl_atr_mult, base), self.config.sl_max_units * self.config.unit)

    def tp_mult(self):
        if not self.config.adapt:
            return 1.0
        return self.config.trend_tp_mult if self.regime == 1 else self.config.range_tp_mult if self.regime == -1 else 1.0

    def tp_dist(self):
        base = self.config.tp_units * self.config.unit
        m = self.tp_mult()
        if not self.config.use_atr_sltp:
            return base * m
        a = self._atr(self.bars["H1"], self.config.vol_short)
        if a is None:
            return base * m
        return min(max(a * self.config.tp_atr_mult, base), self.config.tp_max_units * self.config.unit) * m

    def _day_range(self, ns: int):
        day = self._calendar_day(ns)
        xs = [b for b in self.bars["M15"] if self._calendar_day(b.ts_ns) == day]
        if not xs:
            return None
        return max(x.h for x in xs), min(x.l for x in xs)

    def _zigzag(self, bars: list[B]):
        """Examples/ZigZag-compatible state reconstruction over currently-known bars.

        Uses Depth/Deviation/Backstep extrema maps and final peak/trough alternation. It is
        intentionally recomputed from the current causal history so recent pivots may repaint,
        as they do in the MT5 example indicator.
        """
        depth, dev, back = self.config.zz_depth, self.config.zz_deviation * 0.001, self.config.zz_backstep
        n = len(bars)
        if n < depth + 3:
            return []
        lowmap = [0.0] * n
        highmap = [0.0] * n
        for i in range(depth - 1, n):
            window = bars[i - depth + 1:i + 1]
            lo = min(x.l for x in window)
            hi = max(x.h for x in window)
            if bars[i].l == lo and (i == 0 or abs(bars[i].l - lo) <= dev):
                lowmap[i] = lo
                for j in range(1, back + 1):
                    k = i - j
                    if k >= 0 and lowmap[k] != 0.0 and lowmap[k] > lo:
                        lowmap[k] = 0.0
            if bars[i].h == hi and (i == 0 or abs(bars[i].h - hi) <= dev):
                highmap[i] = hi
                for j in range(1, back + 1):
                    k = i - j
                    if k >= 0 and highmap[k] != 0.0 and highmap[k] < hi:
                        highmap[k] = 0.0
        piv = []
        last_kind = None
        last_i = None
        for i in range(n):
            lo, hi = lowmap[i], highmap[i]
            if lo and hi:
                # Prefer the more extreme move relative to prior pivot.
                if not piv:
                    kind, px = ("H", hi) if bars[i].c < (hi + lo) / 2 else ("L", lo)
                else:
                    prev = piv[-1][2]
                    kind, px = ("H", hi) if abs(hi - prev) >= abs(lo - prev) else ("L", lo)
            elif hi:
                kind, px = "H", hi
            elif lo:
                kind, px = "L", lo
            else:
                continue
            if last_kind == kind and piv:
                better = px > piv[-1][2] if kind == "H" else px < piv[-1][2]
                if better:
                    piv[-1] = (i, kind, px)
                    last_i = i
                continue
            piv.append((i, kind, px))
            last_kind, last_i = kind, i
        return piv

    def _pending_count(self, side):
        return sum(1 for x in self.pending if x.side == side)

    def _near(self, side, price):
        return any(x.side == side and abs(x.price - price) < self.min_dist() for x in self.pending)

    def _placed_has(self, price):
        return any(abs(x - price) < self.min_dist() for x in self.placed)

    def _place(self, side, price, layer, ns, track_placed=False):
        if self.config.max_pending_per_side > 0 and self._pending_count(side) >= self.config.max_pending_per_side:
            self.rejected_stops += 1
            return False
        if self.last_ask is None:
            return False
        # MT5 SYMBOL_TRADE_STOPS_LEVEL is broker-specific; absent from generic raw feed.
        if side > 0 and price <= self.last_ask:
            self.rejected_stops += 1
            return False
        if side < 0 and price >= self.last_bid:
            self.rejected_stops += 1
            return False
        sl, tp = self.sl_dist(), self.tp_dist()
        self.pending.append(Pending(side, price, price - side * sl, price + side * tp, layer, ns))
        if track_placed:
            self.placed.append(price)
        self.max_pending = max(self.max_pending, len(self.pending))
        return True

    def _delete_pending(self):
        self.pending.clear()

    def _scan(self, tf: str, cap: int, ns: int, layer: str):
        dr = self._day_range(ns)
        if dr is None:
            return
        dhi, dlo = dr
        hibar, lobar = dhi + self.min_dist(), dlo - self.min_dist()
        xs = list(self.bars[tf])[-(self.config.lookback + 1):]
        piv = self._zigzag(xs)
        nb = nsell = 0
        # EA arrays are series and scan shift 2 -> old, so newest confirmed/repainting bars first.
        for i, kind, px in reversed(piv[:-2] if len(piv) > 2 else []):
            if kind == "H" and px > hibar and nb < cap:
                nb += 1
                p = px - self.off()
                if not self._placed_has(p):
                    self._place(+1, p, layer, ns, track_placed=True)
                hibar = px
            elif kind == "L" and px < lobar and nsell < cap:
                nsell += 1
                q = px + self.off()
                if not self._placed_has(q):
                    self._place(-1, q, layer, ns, track_placed=True)
                lobar = px

    def _session_rebuild(self, ns: int, changed_tf: str):
        h = self.shifted_hour(ns)
        in_session = h in self.config.trade_hours
        dk = self._day_key(ns)
        if self.day_key != dk:
            self.day_key = dk
            self.placed.clear()
        if not in_session or self.spread_break:
            if not in_session:
                self._delete_pending()
            return
        if self.last_build_day != dk:
            self._delete_pending()
            cap = self.config.levels_a + (self.config.trend_extra_levels if self.config.adapt and self.regime == 1 else 0)
            self._scan("H1", cap, ns, "A_H1")
            self._scan("M15", self.config.levels_b, ns, "B_M15")
            self.last_build_day = dk
            return
        if changed_tf == "H1":
            cap = self.config.levels_a + (self.config.trend_extra_levels if self.config.adapt and self.regime == 1 else 0)
            self._scan("H1", cap, ns, "A_H1")
        elif changed_tf == "M15":
            self._scan("M15", self.config.levels_b, ns, "B_M15")

    def _boost(self, ns):
        h = self.shifted_hour(ns)
        if h not in self.config.trade_hours or h < self.config.boost_hour:
            return
        if self.config.min_entries_per_day <= 0 or self.entries_today >= self.config.min_entries_per_day:
            return
        if self.config.adapt and self.config.range_disable_c and self.regime == -1:
            return
        dr = self._day_range(ns)
        if dr is None:
            return
        dhi, dlo = dr
        off = self.config.boost_off_units * self.config.unit * self.g_vol
        if self.config.adapt and self.regime == -1:
            off *= self.config.range_boost_mult
        p, q = dhi + off, dlo - off
        if not self._near(+1, p):
            self._place(+1, p, "C_DAILY", ns)
        if not self._near(-1, q):
            self._place(-1, q, "C_DAILY", ns)

    def _open(self, pend: Pending, fill: float, ns: int):
        shift = fill - pend.price
        pos = Pos(pend.side, fill, pend.sl + shift, pend.tp + shift, pend.layer, ns)
        self.positions.append(pos)
        self.entries_today += 1
        self.layer_entries[pend.layer] += 1
        self.stop_fills += 1
        self.max_positions = max(self.max_positions, len(self.positions))

    def _close(self, idx: int, px: float, reason: str):
        p = self.positions.pop(idx)
        pnl = (px - p.entry) * p.side * self.config.contract_size * self.config.lot
        self.realized += pnl
        self.closed += 1
        self.layer_pnl[p.layer] += pnl
        if pnl > 0:
            self.wins += 1
            self.gross_win += pnl
        elif pnl < 0:
            self.losses += 1
            self.gross_loss += abs(pnl)

    def _mark(self):
        if self.last_bid is None:
            return 0.0
        return sum(((self.last_bid if p.side > 0 else self.last_ask) - p.entry) * p.side * self.config.contract_size * self.config.lot for p in self.positions)

    def _update_dd(self):
        eq = self.config.initial_balance + self.realized + self._mark()
        self.equity_peak = max(self.equity_peak, eq)
        dd = max(0.0, (self.equity_peak - eq) / max(self.equity_peak, 1e-9) * 100)
        self.max_dd_pct = max(self.max_dd_pct, dd)

    def _trail_once(self, ns):
        if not self.positions or len(self.bars["M1"]) < 2 or not self.bars["H1"]:
            return
        if self.last_trail_m1 == ns:
            return
        self.last_trail_m1 = ns
        m1 = list(self.bars["M1"])[-2]
        h1 = self.bars["H1"][-1]
        band, ofs, trig = self.config.tr_band * self.g_vol, self.config.tr_offset_units * self.config.unit * self.g_vol, self.config.tr_trigger_units * self.config.unit * self.g_vol
        gate = m1.h >= h1.h - band or m1.l <= h1.l + band
        if not gate:
            return
        for p in self.positions:
            if p.side > 0 and p.entry + trig < m1.h:
                nsx = m1.h - ofs
                if nsx > p.sl:
                    p.sl = nsx
                    self.trail_moves += 1
            elif p.side < 0 and p.entry - trig > m1.l:
                nsx = m1.l + ofs
                if nsx < p.sl:
                    p.sl = nsx
                    self.trail_moves += 1

    def _be(self):
        trig, lock = self.config.be_trigger_units * self.config.unit * self.g_vol, self.config.be_lock_units * self.config.unit * self.g_vol
        for p in self.positions:
            if p.side > 0 and self.last_bid - p.entry >= trig:
                ns = p.entry + lock
                if p.sl < ns:
                    p.sl = ns
                    self.be_moves += 1
            elif p.side < 0 and p.entry - self.last_ask >= trig:
                ns = p.entry - lock
                if p.sl > ns:
                    p.sl = ns
                    self.be_moves += 1

    def on_quote_tick(self, tick: QuoteTick):
        ns = int(tick.ts_event)
        self.first_ns = ns if self.first_ns is None else self.first_ns
        self.last_ns = ns
        self.last_bid, self.last_ask = self.f(tick.bid_price), self.f(tick.ask_price)
        spread_units = (self.last_ask - self.last_bid) / self.config.unit
        h = self.shifted_hour(ns)
        if h in self.config.trade_hours:
            if spread_units > self.config.spread_break_units:
                if not self.spread_break:
                    self._delete_pending()
                    self.spread_break_events += 1
                self.spread_break = True
                self.break_started_ns = ns
            elif self.spread_break and self.break_started_ns is not None and ns - self.break_started_ns >= 120_000_000_000 and spread_units < self.config.spread_break_units * 0.5:
                self.spread_break = False
                self.last_build_day = None
        # Pending crossing against true executable side.
        keep = []
        for x in self.pending:
            hit = self.last_ask >= x.price if x.side > 0 else self.last_bid <= x.price
            if hit and h in self.config.trade_hours and not self.spread_break:
                fill = self.last_ask if x.side > 0 else self.last_bid
                self._open(x, fill, ns)
            else:
                keep.append(x)
        self.pending = keep
        self._be()
        # SL first, then TP: conservative if a quote gaps across both impossible on same quote.
        for i in range(len(self.positions) - 1, -1, -1):
            p = self.positions[i]
            if p.side > 0:
                if self.last_bid <= p.sl:
                    self._close(i, self.last_bid, "SL")
                elif self.last_bid >= p.tp:
                    self._close(i, self.last_bid, "TP")
            else:
                if self.last_ask >= p.sl:
                    self._close(i, self.last_ask, "SL")
                elif self.last_ask <= p.tp:
                    self._close(i, self.last_ask, "TP")
        self._update_dd()

    def on_stop(self):
        if self.last_bid is not None:
            for i in range(len(self.positions) - 1, -1, -1):
                p = self.positions[i]
                self._close(i, self.last_bid if p.side > 0 else self.last_ask, "EOD")
        self._delete_pending()

    def summary(self):
        pf = self.gross_win / self.gross_loss if self.gross_loss > 0 else (math.inf if self.gross_win > 0 else 0.0)
        wr = self.wins / max(self.closed, 1) * 100
        return {
            "schema": "amos.arena.v1",
            "strategy_id": "GoldeBrave_v4",
            "agent": "external_ea_reconstruction",
            "initial_equity": self.config.initial_balance,
            "final_equity": self.config.initial_balance + self.realized,
            "net_profit": self.realized,
            "return_pct": self.realized / self.config.initial_balance * 100,
            "max_dd_pct": self.max_dd_pct,
            "pf": pf,
            "win_rate_pct": wr,
            "trades": self.closed,
            "ruin_probability_pct": None,
            "oos_retention": None,
            "cost_retention": None,
            "regime_retention": None,
            "layer_entries": self.layer_entries,
            "layer_pnl": self.layer_pnl,
            "max_positions": self.max_positions,
            "max_pending": self.max_pending,
            "spread_break_events": self.spread_break_events,
            "be_moves": self.be_moves,
            "trail_moves": self.trail_moves,
            "stop_fills": self.stop_fills,
            "rejected_stops": self.rejected_stops,
            "period_start_utc": self._dt(self.first_ns).isoformat() if self.first_ns else None,
            "period_end_utc": self._dt(self.last_ns).isoformat() if self.last_ns else None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--raw-bidask-only", action="store_true")
    ap.add_argument("--initial-balance", type=float, default=100000.0)
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit("raw-bidask-only is mandatory: OHLC-resample execution is prohibited")
    cp = Path(args.catalog)
    catalog = ParquetDataCatalog(str(cp))
    instrument = next((x for x in catalog.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD"), None)
    if instrument is None:
        raise SystemExit("XAUUSD missing")
    ticks = catalog.query_quote_ticks(identifiers=[instrument.id.value])
    if not ticks:
        raise SystemExit("no raw XAUUSD QuoteTicks")
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"), risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=SIM, oms_type=OmsType.HEDGING, account_type=AccountType.MARGIN, base_currency=USD, starting_balances=[Money(args.initial_balance, USD)], default_leverage=Decimal("2000"))
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    iid = instrument.id.value
    m1 = BarType.from_str(f"{iid}-1-MINUTE-BID-INTERNAL")
    m15 = BarType.from_str(f"{iid}-15-MINUTE-BID-INTERNAL")
    h1 = BarType.from_str(f"{iid}-60-MINUTE-BID-INTERNAL")
    st = GoldeBraveStrategy(GoldeBraveConfig(instrument_id=instrument.id, m1=m1, m15=m15, h1=h1, initial_balance=args.initial_balance))
    engine.add_strategy(st)
    engine.run()
    summary = st.summary()
    summary.update({
        "verification_level": "NAUTILUS_RAW_BIDASK_CAUSAL_RECONSTRUCTION",
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "raw_ticks": len(ticks),
        "data_kind": "RAW_BIDASK QuoteTick",
        "ohlc_resample_used": False,
        "signal_bars": "Nautilus INTERNAL M1/M15/H1 BID bars built causally from raw QuoteTicks",
        "execution": "Virtual MT5 stop-order ledger filled on observed raw ask/bid crossings; SL/TP/BE on executable quote side",
        "source_parity": {
            "EA": "GoldeBrave_v4.mq5 v4.20",
            "layers": {"A_H1": "PivotTF H1 ZigZag", "B_M15": "FastTF M15 ZigZag", "C_DAILY": "daily high/low minimum-entry boost"},
            "inputs": "v4.20 defaults",
        },
        "known_parity_risks": [
            "Generic Dukascopy feed has no MT5 broker SYMBOL_TRADE_STOPS_LEVEL; only executable-side validity is enforced.",
            "MT5 Examples/ZigZag is recreated causally with Depth/Deviation/Backstep and repainting extrema state; exact buffer parity should be checked against exported MT5 pivot timestamps.",
            "Server-time equality depends on the MT5 report broker timezone/DST inputs; defaults GMT=2 GMTS=3 are preserved.",
            "Commission and broker-specific slippage are not inferred without the original MT5 tester report/settings.",
        ],
    })
    out = Path("results/goldebrave") / args.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_json(out / "arena_result.json", orient="records", indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    engine.dispose()


if __name__ == "__main__":
    main()
