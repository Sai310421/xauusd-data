from __future__ import annotations

import numpy as np
import multiedge_nautilus_raw_bt as base


def _atr_fixed(self):
    return float(np.mean(list(self.tr)[-20:])) if self.tr else 0.0


# Keep explicit retest state outside the original class fields.
_orig_init = base.MultiEdgeRawStrategy.__init__
def _init_retest(self, config):
    _orig_init(self, config)
    self.retest = {
        'T3_BREAKOUT_CONT': None,
        'R3_MICRO_BREAK': None,
    }
    self.retest_max_wait = 8
    self.retest_zone_atr = 0.20
    self.retest_hold_atr = 0.08


def _ema_prev(self, arr, n):
    a = list(arr)
    if len(a) < n + 1:
        return None
    alpha = 2.0 / (n + 1.0)
    v = float(a[-(n+1)])
    for x in a[-n:-1]:
        v = alpha * float(x) + (1 - alpha) * v
    return v


def _arm_or_fire(self, edge_id, lookback):
    if len(self.c[1]) < max(70, lookback + 2):
        return 0
    c = np.asarray(self.c[1], float)
    h = np.asarray(self.h[1], float)
    l = np.asarray(self.l[1], float)
    atr = max(self._atr(), 1e-9)
    e21 = self._ema(self.c[1], 21)
    e21p = _ema_prev(self, self.c[1], 21)
    if e21 is None or e21p is None:
        return 0

    ref_hi = float(h[-(lookback+1):-1].max())
    ref_lo = float(l[-(lookback+1):-1].min())
    state = self.retest.get(edge_id)

    # Arm on confirmed close beyond the reference high/low.
    if state is None:
        if c[-1] > ref_hi:
            self.retest[edge_id] = {'side': 1, 'level': ref_hi, 'armed_i': self.m1_i}
            return 0
        if c[-1] < ref_lo:
            self.retest[edge_id] = {'side': -1, 'level': ref_lo, 'armed_i': self.m1_i}
            return 0
        return 0

    side = state['side']
    level = state['level']
    age = self.m1_i - state['armed_i']
    if age > self.retest_max_wait:
        self.retest[edge_id] = None
        return 0

    zone = self.retest_zone_atr * atr
    hold = self.retest_hold_atr * atr
    touched = (l[-1] <= level + zone) if side > 0 else (h[-1] >= level - zone)
    held = (c[-1] >= level - hold) if side > 0 else (c[-1] <= level + hold)
    ema_ok = (c[-1] > e21 and e21 > e21p) if side > 0 else (c[-1] < e21 and e21 < e21p)

    # Re-acceleration: close moves back in breakout direction vs previous close.
    reac = (c[-1] > c[-2]) if side > 0 else (c[-1] < c[-2])

    # Failed hold invalidates setup.
    invalid = (c[-1] < level - 0.35 * atr) if side > 0 else (c[-1] > level + 0.35 * atr)
    if invalid:
        self.retest[edge_id] = None
        return 0

    if touched and held and ema_ok and reac:
        self.retest[edge_id] = None
        self.fire_count += 1
        return side
    return 0


def _signal_retest(self):
    selected = self.config.selected
    if selected == 'T3_BREAKOUT_CONT':
        return _arm_or_fire(self, selected, 20)
    if selected == 'R3_MICRO_BREAK':
        return _arm_or_fire(self, selected, 7)
    if selected == 'TREND_CORE':
        # Retest-led Trend Core: T3 retest is trigger, MTF + EMA direction are filters.
        side = _arm_or_fire(self, 'T3_BREAKOUT_CONT', 20)
        if not side:
            return 0
        s = base.MultiEdgeRawStrategy._edge_scores(self)
        if side > 0 and s.get('T2_MTF_ALIGN', 0) > 0 and s.get('T1_EMA21', 0) > 0:
            return side
        if side < 0 and s.get('T2_MTF_ALIGN', 0) < 0 and s.get('T1_EMA21', 0) < 0:
            return side
        return 0
    if selected == 'RANGE_CORE':
        # Retest-led Range Core: R3 retest trigger + compression/expansion agrees.
        side = _arm_or_fire(self, 'R3_MICRO_BREAK', 7)
        if not side:
            return 0
        s = base.MultiEdgeRawStrategy._edge_scores(self)
        if side > 0 and s.get('R2_COMP_EXP', 0) > 0:
            return side
        if side < 0 and s.get('R2_COMP_EXP', 0) < 0:
            return side
        return 0
    return base.MultiEdgeRawStrategy._signal(self)


base.MultiEdgeRawStrategy.__init__ = _init_retest
base.MultiEdgeRawStrategy._atr = _atr_fixed
base.MultiEdgeRawStrategy._signal = _signal_retest

if __name__ == '__main__':
    base.main()
