from __future__ import annotations

"""OB∩FVG five-depth diagnostic with BE + dynamic ATR trailing exit.

Entry logic is identical to multiedge_ob_fvg_depth_gate.py.
Only the exit engine is replaced so Entry EDGE and Exit EDGE remain separable.

Exit v1:
- Initial hard SL: 0.75 ATR from entry.
- No fixed TP.
- Trail activation: MFE >= 0.75 ATR.
- Once activated, protective floor = BE + 0.05 ATR in trade direction.
- Dynamic trail distance from best price:
    MFE < 1.50 ATR -> 1.20 ATR
    1.50 <= MFE < 2.00 ATR -> 1.00 ATR
    MFE >= 2.00 ATR -> 0.75 ATR
- Horizon: 60 M1 bars.
"""

import multiedge_ob_fvg_depth_gate as base


_orig_open_depth = base.DepthGate._open_depth

def _open_depth_dynamic(self, d, side, px, atr):
    _orig_open_depth(self, d, side, px, atr)
    a = self.active[d]
    if a is not None:
        a['trail_active'] = False
        a['trail_stop'] = None
        a['peak_mfe_atr'] = 0.0


def _on_quote_tick_dynamic(self, tick):
    bid = self._f(tick.bid_price)
    ask = self._f(tick.ask_price)
    self.last_bid = bid
    self.last_ask = ask

    # Entry logic: unchanged from the fixed-TP depth gate.
    if self.zone:
        side = self.zone['side']
        probe = ask if side > 0 else bid
        for d, level in self.zone['levels'].items():
            if d in self.zone['filled']:
                continue
            touched = probe <= level if side > 0 else probe >= level
            if touched:
                self.zone['filled'].add(d)
                self._open_depth(d, side, probe, self.zone['atr'])
        if (side > 0 and bid < self.zone['lo'] - 0.35 * self.zone['atr']) or (side < 0 and ask > self.zone['hi'] + 0.35 * self.zone['atr']):
            self.zone = None

    # Exit logic: initial SL + BE floor + dynamic ATR trail. No fixed TP.
    for d, a in list(self.active.items()):
        if a is None:
            continue
        side = a['side']
        mark = bid if side > 0 else ask
        if side > 0:
            a['best'] = max(a['best'], mark)
            a['worst'] = min(a['worst'], mark)
            mfe = max(0.0, a['best'] - a['entry'])
        else:
            a['best'] = min(a['best'], mark)
            a['worst'] = max(a['worst'], mark)
            mfe = max(0.0, a['entry'] - a['best'])

        atr = max(a['atr'], 1e-9)
        mfe_atr = mfe / atr
        a['peak_mfe_atr'] = max(a.get('peak_mfe_atr', 0.0), mfe_atr)

        # Initial protective stop remains until the trail activates.
        initial_stop = a['entry'] - 0.75 * atr if side > 0 else a['entry'] + 0.75 * atr
        if not a.get('trail_active', False):
            if (side > 0 and mark <= initial_stop) or (side < 0 and mark >= initial_stop):
                self._close_depth(d, bid, ask, 'INITIAL_SL')
                continue
            if mfe_atr >= 0.75:
                a['trail_active'] = True

        if a.get('trail_active', False):
            # Tighten as favorable excursion expands.
            if mfe_atr < 1.50:
                k = 1.20
            elif mfe_atr < 2.00:
                k = 1.00
            else:
                k = 0.75

            if side > 0:
                be_floor = a['entry'] + 0.05 * atr
                candidate = a['best'] - k * atr
                new_stop = max(be_floor, candidate)
                prev = a.get('trail_stop')
                a['trail_stop'] = new_stop if prev is None else max(prev, new_stop)
                if bid <= a['trail_stop']:
                    self._close_depth(d, bid, ask, 'DYNAMIC_ATR_TRAIL')
                    continue
            else:
                be_floor = a['entry'] - 0.05 * atr
                candidate = a['best'] + k * atr
                new_stop = min(be_floor, candidate)
                prev = a.get('trail_stop')
                a['trail_stop'] = new_stop if prev is None else min(prev, new_stop)
                if ask >= a['trail_stop']:
                    self._close_depth(d, bid, ask, 'DYNAMIC_ATR_TRAIL')
                    continue

        if self.m1_i - a['entry_i'] >= self.config.horizon_minutes:
            self._close_depth(d, bid, ask, 'HORIZON')


base.DepthGate._open_depth = _open_depth_dynamic
base.DepthGate.on_quote_tick = _on_quote_tick_dynamic

if __name__ == '__main__':
    base.main()
