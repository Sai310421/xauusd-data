from __future__ import annotations

from decimal import Decimal

import research.amos_allweather_raw_bidask_bt_compat as compat
from research.amos_execution_autofit import ExecutionAutoFit

base = compat.base


class AutoFitCompatStrat(compat.CompatStrat):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.autofit = ExecutionAutoFit(tf_minutes=int(cfg.tf_minutes))
        self.latest_fit = None
        self.lifecycle.update({
            'market_ready_quotes': 0,
            'market_not_ready_quotes': 0,
            'spread_gate_blocks': 0,
            'autofit_orders_submitted': 0,
        })

    def _submit_delayed_intent(self):
        intent = self.entry_intent
        if intent is None:
            return False
        if self.quote_seq <= intent['quote_seq']:
            return False
        fit = self.latest_fit
        if fit is None or not fit.market_ready:
            return False

        self.entry_intent = None
        if self.entry_ref is not None or self.order_pending or not self._is_flat():
            self.lifecycle['entry_intents_expired'] += 1
            return False

        side = intent['side']
        ins = self.cache.instrument(self.config.instrument_id)
        if ins is None:
            self.lifecycle['entry_intents_expired'] += 1
            return False

        base_qty = Decimal(str(self.config.trade_size))
        qty_value = base_qty * Decimal(str(fit.size_multiplier))
        min_inc = Decimal(str(ins.size_increment))
        if qty_value < min_inc:
            qty_value = min_inc
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=base.OrderSide.BUY if side > 0 else base.OrderSide.SELL,
            quantity=ins.make_qty(qty_value),
        )
        self.order_pending = True
        self.pending_side = side
        self.pending_scene = intent['scene']
        self.pending_atr = intent['atr']
        self.lifecycle['orders_submitted'] += 1
        self.lifecycle['autofit_orders_submitted'] += 1
        self.entries += 1
        self.submit_order(order)
        return True

    def on_quote_tick(self, tick):
        self.quote_seq += 1
        bid, ask = self.f(tick.bid_price), self.f(tick.ask_price)
        mid = (bid + ask) / 2
        if self.last_mid is not None:
            self.rets.append(mid - self.last_mid)
        self.last_mid = mid
        self.spreads.append(max(ask - bid, 0))
        self.last_bid, self.last_ask = bid, ask

        self.latest_fit = self.autofit.observe(
            tick.bid_price,
            tick.ask_price,
            getattr(tick, 'bid_size', None),
            getattr(tick, 'ask_size', None),
        )
        if self.latest_fit.market_ready:
            self.lifecycle['market_ready_quotes'] += 1
        else:
            self.lifecycle['market_not_ready_quotes'] += 1
            if self.latest_fit.reason == 'spread_gate':
                self.lifecycle['spread_gate_blocks'] += 1

        if self._submit_delayed_intent():
            return

        x = self.latest_x
        if x is None:
            return
        d = self.decision

        if (
            self.latest_fit.market_ready
            and self.entry_ref is None
            and not self.order_pending
            and self.entry_intent is None
            and self._is_flat()
        ):
            side = self.direction(x, d)
            if side and d.confidence >= .65 and d.scene not in (base.Scene.TRANSITION, base.Scene.NOISE, base.Scene.NEWS):
                av = self.atrs()
                self._queue_entry_intent(side, d.scene.value, av[-1] if av else None)
                return

        if self.entry_ref is None or self.exit_pending:
            return
        px = bid if self.entry_side > 0 else ask
        fav = (px - self.entry_ref) * self.entry_side
        risk = abs(self.entry_ref - self.stop_ref)
        if fav >= .8 * risk:
            cand = px - self.entry_side * .45 * risk
            self.trail_ref = cand if self.trail_ref is None else (
                max(self.trail_ref, cand) if self.entry_side > 0 else min(self.trail_ref, cand)
            )
        st = self.stop_ref if self.trail_ref is None else (
            max(self.stop_ref, self.trail_ref) if self.entry_side > 0 else min(self.stop_ref, self.trail_ref)
        )
        hit = (px <= st or px >= self.tp_ref) if self.entry_side > 0 else (px >= st or px <= self.tp_ref)
        if hit:
            self.close_all_positions(self.config.instrument_id)
            self.exit_pending = True


def fit_snapshot(strategy: AutoFitCompatStrat) -> dict:
    fit = strategy.latest_fit
    if fit is None:
        return {}
    return {
        'market_ready': fit.market_ready,
        'spread_ok': fit.spread_ok,
        'spread': fit.spread,
        'spread_median': fit.spread_median,
        'spread_limit': fit.spread_limit,
        'volatility': fit.volatility,
        'size_multiplier': fit.size_multiplier,
        'tf_weight': fit.tf_weight,
        'execution_quality': fit.execution_quality,
        'reason': fit.reason,
    }
