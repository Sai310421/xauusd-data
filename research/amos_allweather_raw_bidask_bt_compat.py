from __future__ import annotations
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick


def _install_quote_tick_compat() -> str:
    if hasattr(ParquetDataCatalog, 'query_quote_ticks'):
        return 'query_quote_ticks'
    if hasattr(ParquetDataCatalog, 'quote_ticks'):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quote_ticks(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return 'quote_ticks'
    if hasattr(ParquetDataCatalog, 'quotes'):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quotes(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return 'quotes'
    if hasattr(ParquetDataCatalog, 'query'):
        def query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = query_quote_ticks
        return 'query(QuoteTick)'
    raise RuntimeError('No Raw QuoteTick reader found on ParquetDataCatalog')


CATALOG_QUOTE_API = _install_quote_tick_compat()
print(f'CATALOG_QUOTE_API={CATALOG_QUOTE_API}')

import research.amos_allweather_raw_bidask_bt as base


class CompatStrat(base.Strat):
    def _is_flat(self) -> bool:
        p = self.portfolio
        if hasattr(p, 'is_net_flat'):
            return bool(p.is_net_flat(self.config.instrument_id))
        if hasattr(p, 'is_net_long') and hasattr(p, 'is_net_short'):
            return (not p.is_net_long(self.config.instrument_id)) and (not p.is_net_short(self.config.instrument_id))
        return self.entry_ref is None

    def on_quote_tick(self, tick):
        bid, ask = self.f(tick.bid_price), self.f(tick.ask_price)
        mid = (bid + ask) / 2
        if self.last_mid is not None:
            self.rets.append(mid - self.last_mid)
        self.last_mid = mid
        self.spreads.append(max(ask - bid, 0))
        self.last_bid, self.last_ask = bid, ask
        x = self.features()
        if x is None:
            return
        d = self.decision
        if self.entry_ref is None and self._is_flat():
            side = self.direction(x, d)
            if side and d.confidence >= .65 and d.scene not in (base.Scene.TRANSITION, base.Scene.NOISE, base.Scene.NEWS):
                ins = self.cache.instrument(self.config.instrument_id)
                order = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=base.OrderSide.BUY if side > 0 else base.OrderSide.SELL,
                    quantity=ins.make_qty(self.config.trade_size),
                )
                self.submit_order(order)
                self.entry_ref = ask if side > 0 else bid
                self.entry_side = side
                self.entry_scenes.append(d.scene.value)
                atr = self.atrs()[-1]
                if d.scene in (base.Scene.BALANCED_RANGE, base.Scene.LIQUIDITY_BUILD, base.Scene.COMPRESSION):
                    sk, tk = .75, .55
                elif d.scene in (base.Scene.REVERSAL, base.Scene.BREAKOUT, base.Scene.CONTINUATION, base.Scene.CRISIS):
                    sk, tk = 1., 1.8
                else:
                    sk, tk = .9, 1.2
                self.stop_ref = self.entry_ref - side * sk * atr
                self.tp_ref = self.entry_ref + side * tk * atr
                self.trail_ref = None
                self.hold = 0
                self.exit_pending = False
                self.entries += 1
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


base.Strat = CompatStrat
print('PORTFOLIO_FLAT_API=strategy_compat')

if __name__ == '__main__':
    base.main()
