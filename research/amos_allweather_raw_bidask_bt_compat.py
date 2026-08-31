from __future__ import annotations
import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

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
    def __init__(self, cfg):
        super().__init__(cfg)
        self.latest_x = None
        self.active_scene = 'unknown'
        self.closed_trades = []
        self.order_pending = False
        self.pending_side = 0
        self.pending_scene = 'unknown'
        self.pending_atr = None
        self.entry_intent = None
        self.quote_seq = 0
        self.rejection_reasons = Counter()
        self.rejection_samples = []
        self.lifecycle = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'entry_intents_queued': 0,
            'entry_intents_expired': 0,
        }

    def _is_flat(self) -> bool:
        p = self.portfolio
        if hasattr(p, 'is_net_flat'):
            return bool(p.is_net_flat(self.config.instrument_id))
        if hasattr(p, 'is_net_long') and hasattr(p, 'is_net_short'):
            return (not p.is_net_long(self.config.instrument_id)) and (not p.is_net_short(self.config.instrument_id))
        return self.entry_ref is None

    def _reset_pending(self):
        self.order_pending = False
        self.pending_side = 0
        self.pending_scene = 'unknown'
        self.pending_atr = None

    def _queue_entry_intent(self, side, scene, atr):
        self.entry_intent = {
            'side': int(side),
            'scene': str(scene),
            'atr': atr,
            'quote_seq': self.quote_seq,
        }
        self.lifecycle['entry_intents_queued'] += 1

    def _submit_delayed_intent(self):
        intent = self.entry_intent
        if intent is None:
            return False
        if self.quote_seq <= intent['quote_seq']:
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
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=base.OrderSide.BUY if side > 0 else base.OrderSide.SELL,
            quantity=ins.make_qty(self.config.trade_size),
        )
        self.order_pending = True
        self.pending_side = side
        self.pending_scene = intent['scene']
        self.pending_atr = intent['atr']
        self.lifecycle['orders_submitted'] += 1
        self.entries += 1
        self.submit_order(order)
        return True

    def on_bar(self, bar):
        super().on_bar(bar)
        self.latest_x = self.features()

    def on_quote_tick(self, tick):
        self.quote_seq += 1
        bid, ask = self.f(tick.bid_price), self.f(tick.ask_price)
        mid = (bid + ask) / 2
        if self.last_mid is not None:
            self.rets.append(mid - self.last_mid)
        self.last_mid = mid
        self.spreads.append(max(ask - bid, 0))
        self.last_bid, self.last_ask = bid, ask

        # Important for Nautilus 1.230 L1 execution: the strategy callback can run
        # before the simulated exchange has promoted the same QuoteTick into its
        # matching core. Submit a signal only from the following QuoteTick so the
        # previous raw Bid/Ask has already initialized core.bid/core.ask.
        if self._submit_delayed_intent():
            return

        x = self.latest_x
        if x is None:
            return
        d = self.decision

        if self.entry_ref is None and not self.order_pending and self.entry_intent is None and self._is_flat():
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

    def on_order_filled(self, event):
        self.lifecycle['orders_filled'] += 1

    def on_order_rejected(self, event):
        self.lifecycle['orders_rejected'] += 1
        reason = str(getattr(event, 'reason', None) or getattr(event, 'message', None) or 'UNKNOWN')
        self.rejection_reasons[reason] += 1
        if len(self.rejection_samples) < 5:
            self.rejection_samples.append(reason)
            print(f'ORDER_REJECT_SAMPLE={reason}')
        self._reset_pending()
        self.entry_ref = None
        self.entry_side = 0
        self.exit_pending = False

    def on_position_opened(self, event):
        self.lifecycle['positions_opened'] += 1
        side = self.pending_side
        if side == 0:
            side = 1 if self.portfolio.is_net_long(self.config.instrument_id) else -1
        px = getattr(event, 'avg_px_open', None)
        if px is None:
            px = self.last_ask if side > 0 else self.last_bid
        self.entry_ref = float(px)
        self.entry_side = side
        self.active_scene = self.pending_scene
        self.entry_scenes.append(self.active_scene)
        atr = self.pending_atr
        if atr is None:
            av = self.atrs()
            atr = av[-1] if av else max(abs((self.last_ask or 0) - (self.last_bid or 0)), 0.01)
        scene = base.Scene(self.active_scene) if self.active_scene in base.Scene._value2member_map_ else base.Scene.TRANSITION
        if scene in (base.Scene.BALANCED_RANGE, base.Scene.LIQUIDITY_BUILD, base.Scene.COMPRESSION):
            sk, tk = .75, .55
        elif scene in (base.Scene.REVERSAL, base.Scene.BREAKOUT, base.Scene.CONTINUATION, base.Scene.CRISIS):
            sk, tk = 1., 1.8
        else:
            sk, tk = .9, 1.2
        self.stop_ref = self.entry_ref - side * sk * atr
        self.tp_ref = self.entry_ref + side * tk * atr
        self.trail_ref = None
        self.hold = 0
        self.exit_pending = False
        self._reset_pending()

    def on_position_closed(self, event):
        self.lifecycle['positions_closed'] += 1
        self.closed_trades.append({
            'scene': self.active_scene,
            'pnl': base.money(getattr(event, 'realized_pnl', None)),
            'ts_closed': int(getattr(event, 'ts_closed', 0) or 0),
            'realized_return': float(getattr(event, 'realized_return', 0.0) or 0.0),
        })
        self.active_scene = 'unknown'
        self.entry_intent = None
        self._reset_pending()
        self.entry_ref = None
        self.stop_ref = None
        self.tp_ref = None
        self.trail_ref = None
        self.entry_side = 0
        self.hold = 0
        self.exit_pending = False


base.Strat = CompatStrat
print('PORTFOLIO_FLAT_API=strategy_compat')
print('FEATURE_UPDATE_MODE=BAR_CACHED_TICK_EXECUTION')
print('REPORT_MODE=POSITION_CLOSED_EVENTS')
print('LIFECYCLE_MODE=EVENT_DRIVEN_OPEN_CLOSE_REENTRY')
print('ENTRY_SIGNAL_DELAY_TICKS=1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--symbols', nargs='+', required=True)
    ap.add_argument('--timeframes', nargs='+', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--raw-bidask-only', action='store_true')
    a = ap.parse_args()
    if not a.raw_bidask_only:
        raise SystemExit('raw-bidask-only is mandatory')

    path = Path(a.catalog)
    manifest = json.loads((path / 'catalog_manifest.json').read_text())
    days = int(manifest['days'])
    cat = ParquetDataCatalog(str(path))
    insts = {x.id.symbol.value.replace('/', ''): x for x in cat.instruments()}
    out = Path('results/ae-bt') / a.experiment_id
    out.mkdir(parents=True, exist_ok=True)

    allts, cells, scenes, counts, trans, raw, lifecycle = [], {}, {}, {}, {}, {}, {}
    for symbol in [s for s in a.symbols if s == 'XAUUSD']:
        ins = insts.get(symbol)
        if ins is None:
            raise SystemExit(f'instrument missing: {symbol}')
        ticks = cat.query_quote_ticks(identifiers=[ins.id.value])
        raw[symbol] = len(ticks)
        if not ticks:
            raise SystemExit('no raw QuoteTicks: XAUUSD')
        for tf in a.timeframes:
            mins = base.TF_MIN[tf]
            eng = base.BacktestEngine(config=base.BacktestEngineConfig(
                logging=base.LoggingConfig(log_level='ERROR'),
                risk_engine=base.RiskEngineConfig(bypass=True),
            ))
            eng.add_venue(
                venue=base.SIM,
                oms_type=base.OmsType.NETTING,
                account_type=base.AccountType.MARGIN,
                base_currency=base.USD,
                starting_balances=[base.Money(1000, base.USD)],
                default_leverage=Decimal('2000'),
            )
            eng.add_instrument(ins)
            eng.add_data(ticks)
            bt = base.BarType.from_str(f'{ins.id.value}-{mins}-MINUTE-BID-INTERNAL')
            st = CompatStrat(base.Cfg(
                instrument_id=ins.id,
                bar_type=bt,
                trade_size=Decimal('1'),
                tf_minutes=mins,
            ))
            eng.add_strategy(st)
            eng.run()
            tt = [dict(t, symbol=symbol, tf=tf) for t in st.closed_trades]
            allts += tt
            key = f'{symbol}:{tf}'
            cells[key] = {
                **base.metrics(tt, days=days),
                'raw_ticks': len(ticks),
                'signals_submitted': st.entries,
            }
            lifecycle[key] = {**dict(st.lifecycle), 'rejection_reasons': dict(st.rejection_reasons)}
            counts[key] = dict(st.scene_counts)
            trans[key] = dict(st.transitions)
            for sc in sorted({t['scene'] for t in tt}):
                scenes[f'{key}:{sc}'] = base.metrics([t for t in tt if t['scene'] == sc], days=days)
            eng.dispose()

    allts.sort(key=lambda x: (x['ts_closed'], x['symbol'], x['tf']))
    summary = {
        'verification_level': 'NAUTILUS_BT_RAW_BIDASK',
        'strategy': 'AMOS_AllWeather_XAUUSD_MetaBot_v0.2',
        'engine': 'NautilusTrader BacktestEngine',
        'nautilus_version': getattr(base.nautilus_trader, '__version__', 'unknown'),
        'data_kind': 'RAW_BIDASK QuoteTick',
        'ohlc_resample_used': False,
        'signal_bars': 'Nautilus INTERNAL BID bars built from raw QuoteTicks',
        'execution': 'MARKET orders on raw QuoteTicks with 1-Quote execution delay; native observed spread included; explicit commission/slippage model not yet added',
        'symbols': ['XAUUSD'],
        'timeframes': a.timeframes,
        'period': {'start': manifest['start'], 'days': days, 'end_exclusive': manifest['end_exclusive']},
        'portfolio_realized_close_ordered': base.metrics(allts, days=days),
        'cell_metrics': cells,
        'lifecycle_metrics': lifecycle,
        'scene_metrics': scenes,
        'scene_bar_counts': counts,
        'state_transitions': trans,
        'raw_tick_counts': raw,
        'limitations': [
            'Scheduled-news calendar is not present in the raw catalog; NEWS is not directly event-labelled in v0.2.',
            'IFVG/BPR/Breaker are reserved but not fully reconstructed in v0.2.',
            'Portfolio DD is reconstructed from realized closed PnL across independent TF cells, not synchronized mark-to-market equity.',
            'No explicit commission/probabilistic slippage model yet; raw Bid/Ask spread is native.',
        ],
    }
    base.pd.DataFrame(allts).to_csv(out / 'trades.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / 'catalog_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
