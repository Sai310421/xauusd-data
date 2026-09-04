from __future__ import annotations

import argparse
import json
import math
from collections import deque
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue('SIM')

PARAMS = {
    'spike_body_atr': 2.5,
    'spike_range_atr': 2.0,
    'confirm_bars': 2,
    'rebound_atr': 0.6,
    'guard_ext': 0.5,
    'tp_atr': 1.0,
    'trail_act': 0.40,
    'trail_dist': 0.20,
    'min_hold': 1,
    'max_hold_minutes': 720,
}
TF_MIN = {'M1': 1, 'M5': 5, 'M15': 15}
TRADE_SIZE = {
    'XAUUSD': Decimal('1'),
    'EURUSD': Decimal('1000'),
    'GBPUSD': Decimal('1000'),
    'USDJPY': Decimal('1000'),
    'AUDUSD': Decimal('1000'),
    'USDCHF': Decimal('1000'),
}


class MinimumSpikeRawConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    tf_minutes: int


class MinimumSpikeRawStrategy(Strategy):
    def __init__(self, config: MinimumSpikeRawConfig):
        super().__init__(config)
        self.bars = deque(maxlen=32)
        self.pending = None
        self.armed = None
        self.entry_ref = None
        self.stop_ref = None
        self.tp_ref = None
        self.trail_ref = None
        self.hold_bars = 0
        self.exit_pending = False
        self.entries = 0

    def on_start(self) -> None:
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    @staticmethod
    def _f(px) -> float:
        return float(px.as_double()) if hasattr(px, 'as_double') else float(px)

    def _atr14(self) -> float | None:
        if len(self.bars) < 15:
            return None
        xs = list(self.bars)
        trs = []
        for i in range(-14, 0):
            cur = xs[i]
            prev = xs[i - 1]
            trs.append(max(cur['h'] - cur['l'], abs(cur['h'] - prev['c']), abs(cur['l'] - prev['c'])))
        a = float(np.mean(trs))
        return a if math.isfinite(a) and a > 0 else None

    def on_bar(self, bar: Bar) -> None:
        b = {
            'o': self._f(bar.open), 'h': self._f(bar.high),
            'l': self._f(bar.low), 'c': self._f(bar.close),
            'ts': int(bar.ts_event),
        }
        self.bars.append(b)
        if self.entry_ref is not None:
            self.hold_bars += 1
        if self.pending is not None:
            p = self.pending
            if b['ts'] > p['ts']:
                if b['c'] - p['low'] >= PARAMS['rebound_atr'] * p['atr'] and b['c'] > b['o']:
                    self.armed = dict(p)
                    self.pending = None
                else:
                    p['remaining'] -= 1
                    if p['remaining'] <= 0:
                        self.pending = None
        a = self._atr14()
        if a is not None:
            body = b['o'] - b['c']
            rng = b['h'] - b['l']
            bearish = b['c'] < b['o'] and body >= PARAMS['spike_body_atr'] * a and rng >= PARAMS['spike_range_atr'] * a
            if bearish:
                self.pending = {'low': b['l'], 'atr': a, 'remaining': PARAMS['confirm_bars'], 'ts': b['ts']}
        max_hold_bars = max(1, int(PARAMS['max_hold_minutes'] / self.config.tf_minutes))
        if self.entry_ref is not None and self.hold_bars >= max_hold_bars and not self.exit_pending:
            self.close_all_positions(self.config.instrument_id)
            self.exit_pending = True

    def on_quote_tick(self, tick: QuoteTick) -> None:
        bid = self._f(tick.bid_price)
        ask = self._f(tick.ask_price)
        is_flat = not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
        if self.armed is not None and self.entry_ref is None and is_flat:
            instrument = self.cache.instrument(self.config.instrument_id)
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=instrument.make_qty(self.config.trade_size),
            )
            self.submit_order(order)
            p = self.armed
            self.entry_ref = ask
            self.stop_ref = p['low'] - PARAMS['guard_ext'] * p['atr']
            self.tp_ref = ask + PARAMS['tp_atr'] * p['atr']
            self.trail_ref = None
            self.hold_bars = 0
            self.exit_pending = False
            self.entries += 1
            self.armed = None
            return
        if self.entry_ref is None or self.exit_pending:
            return
        atr = (self.tp_ref - self.entry_ref) / PARAMS['tp_atr']
        if bid >= self.entry_ref + PARAMS['trail_act'] * atr:
            nt = bid - PARAMS['trail_dist'] * atr
            self.trail_ref = nt if self.trail_ref is None else max(self.trail_ref, nt)
        active_stop = max(self.stop_ref, self.trail_ref) if self.trail_ref is not None else self.stop_ref
        if self.hold_bars >= PARAMS['min_hold'] and (bid <= active_stop or bid >= self.tp_ref):
            self.close_all_positions(self.config.instrument_id)
            self.exit_pending = True

    def on_position_closed(self, event) -> None:
        self.entry_ref = None
        self.stop_ref = None
        self.tp_ref = None
        self.trail_ref = None
        self.hold_bars = 0
        self.exit_pending = False

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)


def parse_money(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (float, int, np.number)):
        return float(v)
    s = str(v).replace(',', '').strip()
    try:
        return float(s.split()[0])
    except Exception:
        return 0.0


def extract_trades(report: pd.DataFrame, symbol: str, tf: str):
    if report is None or report.empty:
        return []
    pnl_col = next((c for c in report.columns if str(c).lower() in ('realized_pnl','pnl','realizedpnl')), None)
    if pnl_col is None:
        pnl_col = next((c for c in report.columns if 'pnl' in str(c).lower()), None)
    ts_col = next((c for c in report.columns if 'closed' in str(c).lower() and ('ts' in str(c).lower() or 'time' in str(c).lower())), None)
    out = []
    for i, row in report.iterrows():
        pnl = parse_money(row[pnl_col]) if pnl_col is not None else 0.0
        ts = row[ts_col] if ts_col is not None else i
        try:
            ts = pd.Timestamp(ts).value
        except Exception:
            try:
                ts = int(ts)
            except Exception:
                ts = len(out)
        out.append({'symbol': symbol, 'tf': tf, 'pnl': pnl, 'ts_closed': int(ts)})
    return out


def metrics(trades, initial=1000.0, days=30):
    if not trades:
        return {'N':0,'WR_pct':0.0,'PF':0.0,'NetProfit':0.0,'MaxDD_pct':0.0,'RF':0.0,'Monthly21_pct':0.0}
    a = np.array([t['pnl'] for t in trades], float)
    wins = a[a > 0]; losses = a[a < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (float('inf') if len(wins) else 0.0)
    eq = initial; peak = initial; mdd_d = 0.0
    for x in a:
        eq += x; peak = max(peak, eq); mdd_d = max(mdd_d, peak - eq)
    net = float(a.sum())
    mdd_pct = mdd_d / peak * 100 if peak > 0 else 0.0
    monthly = ((max(eq, 1e-9) / initial) ** (21 / days) - 1) * 100
    return {
        'N': int(len(a)), 'WR_pct': float((a > 0).mean() * 100), 'PF': pf,
        'NetProfit': net, 'MaxDD_pct': float(mdd_pct),
        'RF': float(net / mdd_d) if mdd_d > 0 else None,
        'Monthly21_pct': float(monthly),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--symbols', nargs='+', required=True)
    ap.add_argument('--timeframes', nargs='+', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--raw-bidask-only', action='store_true')
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit('raw-bidask-only is mandatory')
    catalog_path = Path(args.catalog)
    manifest = json.loads((catalog_path / 'catalog_manifest.json').read_text(encoding='utf-8'))
    days = int(manifest['days'])
    catalog = ParquetDataCatalog(str(catalog_path))
    inst_by_plain = {x.id.symbol.value.replace('/',''): x for x in catalog.instruments()}
    outdir = Path('results/ae-bt') / args.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    all_trades = []
    cell_metrics = {}
    raw_counts = {}
    for symbol in args.symbols:
        instrument = inst_by_plain.get(symbol)
        if instrument is None:
            raise SystemExit(f'instrument missing from Nautilus catalog: {symbol}')
        ticks = catalog.query_quote_ticks(identifiers=[instrument.id.value])
        raw_counts[symbol] = len(ticks)
        if not ticks:
            raise SystemExit(f'no raw QuoteTicks: {symbol}')
        for tf in args.timeframes:
            minutes = TF_MIN[tf]
            config = BacktestEngineConfig(
                logging=LoggingConfig(log_level='ERROR'),
                risk_engine=RiskEngineConfig(bypass=True),
            )
            engine = BacktestEngine(config=config)
            engine.add_venue(
                venue=SIM,
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                base_currency=USD,
                starting_balances=[Money(1000, USD)],
                default_leverage=Decimal('2000'),
            )
            engine.add_instrument(instrument)
            engine.add_data(ticks)
            bar_type = BarType.from_str(f'{instrument.id.value}-{minutes}-MINUTE-BID-INTERNAL')
            strat = MinimumSpikeRawStrategy(MinimumSpikeRawConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                trade_size=TRADE_SIZE[symbol],
                tf_minutes=minutes,
            ))
            engine.add_strategy(strat)
            engine.run()
            report = engine.trader.generate_positions_report()
            trades = extract_trades(report, symbol, tf)
            all_trades.extend(trades)
            cell_metrics[f'{symbol}:{tf}'] = {**metrics(trades, days=days), 'raw_ticks': len(ticks), 'signals_submitted': strat.entries}
            engine.dispose()
    all_trades.sort(key=lambda x: (x['ts_closed'], x['symbol'], x['tf']))
    portfolio = metrics(all_trades, days=days)
    summary = {
        'verification_level': 'NAUTILUS_BT_RAW_BIDASK',
        'engine': 'NautilusTrader BacktestEngine',
        'nautilus_version': getattr(nautilus_trader, '__version__', 'unknown'),
        'data_kind': 'RAW_BIDASK QuoteTick',
        'ohlc_resample_used': False,
        'signal_bars': 'Nautilus INTERNAL BID bars built from raw QuoteTicks',
        'execution': 'Nautilus simulated venue MARKET orders on raw QuoteTicks; observed spread included; commission/slippage model not yet added',
        'symbols': args.symbols,
        'timeframes': args.timeframes,
        'period': {'start': manifest['start'], 'days': days, 'end_exclusive': manifest['end_exclusive']},
        'portfolio_realized_close_ordered': portfolio,
        'cell_metrics': cell_metrics,
        'raw_tick_counts': raw_counts,
        'limitations': [
            'Portfolio DD is reconstructed from realized closed-position PnL across independent symbol/TF engine cells, not synchronized mark-to-market portfolio equity.',
            'No explicit commission or probabilistic slippage model in this first RAW_BIDASK gate; raw Bid/Ask spread is native.',
            'Each symbol/TF is run as an independent Nautilus BacktestEngine cell to avoid cross-TF position-netting contamination.',
        ],
    }
    pd.DataFrame(all_trades).to_csv(outdir / 'trades.csv', index=False)
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir / 'catalog_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
