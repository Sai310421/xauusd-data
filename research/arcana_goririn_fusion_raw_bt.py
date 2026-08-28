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
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, TraderId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue('SIM')
MODES = ('PASS_THROUGH', 'FILTER', 'TIMING', 'TIMING_APEX_EXIT')


class FusionConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    h1_bar_type: BarType
    m1_bar_type: BarType
    mode: str
    trade_size: Decimal
    intent_expiry_minutes: int = 60
    stop_atr: float = 1.0
    tp_r: float = 1.0
    apex_lock_trigger_r: float = 0.85
    apex_lock_r: float = 0.15
    trail_trigger_r: float = 1.05
    trail_atr_mult: float = 1.25


class FusionStrategy(Strategy):
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        self.h1 = deque(maxlen=80)
        self.m1 = deque(maxlen=240)
        self.intent_side = 0
        self.intent_ts = None
        self.intent_count = 0
        self.goririn_candidates = 0
        self.goririn_rejects = 0
        self.entries = 0
        self.entry_ref = None
        self.side = 0
        self.stop_ref = None
        self.tp_ref = None
        self.initial_risk = None
        self.mfe = 0.0
        self.exit_pending = False
        self.cached_atr = None
        self.cached_goririn = {1: (False, False, 0.0), -1: (False, False, 0.0)}
        self.entry_allowed_cache = False
        self.cache_updates = 0
        self.quote_signal_recomputes = 0

    @staticmethod
    def _f(px) -> float:
        return float(px.as_double()) if hasattr(px, 'as_double') else float(px)

    @staticmethod
    def _ema(values, period):
        if len(values) < period:
            return None
        a = 2.0 / (period + 1.0)
        e = float(values[-period])
        for v in values[-period + 1:]:
            e = a * float(v) + (1.0 - a) * e
        return e

    @staticmethod
    def _rsi(values, period=14):
        if len(values) < period + 1:
            return None
        d = np.diff(np.asarray(values[-period - 1:], dtype=float))
        up = np.maximum(d, 0.0).mean()
        dn = np.maximum(-d, 0.0).mean()
        if dn == 0:
            return 100.0
        rs = up / dn
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _cci(bars, period=20):
        if len(bars) < period:
            return None
        tps = np.asarray([(b['h'] + b['l'] + b['c']) / 3.0 for b in list(bars)[-period:]], dtype=float)
        ma = tps.mean()
        md = np.mean(np.abs(tps - ma))
        return 0.0 if md <= 0 else float((tps[-1] - ma) / (0.015 * md))

    @staticmethod
    def _rank_corr(values, period):
        if len(values) < period:
            return None
        x = np.asarray(values[-period:], dtype=float)
        pr = pd.Series(x).rank(method='average').to_numpy(dtype=float)
        tr = np.arange(1, period + 1, dtype=float)
        d = tr - pr
        den = period * (period * period - 1)
        return None if den == 0 else float((1.0 - 6.0 * np.sum(d * d) / den) * 100.0)

    def _atr(self, bars, period=14):
        if len(bars) < period + 1:
            return None
        xs = list(bars)
        trs = []
        for i in range(-period, 0):
            cur, prev = xs[i], xs[i - 1]
            trs.append(max(cur['h'] - cur['l'], abs(cur['h'] - prev['c']), abs(cur['l'] - prev['c'])))
        v = float(np.mean(trs))
        return v if math.isfinite(v) and v > 0 else None

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.h1_bar_type)
        self.subscribe_bars(self.config.m1_bar_type)

    def _arcana_candidate_intent(self):
        if len(self.h1) < 55:
            return 0
        closes = [b['c'] for b in self.h1]
        e20, e50, prev20 = self._ema(closes, 20), self._ema(closes, 50), self._ema(closes[:-1], 20)
        if None in (e20, e50, prev20):
            return 0
        b, p = self.h1[-1], self.h1[-2]
        bull = e20 > e50 and e20 > prev20 and b['c'] > e20 and p['l'] <= e20
        bear = e20 < e50 and e20 < prev20 and b['c'] < e20 and p['h'] >= e20
        return 1 if bull else (-1 if bear else 0)

    def _compute_goririn(self, side):
        if len(self.m1) < 55:
            return False, False, 0.0
        closes = [b['c'] for b in self.m1]
        rcis = [self._rank_corr(closes, p) for p in (9, 26, 52)]
        if any(v is None for v in rcis):
            return False, False, 0.0
        rsi, cci = self._rsi(closes, 14), self._cci(self.m1, 20)
        if rsi is None or cci is None:
            return False, False, 0.0
        b, p = self.m1[-1], self.m1[-2]
        if side > 0:
            extreme = sum(v <= -90.0 for v in rcis)
            reversal = b['c'] > b['o'] and b['c'] > p['c']
            momentum = rsi < 45.0 and cci < 0.0
        else:
            extreme = sum(v >= 90.0 for v in rcis)
            reversal = b['c'] < b['o'] and b['c'] < p['c']
            momentum = rsi > 55.0 and cci > 0.0
        score = float(extreme) + float(reversal) + float(momentum)
        candidate = extreme >= 2 and reversal
        return candidate, bool(candidate and momentum), score

    def _refresh_m1_cache(self):
        self.cached_atr = self._atr(self.m1, 14)
        self.cached_goririn[1] = self._compute_goririn(1)
        self.cached_goririn[-1] = self._compute_goririn(-1)
        self.cache_updates += 1

    def on_bar(self, bar: Bar):
        b = {'o': self._f(bar.open), 'h': self._f(bar.high), 'l': self._f(bar.low), 'c': self._f(bar.close), 'ts': int(bar.ts_event)}
        bt = str(bar.bar_type)
        if '60-MINUTE' in bt:
            self.h1.append(b)
            side = self._arcana_candidate_intent()
            if side == 0:
                return
            self.intent_count += 1
            self.intent_side, self.intent_ts = side, b['ts']
            self.entry_allowed_cache = False
            if self.config.mode == 'PASS_THROUGH':
                self.entry_allowed_cache = True
            elif self.config.mode == 'FILTER':
                cand, _, _ = self.cached_goririn.get(side, (False, False, 0.0))
                self.entry_allowed_cache = cand
                if cand:
                    self.goririn_candidates += 1
                else:
                    self.goririn_rejects += 1
                    self.intent_side, self.intent_ts = 0, None
            return
        if '1-MINUTE' not in bt:
            return
        self.m1.append(b)
        self._refresh_m1_cache()
        if self.intent_side == 0 or self.intent_ts is None:
            return
        age_min = (b['ts'] - self.intent_ts) / 60_000_000_000
        if age_min > self.config.intent_expiry_minutes:
            self.intent_side, self.intent_ts = 0, None
            self.entry_allowed_cache = False
            return
        if self.config.mode in ('TIMING', 'TIMING_APEX_EXIT'):
            cand, _, _ = self.cached_goririn[self.intent_side]
            if cand and not self.entry_allowed_cache:
                self.goririn_candidates += 1
                self.entry_allowed_cache = True
            elif not cand:
                self.goririn_rejects += 1

    def on_quote_tick(self, tick: QuoteTick):
        bid, ask = self._f(tick.bid_price), self._f(tick.ask_price)
        if self.entry_ref is None and self.portfolio.is_net_flat(self.config.instrument_id) and self.entry_allowed_cache:
            atr = self.cached_atr
            if atr is None or self.intent_side == 0:
                return
            instrument = self.cache.instrument(self.config.instrument_id)
            side = self.intent_side
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY if side > 0 else OrderSide.SELL,
                quantity=instrument.make_qty(self.config.trade_size),
            )
            self.submit_order(order)
            px = ask if side > 0 else bid
            risk = self.config.stop_atr * atr
            self.entry_ref, self.side, self.initial_risk = px, side, risk
            self.stop_ref, self.tp_ref = px - side * risk, px + side * self.config.tp_r * risk
            self.mfe, self.exit_pending = 0.0, False
            self.entries += 1
            self.intent_side, self.intent_ts, self.entry_allowed_cache = 0, None, False
            return
        if self.entry_ref is None or self.exit_pending:
            return
        exit_px = bid if self.side > 0 else ask
        self.mfe = max(self.mfe, self.side * (exit_px - self.entry_ref))
        active_stop = self.stop_ref
        if self.config.mode == 'TIMING_APEX_EXIT' and self.initial_risk and self.initial_risk > 0:
            if self.mfe >= self.config.apex_lock_trigger_r * self.initial_risk:
                lock = self.entry_ref + self.side * self.config.apex_lock_r * self.initial_risk
                active_stop = max(active_stop, lock) if self.side > 0 else min(active_stop, lock)
            if self.cached_atr and self.mfe >= self.config.trail_trigger_r * self.initial_risk:
                trail = exit_px - self.side * self.config.trail_atr_mult * self.cached_atr
                active_stop = max(active_stop, trail) if self.side > 0 else min(active_stop, trail)
                self.stop_ref = active_stop
        hit_stop = exit_px <= active_stop if self.side > 0 else exit_px >= active_stop
        hit_tp = exit_px >= self.tp_ref if self.side > 0 else exit_px <= self.tp_ref
        if hit_stop or hit_tp:
            self.close_all_positions(self.config.instrument_id)
            self.exit_pending = True

    def on_position_closed(self, event):
        self.entry_ref = None
        self.side = 0
        self.stop_ref = self.tp_ref = self.initial_risk = None
        self.mfe = 0.0
        self.exit_pending = False

    def on_stop(self):
        self.close_all_positions(self.config.instrument_id)


def parse_money(v):
    if v is None:
        return 0.0
    if isinstance(v, (float, int, np.number)):
        return float(v)
    try:
        return float(str(v).replace(',', '').strip().split()[0])
    except Exception:
        return 0.0


def extract_trades(report, mode):
    if report is None or report.empty:
        return []
    pnl_col = next((c for c in report.columns if 'pnl' in str(c).lower()), None)
    ts_col = next((c for c in report.columns if 'closed' in str(c).lower() and ('ts' in str(c).lower() or 'time' in str(c).lower())), None)
    out = []
    for i, row in report.iterrows():
        pnl = parse_money(row[pnl_col]) if pnl_col is not None else 0.0
        ts = row[ts_col] if ts_col is not None else i
        try:
            ts = int(pd.Timestamp(ts).value)
        except Exception:
            try:
                ts = int(ts)
            except Exception:
                ts = len(out)
        out.append({'mode': mode, 'pnl': pnl, 'ts_closed': ts})
    return out


def metrics(trades, initial=1000.0, days=30):
    if not trades:
        return {'N':0,'WR_pct':0.0,'PF':0.0,'NetProfit':0.0,'MaxDD_pct':0.0,'RF':0.0,'Monthly21_pct':0.0}
    a = np.asarray([t['pnl'] for t in trades], dtype=float)
    wins, losses = a[a > 0], a[a < 0]
    pf = float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    eq = peak = initial
    mdd = 0.0
    for x in a:
        eq += x
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    net = float(a.sum())
    return {
        'N': int(len(a)), 'WR_pct': float((a>0).mean()*100), 'PF': pf,
        'NetProfit': net, 'MaxDD_pct': float(mdd/peak*100 if peak>0 else 0.0),
        'RF': float(net/mdd) if mdd>0 else None,
        'Monthly21_pct': float(((max(eq,1e-9)/initial)**(21/days)-1)*100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--symbol', default='USDJPY')
    ap.add_argument('--raw-bidask-only', action='store_true')
    ap.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit('raw-bidask-only is mandatory')
    catalog_path = Path(args.catalog).resolve()
    manifest = json.loads((catalog_path/'catalog_manifest.json').read_text(encoding='utf-8'))
    days = int(manifest['days'])
    catalog = ParquetDataCatalog(str(catalog_path))
    inst_by_plain = {x.id.symbol.value.replace('/',''): x for x in catalog.instruments()}
    instrument = inst_by_plain.get(args.symbol)
    if instrument is None:
        raise SystemExit(f'instrument missing from Nautilus catalog: {args.symbol}')
    ticks = catalog.quote_ticks(instrument_ids=[instrument.id.value])
    if not ticks:
        raise SystemExit(f'no raw QuoteTicks: {args.symbol}')
    outdir = Path('results/ae-bt')/args.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    results, all_trades = {}, []
    for mode in args.modes:
        engine = BacktestEngine(config=BacktestEngineConfig(
            trader_id=TraderId('FUSION-001'),
            logging=LoggingConfig(log_level='ERROR'),
            risk_engine=RiskEngineConfig(bypass=True),
        ))
        engine.add_venue(
            venue=SIM, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
            base_currency=USD, starting_balances=[Money(1000, USD)], default_leverage=Decimal('2000'),
        )
        engine.add_instrument(instrument)
        engine.add_data(ticks)
        strat = FusionStrategy(FusionConfig(
            instrument_id=instrument.id,
            h1_bar_type=BarType.from_str(f'{instrument.id.value}-60-MINUTE-BID-INTERNAL'),
            m1_bar_type=BarType.from_str(f'{instrument.id.value}-1-MINUTE-BID-INTERNAL'),
            mode=mode, trade_size=Decimal('1000'),
        ))
        engine.add_strategy(strat)
        engine.run()
        trades = extract_trades(engine.generate_positions_report(), mode)
        all_trades.extend(trades)
        results[mode] = {
            **metrics(trades, days=days),
            'arcana_candidate_intents': strat.intent_count,
            'goririn_candidates': strat.goririn_candidates,
            'goririn_reject_observations': strat.goririn_rejects,
            'entries_submitted': strat.entries,
            'm1_signal_cache_updates': strat.cache_updates,
            'quote_signal_recomputes': strat.quote_signal_recomputes,
        }
        engine.dispose()
    summary = {
        'verification_level': 'NAUTILUS_BT_RAW_BIDASK_FUSION_CANDIDATE',
        'engine': 'NautilusTrader BacktestEngine',
        'nautilus_version': getattr(nautilus_trader,'__version__','unknown'),
        'symbol': args.symbol,
        'data_kind': 'RAW_BIDASK QuoteTick',
        'ohlc_resample_used': False,
        'signal_compute_policy': 'M1/H1 bar-close cache; QuoteTick execution reads cached state only',
        'execution': 'Nautilus MARKET orders on raw QuoteTicks; native spread included',
        'floating_dd_status': 'REQUIRED_BY_STANDARD_BUT_NOT_YET_IMPLEMENTED_IN_THIS_RUNNER',
        'reality_profile_status': 'native spread only; commission/slippage/latency remain production-gate requirements',
        'period': {'start':manifest['start'],'days':days,'end_exclusive':manifest['end_exclusive']},
        'modes': results,
        'source_boundary': {
            'ARCANA_exact_original_entry': 'UNKNOWN',
            'ARCANA_H1_intent_in_this_runner': 'CLEAN_ROOM_CANDIDATE EMA20/50 pullback-reclaim',
            'GORIRIN_architecture': 'FUNCTIONAL_EXTRACTION_BASED',
            'GORIRIN_RCI_periods': 'CLEAN_ROOM_PROBE 9/26/52; original periods unverified',
            'GORIRIN_RCI_extreme': '±90 and >=2 of 3 based on supplied extraction pack',
        },
        'limitations': [
            'Fusion-candidate architecture test, not ARCANA source parity.',
            'Fixed trade size for signal-quality isolation; ARCANA AutoLot excluded.',
            'No GORIRIN martingale/nanpin in phase 1.',
            'Production-grade floating MTM DD and explicit commission/slippage/latency are fail-closed future gates.',
        ],
    }
    pd.DataFrame(all_trades).to_csv(outdir/'fusion_trades.csv', index=False)
    (outdir/'fusion_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (outdir/'catalog_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
