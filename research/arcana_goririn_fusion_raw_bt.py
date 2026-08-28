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
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import LogLevel
from nautilus_trader.config import BacktestEngineConfig, LoggerConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue('SIM')

# IMPORTANT:
# ARCANA exact original Entry/Exit is not recovered.
# The H1 intent below is a clean-room candidate used only to test the fusion architecture.
# GORIRIN timing uses the supplied functional-extraction architecture: 3 RCI families,
# extreme threshold, 2-of-3 agreement, plus RSI/CCI/reversal-style confirmation.

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

    @staticmethod
    def _f(px) -> float:
        return float(px.as_double()) if hasattr(px, 'as_double') else float(px)

    @staticmethod
    def _ema(values, period: int) -> float | None:
        if len(values) < period:
            return None
        a = 2.0 / (period + 1.0)
        e = float(values[-period])
        for v in values[-period + 1:]:
            e = a * float(v) + (1.0 - a) * e
        return e

    @staticmethod
    def _rsi(values, period: int = 14) -> float | None:
        if len(values) < period + 1:
            return None
        xs = np.asarray(values[-period - 1:], dtype=float)
        d = np.diff(xs)
        up = np.maximum(d, 0.0).mean()
        dn = np.maximum(-d, 0.0).mean()
        if dn == 0:
            return 100.0
        rs = up / dn
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _cci(bars, period: int = 20) -> float | None:
        if len(bars) < period:
            return None
        tps = np.array([(b['h'] + b['l'] + b['c']) / 3.0 for b in list(bars)[-period:]], dtype=float)
        ma = tps.mean()
        md = np.mean(np.abs(tps - ma))
        if md <= 0:
            return 0.0
        return float((tps[-1] - ma) / (0.015 * md))

    @staticmethod
    def _rank_corr(values, period: int) -> float | None:
        # Clean-room RCI proxy: Spearman rank correlation of time rank and price rank.
        if len(values) < period:
            return None
        x = np.asarray(values[-period:], dtype=float)
        time_ranks = np.arange(1, period + 1, dtype=float)
        price_ranks = pd.Series(x).rank(method='average').to_numpy(dtype=float)
        d = time_ranks - price_ranks
        denom = period * (period * period - 1)
        if denom == 0:
            return None
        return float((1.0 - 6.0 * np.sum(d * d) / denom) * 100.0)

    def _atr(self, bars, period: int = 14) -> float | None:
        if len(bars) < period + 1:
            return None
        xs = list(bars)
        trs = []
        for i in range(-period, 0):
            cur, prev = xs[i], xs[i - 1]
            trs.append(max(cur['h'] - cur['l'], abs(cur['h'] - prev['c']), abs(cur['l'] - prev['c'])))
        v = float(np.mean(trs))
        return v if math.isfinite(v) and v > 0 else None

    def on_start(self) -> None:
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.h1_bar_type)
        self.subscribe_bars(self.config.m1_bar_type)

    def _arcana_candidate_intent(self) -> int:
        # Clean-room H1 candidate, NOT recovered ARCANA source.
        if len(self.h1) < 55:
            return 0
        closes = [b['c'] for b in self.h1]
        e20 = self._ema(closes, 20)
        e50 = self._ema(closes, 50)
        prev20 = self._ema(closes[:-1], 20)
        if None in (e20, e50, prev20):
            return 0
        b = self.h1[-1]
        p = self.h1[-2]
        bull = e20 > e50 and e20 > prev20 and b['c'] > e20 and p['l'] <= e20
        bear = e20 < e50 and e20 < prev20 and b['c'] < e20 and p['h'] >= e20
        return 1 if bull else (-1 if bear else 0)

    def _goririn_candidate(self, side: int) -> tuple[bool, bool, float]:
        if len(self.m1) < 55 or side == 0:
            return False, False, 0.0
        closes = [b['c'] for b in self.m1]
        # Periods are clean-room probes; original periods remain unverified.
        rcis = [self._rank_corr(closes, p) for p in (9, 26, 52)]
        if any(v is None for v in rcis):
            return False, False, 0.0
        rsi = self._rsi(closes, 14)
        cci = self._cci(self.m1, 20)
        if rsi is None or cci is None:
            return False, False, 0.0
        b = self.m1[-1]
        p = self.m1[-2]

        if side > 0:
            extreme = sum(v <= -90.0 for v in rcis)
            reversal = b['c'] > b['o'] and b['c'] > p['c']
            momentum = rsi < 45.0 and cci < 0.0
        else:
            extreme = sum(v >= 90.0 for v in rcis)
            reversal = b['c'] < b['o'] and b['c'] < p['c']
            momentum = rsi > 55.0 and cci > 0.0

        score = float(extreme) + (1.0 if reversal else 0.0) + (1.0 if momentum else 0.0)
        candidate = extreme >= 2 and reversal
        high_conf = candidate and momentum and extreme >= 2
        return candidate, high_conf, score

    def on_bar(self, bar: Bar) -> None:
        b = {'o': self._f(bar.open), 'h': self._f(bar.high), 'l': self._f(bar.low), 'c': self._f(bar.close), 'ts': int(bar.ts_event)}
        bt = str(bar.bar_type)

        if '60-MINUTE' in bt:
            self.h1.append(b)
            side = self._arcana_candidate_intent()
            if side != 0:
                self.intent_side = side
                self.intent_ts = b['ts']
                self.intent_count += 1
            return

        if '1-MINUTE' not in bt:
            return
        self.m1.append(b)

        if self.intent_side == 0 or self.intent_ts is None:
            return
        age_min = (b['ts'] - self.intent_ts) / 60_000_000_000
        if age_min > self.config.intent_expiry_minutes:
            self.intent_side = 0
            self.intent_ts = None
            return

        cand, high, score = self._goririn_candidate(self.intent_side)
        if cand:
            self.goririn_candidates += 1
        elif self.config.mode in ('FILTER', 'TIMING', 'TIMING_APEX_EXIT'):
            self.goririn_rejects += 1

        if self.config.mode == 'PASS_THROUGH':
            # armed immediately after H1 intent; quote tick performs execution.
            return
        if cand:
            # keep intent armed; first subsequent quote executes.
            return

    def _fusion_allows(self) -> bool:
        if self.intent_side == 0 or self.intent_ts is None:
            return False
        if self.config.mode == 'PASS_THROUGH':
            return True
        cand, _, _ = self._goririn_candidate(self.intent_side)
        return cand

    def on_quote_tick(self, tick: QuoteTick) -> None:
        bid = self._f(tick.bid_price)
        ask = self._f(tick.ask_price)

        if self.entry_ref is None and self.portfolio.is_net_flat(self.config.instrument_id) and self._fusion_allows():
            atr = self._atr(self.m1, 14)
            if atr is None:
                return
            instrument = self.cache.instrument(self.config.instrument_id)
            side = self.intent_side
            order_side = OrderSide.BUY if side > 0 else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=order_side,
                quantity=instrument.make_qty(self.config.trade_size),
            )
            self.submit_order(order)
            px = ask if side > 0 else bid
            risk = self.config.stop_atr * atr
            self.entry_ref = px
            self.side = side
            self.initial_risk = risk
            self.stop_ref = px - side * risk
            self.tp_ref = px + side * self.config.tp_r * risk
            self.mfe = 0.0
            self.exit_pending = False
            self.entries += 1
            self.intent_side = 0
            self.intent_ts = None
            return

        if self.entry_ref is None or self.exit_pending:
            return

        exit_px = bid if self.side > 0 else ask
        favorable = self.side * (exit_px - self.entry_ref)
        self.mfe = max(self.mfe, favorable)

        active_stop = self.stop_ref
        if self.config.mode == 'TIMING_APEX_EXIT' and self.initial_risk and self.initial_risk > 0:
            if self.mfe >= self.config.apex_lock_trigger_r * self.initial_risk:
                lock = self.entry_ref + self.side * self.config.apex_lock_r * self.initial_risk
                active_stop = max(active_stop, lock) if self.side > 0 else min(active_stop, lock)
            atr = self._atr(self.m1, 14)
            if atr and self.mfe >= self.config.trail_trigger_r * self.initial_risk:
                trail = exit_px - self.side * self.config.trail_atr_mult * atr
                active_stop = max(active_stop, trail) if self.side > 0 else min(active_stop, trail)
                self.stop_ref = active_stop

        hit_stop = exit_px <= active_stop if self.side > 0 else exit_px >= active_stop
        hit_tp = exit_px >= self.tp_ref if self.side > 0 else exit_px <= self.tp_ref
        if hit_stop or hit_tp:
            self.close_all_positions(self.config.instrument_id)
            self.exit_pending = True

    def on_position_closed(self, event) -> None:
        self.entry_ref = None
        self.side = 0
        self.stop_ref = None
        self.tp_ref = None
        self.initial_risk = None
        self.mfe = 0.0
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


def extract_trades(report: pd.DataFrame, mode: str):
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
    wins = a[a > 0]; losses = a[a < 0]
    pf = float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    eq=initial; peak=initial; mdd=0.0
    for x in a:
        eq += x
        peak=max(peak,eq)
        mdd=max(mdd,peak-eq)
    net=float(a.sum())
    mdd_pct=mdd/peak*100 if peak>0 else 0.0
    monthly=((max(eq,1e-9)/initial)**(21/days)-1)*100
    return {
        'N':int(len(a)), 'WR_pct':float((a>0).mean()*100), 'PF':pf,
        'NetProfit':net, 'MaxDD_pct':float(mdd_pct),
        'RF':float(net/mdd) if mdd>0 else None, 'Monthly21_pct':float(monthly),
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--symbol', default='USDJPY')
    ap.add_argument('--raw-bidask-only', action='store_true')
    ap.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    args=ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit('raw-bidask-only is mandatory')

    catalog_path=Path(args.catalog)
    manifest=json.loads((catalog_path/'catalog_manifest.json').read_text(encoding='utf-8'))
    days=int(manifest['days'])
    catalog=ParquetDataCatalog(str(catalog_path))
    inst_by_plain={x.id.symbol.value.replace('/',''): x for x in catalog.instruments()}
    instrument=inst_by_plain.get(args.symbol)
    if instrument is None:
        raise SystemExit(f'instrument missing from Nautilus catalog: {args.symbol}')
    ticks=catalog.query_quote_ticks(identifiers=[instrument.id.value])
    if not ticks:
        raise SystemExit(f'no raw QuoteTicks: {args.symbol}')

    outdir=Path('results/ae-bt')/args.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    results={}
    all_trades=[]

    for mode in args.modes:
        cfg=BacktestEngineConfig(
            trader_id=f'ARCANA-GORIRIN-{mode}',
            logging=LoggerConfig(stdout_level=LogLevel.ERROR),
            risk_engine=RiskEngineConfig(bypass=True),
        )
        engine=BacktestEngine(config=cfg)
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
        h1=BarType.from_str(f'{instrument.id.value}-60-MINUTE-BID-INTERNAL')
        m1=BarType.from_str(f'{instrument.id.value}-1-MINUTE-BID-INTERNAL')
        strat=FusionStrategy(FusionConfig(
            instrument_id=instrument.id,
            h1_bar_type=h1,
            m1_bar_type=m1,
            mode=mode,
            trade_size=Decimal('1000'),
        ))
        engine.add_strategy(strat)
        engine.run()
        trades=extract_trades(engine.generate_positions_report(), mode)
        all_trades.extend(trades)
        results[mode]={
            **metrics(trades, days=days),
            'arcana_candidate_intents':strat.intent_count,
            'goririn_candidates':strat.goririn_candidates,
            'goririn_reject_observations':strat.goririn_rejects,
            'entries_submitted':strat.entries,
        }
        engine.dispose()

    summary={
        'verification_level':'NAUTILUS_BT_RAW_BIDASK_FUSION_CANDIDATE',
        'engine':'NautilusTrader BacktestEngine',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'symbol':args.symbol,
        'data_kind':'RAW_BIDASK QuoteTick',
        'ohlc_resample_used':False,
        'signal_bars':'Nautilus INTERNAL BID M1/H1 bars built from raw QuoteTicks',
        'execution':'Nautilus MARKET orders on raw QuoteTicks; native spread included; no explicit commission/slippage model in this first fusion gate',
        'period':{'start':manifest['start'],'days':days,'end_exclusive':manifest['end_exclusive']},
        'modes':results,
        'source_boundary':{
            'ARCANA_exact_original_entry':'UNKNOWN',
            'ARCANA_H1_intent_in_this_runner':'CLEAN_ROOM_CANDIDATE EMA20/50 pullback-reclaim',
            'GORIRIN_architecture':'FUNCTIONAL_EXTRACTION_BASED',
            'GORIRIN_RCI_periods':'CLEAN_ROOM_PROBE 9/26/52; original periods unverified',
            'GORIRIN_RCI_extreme':'±90 and >=2 of 3 based on supplied extraction pack',
        },
        'limitations':[
            'This is a fusion-candidate architecture test, not ARCANA source parity.',
            'Trade size is fixed for the first signal-quality comparison; ARCANA AutoLot is intentionally excluded to prevent risk from masking signal quality.',
            'No GORIRIN martingale/nanpin in phase 1.',
            'No explicit commission or stochastic slippage model yet; raw Bid/Ask spread is native.',
        ],
    }
    pd.DataFrame(all_trades).to_csv(outdir/'fusion_trades.csv', index=False)
    (outdir/'fusion_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (outdir/'catalog_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
