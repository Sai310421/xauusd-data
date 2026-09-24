"""Standalone, non-ICT Dow/Elliott/video Raw Bid/Ask Nautilus screen.

Signals use completed internal BID bars only. A market order is sent on a later
QuoteTick; quote-side exits and Nautilus position reports supply the KPIs.
This is a research port of AMOS_Dow_Elliott_Video_Standalone_v1.mq5, not an
MQL5 bytecode backtest. Default quantities and cost assumptions are disclosed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, deque
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
from nautilus_trader.model.enums import AccountType, BookType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from research.minimumspike_raw6x3_bt import extract_trades, metrics
from research.minimumspike_raw6x3_bt_compat import CATALOG_QUOTE_API


class Config(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    m1: BarType
    m15: BarType
    size: Decimal = Decimal('1')
    elliott_gate: bool = False
    max_spread: float = 0.40


def swings(bars: list[dict], k: int = 2) -> list[tuple[int, int, float]]:
    """(index, kind, price), first observable after k later CLOSED bars."""
    out = []
    for i in range(k, len(bars) - k):
        b = bars[i]
        hi = all(b['h'] > bars[i-j]['h'] and b['h'] > bars[i+j]['h'] for j in range(1, k+1))
        lo = all(b['l'] < bars[i-j]['l'] and b['l'] < bars[i+j]['l'] for j in range(1, k+1))
        if hi ^ lo:
            out.append((i, 1 if hi else -1, b['h'] if hi else b['l']))
    return out


def dow(ss: list[tuple[int, int, float]], eps: float = .02) -> tuple[int, dict]:
    hs = [(i, v) for i, kind, v in ss if kind == 1]
    ls = [(i, v) for i, kind, v in ss if kind == -1]
    data = {'highs': hs, 'lows': ls}
    if len(hs) < 2 or len(ls) < 2:
        return 0, data
    if hs[-1][1] > hs[-2][1] + eps and ls[-1][1] > ls[-2][1] + eps:
        return 1, data
    if hs[-1][1] < hs[-2][1] - eps and ls[-1][1] < ls[-2][1] - eps:
        return -1, data
    return 0, data


def elliott(ss: list[tuple[int, int, float]], side: int) -> tuple[bool, float, tuple]:
    chosen = []
    for _, kind, price in reversed(ss):
        if not chosen or chosen[-1][0] != kind:
            chosen.append((kind, side * price))
        if len(chosen) == 6:
            break
    chosen.reverse()
    if len(chosen) == 6 and chosen[0][0] != -side:
        chosen = chosen[1:]  # newest five pivots can identify P0..P4 before P5 exists
    if len(chosen) < 5 or chosen[0][0] != -side:
        return False, 0.0, ()
    p = [x[1] for x in chosen]
    w1, w3 = p[1]-p[0], p[3]-p[2]
    if min(w1, w3) <= 0 or p[2] <= p[0] or p[3] <= p[1] or p[4] <= p[1]:
        return False, 0.0, ()
    if len(p) == 6 and (p[5] <= p[3] or w3 < min(w1, p[5]-p[4])):
        return False, 0.0, ()
    r2, e3, r4 = (p[1]-p[2])/w1, w3/w1, (p[3]-p[4])/w3
    score = float(.5 <= r2 <= .618) + float(1.618 <= e3 <= 2.618) + float(abs(r4-.384) <= .10)
    return True, score, (r2, e3, r4)


def range_wave1(bars: list[dict], pivots: list[tuple[int, int, float]], atr: float,
                eps: float = .02, count: int = 12) -> dict:
    """Match the MQ5 closed-bar range probe; P1 remains unconfirmed."""
    if len(bars) < count + 5 or atr <= 0:
        return {'candidate': False, 'probe': False}
    history = bars[-count-1:-1]  # excludes signal bar, matches MQ5 shifts 2..N+1
    floor = min(b['l'] for b in history)
    ceiling = max(b['h'] for b in history)
    left, right = len(bars)-count-1, len(bars)-2
    hi = sum(1 for i, k, _ in pivots if left <= i <= right and k == 1 and i+2 <= right)
    lo = sum(1 for i, k, _ in pivots if left <= i <= right and k == -1 and i+2 <= right)
    qualified = ceiling-floor <= 1.8*atr and hi >= 2 and lo >= 2
    return {'candidate': qualified and bars[-1]['c'] > ceiling+eps,
            'probe': qualified and bars[-1]['h'] >= ceiling-eps and bars[-1]['c'] <= ceiling+eps,
            'floor': floor, 'ceiling': ceiling, 'high_pivots': hi, 'low_pivots': lo}


def wick_cluster(bars: list[dict], upper: bool, level: float, atr: float,
                 tick_size: float = .001, lookback: int = 40) -> int:
    """Historical wick-only excursions, excluding the current signal bar."""
    tol = max(2*tick_size, .10*atr)
    count, last = 0, -1000
    for i, bar in enumerate(bars[max(0, len(bars)-lookback-1):-1]):
        body_hi, body_lo = max(bar['o'],bar['c']), min(bar['o'],bar['c'])
        in_wick = (bar['h'] >= level-tol and body_hi < level-tol) if upper else (bar['l'] <= level+tol and body_lo > level+tol)
        if in_wick and i-last > 1:
            count, last = count+1, i
    return count


class VideoStrategy(Strategy):
    def __init__(self, config: Config):
        super().__init__(config)
        # MT5 CopyRates(..., 180) contains shift 0 (forming) plus 179 closed bars.
        self.b1 = deque(maxlen=179)
        self.b15 = deque(maxlen=179)
        self.armed_c = None
        self.armed_d = None
        self.pending_entry = None
        self.entry_pending = False
        self.open_side = 0
        self.stop_px = self.target_px = None
        self.exit_pending = False
        self.signal_count = Counter()
        self.denials = Counter()
        self.completed = []
        self.current_setup = None
        self.last_bid = self.last_ask = None
        self.tick_count = 0
        self.wave_candidate_count = 0
        self.range_probe_count = 0
        self.wick_counts = Counter()
        self.wave_scores = Counter()
        self.wave_valid_count = 0
        self.filled_features = []
        self.current_features = None
        self.spread_denials = 0
        self.entry_spreads = []
        self.wick_target_pending = []
        self.wick_target_outcomes = Counter()

    @staticmethod
    def f(x):
        return float(x.as_double()) if hasattr(x, 'as_double') else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.m1)
        self.subscribe_bars(self.config.m15)

    def on_order_rejected(self, event):
        self.denials['rejected'] += 1
        self.denials['reject_reason: '+str(getattr(event, 'reason', 'unknown'))] += 1
        self.entry_pending = False
        self.pending_entry = None

    def on_order_denied(self, event):
        self.denials['denied'] += 1
        self.entry_pending = False
        self.pending_entry = None

    def on_position_opened(self, event):
        if self.pending_entry is not None:
            self.open_side = self.pending_entry['side']
            self.stop_px = self.pending_entry['stop']
            self.target_px = self.pending_entry['target']
            self.current_setup = self.pending_entry['setup']
            self.current_features = self.pending_entry['features']
            self.pending_entry = None
            self.entry_pending = False

    def on_position_closed(self, event):
        self.completed.append(self.current_setup)
        self.filled_features.append(self.current_features)
        self.current_setup = None
        self.current_features = None
        self.open_side = 0
        self.stop_px = self.target_px = None
        self.exit_pending = False
        self.entry_pending = False

    def on_stop(self):
        self.close_all_positions(self.config.instrument_id)

    def observe_wick_targets(self, row, atr, upper_level, lower_level, upper_count, lower_count):
        """Freeze candidate levels, then count hits on later closed bars only."""
        for target in list(self.wick_target_pending):
            target['remaining']-=1
            hit=(row['h']>=target['level']-target['tol'] if target['kind']=='upper'
                 else row['l']<=target['level']+target['tol'])
            if hit or target['remaining']==0:
                self.wick_target_outcomes[target['kind']+'_hit' if hit else target['kind']+'_miss']+=1
                self.wick_target_pending.remove(target)
        for kind,level,count in (('upper',upper_level,upper_count),('lower',lower_level,lower_count)):
            tol=max(.002,.10*atr)
            if count<3 or level is None or any(p['kind']==kind for p in self.wick_target_pending):
                continue
            if kind=='upper' and row['h']>=level-tol:continue
            if kind=='lower' and row['l']<=level+tol:continue
            self.wick_target_pending.append(dict(kind=kind,level=level,tol=tol,remaining=15))
            self.wick_target_outcomes[kind+'_armed']+=1

    def on_bar(self, bar: Bar):
        row = dict(o=self.f(bar.open), h=self.f(bar.high), l=self.f(bar.low),
                   c=self.f(bar.close), ts=int(bar.ts_event))
        if str(bar.bar_type) == str(self.config.m15):
            self.b15.append(row)
            return
        if str(bar.bar_type) != str(self.config.m1):
            return
        self.b1.append(row)
        bs = list(self.b1)
        ht = [b for b in self.b15 if b['ts'] <= row['ts']]
        if len(bs) < 179 or len(ht) < 179:
            return
        tr = [max(bs[i]['h']-bs[i]['l'], abs(bs[i]['h']-bs[i-1]['c']), abs(bs[i]['l']-bs[i-1]['c']))
              for i in range(1,len(bs))]
        atr = float(np.mean(tr[:14]))
        for value in tr[14:]:
            atr += (value-atr)/14.0
        if atr <= 0:
            return
        s1, s15 = swings(bs), swings(ht)
        ldir, ld = dow(s1)
        hdir, _ = dow(s15)
        wave1=range_wave1(bs,s1,atr)
        if wave1['candidate']:self.wave_candidate_count+=1
        if wave1['probe']:self.range_probe_count+=1
        upper=wick_cluster(bs,True,ld['highs'][-1][1],atr) if ld['highs'] else 0
        lower=wick_cluster(bs,False,ld['lows'][-1][1],atr) if ld['lows'] else 0
        if upper>=3:self.wick_counts['upper']+=1
        if lower>=3:self.wick_counts['lower']+=1
        self.observe_wick_targets(row,atr,ld['highs'][-1][1] if ld['highs'] else None,
                                  ld['lows'][-1][1] if ld['lows'] else None,upper,lower)
        if self.open_side or self.entry_pending or self.pending_entry:return
        picks = []
        # A: historical bullish supply candle + bearish impulse + return/rejection.
        if hdir == -1 and row['c'] < bs[-2]['l']-.02:
            for j in range(len(bs)-5,max(len(bs)-26,-1),-1):
                if j <= 2: break
                b, impulse = bs[j], bs[j+1]
                if b['c'] <= b['o'] or impulse['o']-impulse['c'] < atr:
                    continue
                if any(x['c'] > b['h']+.02 for x in bs[j+2:-2]):
                    continue
                reject = bs[-2]
                rng = reject['h']-reject['l']
                if rng > 0 and reject['h'] >= b['o'] and reject['l'] <= b['h'] and reject['c']<=b['h']+.02 and (reject['h']-max(reject['o'],reject['c']))/rng >= .45:
                    picks.append(('A',-1,max(reject['h'],b['h'])+.03,b['o']))
                    break
        # B: weak consolidation after a downward impulse, break below its low.
        if hdir == -1 and len(bs) >= 17:
            middle=bs[-13:-1]; imp=bs[-14]
            hi=max(b['h'] for b in middle);lo=min(b['l'] for b in middle)
            if imp['o']-imp['c'] >= atr and hi-lo < 1.5*atr and hi-imp['l'] < 1.5*atr and row['c'] < lo-.02 and row['c'] < row['o']:
                picks.append(('B',-1,hi+.03,lo))
        # C: arm on an older confirmed pivot high breakout; retest on a later bar.
        if hdir != 1:
            self.armed_c=None
        elif self.armed_c:
            p=self.armed_c
            if row['ts']-p['ts']>15*60*1_000_000_000 or row['c']<p['level']-.02:
                self.armed_c=None
            elif row['l']<=p['level']+.02 and row['c']>p['level']+.02 and row['h']>bs[-2]['h']:
                picks.append(('C',1,min(p['extreme'],row['l'])-.03,p['level']))
                self.armed_c=None
            else:
                p['extreme']=min(p['extreme'],row['l'])
        elif ld['highs']:
            ix, res=ld['highs'][-1]
            if ix < len(bs)-2 and bs[-2]['c']<=res+.02 and row['c']>res+.02:
                self.armed_c=dict(ts=row['ts'],level=res,extreme=row['l'])
        # D: fixed line through 2 confirmed equal-kind pivots, break and retest.
        if self.armed_d:
            p=self.armed_d
            line=p['level']+p['slope']*(row['ts']-p['ts'])/1_000_000_000
            if hdir != -p['side'] or row['ts']-p['ts']>15*60*1_000_000_000 or (row['c']-line)*p['side']<-.02:
                self.armed_d=None
            elif row['l']<=line+.02<=row['h']+.04 and (row['c']-line)*p['side']>.02 and (row['h']>bs[-2]['h'] if p['side']==1 else row['l']<bs[-2]['l']):
                stop=min(row['l'],p['extreme'])-.03 if p['side']==1 else max(row['h'],p['extreme'])+.03
                picks.append(('D',p['side'],stop,line))
                self.armed_d=None
            else:
                p['extreme']=min(p['extreme'],row['l']) if p['side']==1 else max(p['extreme'],row['h'])
        elif hdir in (-1,1):
            side=-hdir
            anchors=ld['highs'] if side==1 else ld['lows']
            if len(anchors)>=2:
                (ia,pa),(ib,pb)=anchors[-2:]
                if ib>ia and ib<len(bs)-3:
                    slope=(pb-pa)/((bs[ib]['ts']-bs[ia]['ts'])/1_000_000_000)
                    line=pb+slope*(row['ts']-bs[ib]['ts'])/1_000_000_000
                    prev=pb+slope*(bs[-2]['ts']-bs[ib]['ts'])/1_000_000_000
                    if (bs[-2]['c']-prev)*side<=.02 and (row['c']-line)*side>.02 and (row['c']-pb)*side>.02:
                        self.armed_d=dict(ts=row['ts'],level=line,slope=slope,extreme=row['l'] if side==1 else row['h'],side=side)
        if len(picks) != 1:
            if len(picks)>1:self.denials['ambiguous']+=1
            return
        name,side,stop,level=picks[0]
        valid,score,ratios=elliott(s1,side)
        self.signal_count[name]+=1
        self.wave_scores[str(int(score)) if valid else 'invalid']+=1
        if valid:self.wave_valid_count+=1
        if self.config.elliott_gate and (not valid or score<2):
            self.denials['wave']+=1
            return
        features={'wave_valid':valid,'wave_score':score,'fib_ratios':ratios,
                  'wave1_range_candidate':wave1['candidate'], 'range_probe':wave1['probe'],
                  'upper_wick_cluster':upper>=3,'lower_wick_cluster':lower>=3}
        self.pending_entry=dict(setup=name,side=side,stop=stop,signal_ts=row['ts'],level=level,
                                wave_score=score,features=features)

    def on_quote_tick(self, tick):
        self.tick_count+=1
        bid=self.f(tick.bid_price);ask=self.f(tick.ask_price)
        self.last_bid=bid;self.last_ask=ask
        if ask<=bid:return
        if self.open_side and not self.exit_pending:
            if (self.open_side==1 and (bid<=self.stop_px or bid>=self.target_px)) or (self.open_side==-1 and (ask>=self.stop_px or ask<=self.target_px)):
                self.exit_pending=True
                self.close_all_positions(self.config.instrument_id)
            return
        p=self.pending_entry
        if p is None or self.entry_pending or int(tick.ts_event)<=p['signal_ts']:
            return
        price=ask if p['side']==1 else bid
        self.entry_spreads.append(ask-bid)
        if self.config.max_spread>0 and ask-bid>self.config.max_spread:
            self.pending_entry=None;self.denials['spread']+=1
            return
        # MT5 validates a long stop against bid and a short stop against ask.
        # The catalog has no broker-specific SYMBOL_TRADE_STOPS_LEVEL; assume zero.
        if (bid-p['stop'] if p['side']==1 else p['stop']-ask)<=0:
            self.denials['invalid_stop_distance']+=1
            self.pending_entry=None
            return
        p['target']=price+p['side']*2.0*abs(price-p['stop'])
        self.entry_pending=True
        inst=self.cache.instrument(self.config.instrument_id)
        order=self.order_factory.market(instrument_id=self.config.instrument_id,
              order_side=OrderSide.BUY if p['side']==1 else OrderSide.SELL,
              quantity=inst.make_qty(self.config.size))
        self.submit_order(order)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--elliott-gate',action='store_true')
    ap.add_argument('--max-spread',type=float,default=.40,
                    help='MQ5 entry spread ceiling in quoted price units; 0 disables for sensitivity study')
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    root=Path(args.catalog)
    manifest=json.loads((root/'catalog_manifest.json').read_text())
    if manifest.get('status')!='COMPLETE' or manifest.get('data_kind')!='RAW_BIDASK' or manifest.get('ohlc_resample_used') is not False:
        raise SystemExit('RAW_CATALOG_FAIL_CLOSED')
    catalog=ParquetDataCatalog(str(root))
    inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD_INSTRUMENT_MISSING')
    ticks=catalog.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:raise SystemExit('RAW_QUOTE_TICKS_MISSING')
    one=Quantity.from_int(1)
    repaired_sizes=0
    for i,t in enumerate(ticks):
        if float(t.bid_size.as_double())<=0 or float(t.ask_size.as_double())<=0:
            ticks[i]=QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,
                bid_size=one if float(t.bid_size.as_double())<=0 else t.bid_size,
                ask_size=one if float(t.ask_size.as_double())<=0 else t.ask_size,
                ts_event=t.ts_event,ts_init=t.ts_init)
            repaired_sizes+=1
    cfg=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True))
    engine=BacktestEngine(config=cfg)
    engine.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,
                     book_type=BookType.L1_MBP,base_currency=USD,
                     starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(inst);engine.add_data(ticks)
    strat=VideoStrategy(Config(instrument_id=inst.id,
                 m1=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL'),
                 m15=BarType.from_str(f'{inst.id.value}-15-MINUTE-BID-INTERNAL'),
                 size=Decimal('1'),elliott_gate=args.elliott_gate,max_spread=args.max_spread))
    engine.add_strategy(strat);engine.run()
    report=engine.trader.generate_positions_report()
    trades=extract_trades(report,'XAUUSD','M1')
    trades.sort(key=lambda t:t['ts_closed'])
    for i,t in enumerate(trades):t['setup']=strat.completed[i] if i<len(strat.completed) else 'UNMATCHED'
    if trades and any(t['setup']=='UNMATCHED' for t in trades):
        raise SystemExit('POSITION_SETUP_MATCH_FAIL_CLOSED')
    days=int(manifest['days']);m=metrics(trades,initial=1000,days=days)
    m['EV_USD']=m['NetProfit']/m['N'] if m['N'] else None
    m['Return_pct']=m['NetProfit']/1000*100
    m['MaxDD_kind']='REALIZED_CLOSE_ONLY'
    by={name:metrics([t for t in trades if t['setup']==name],initial=1000,days=days) for name in 'ABCD'}
    if len(strat.filled_features)!=len(trades):
        raise SystemExit('POSITION_FEATURE_MATCH_FAIL_CLOSED')
    for trade,feature in zip(trades,strat.filled_features):
        for key in ('wave_valid','wave_score','wave1_range_candidate','range_probe',
                    'upper_wick_cluster','lower_wick_cluster'):
            trade[key]=feature[key]
        trade['fib_ratios']=json.dumps(feature['fib_ratios'])
    feature_metrics={key:{str(value):metrics([t for t in trades if t[key]==value],initial=1000,days=days)
                   for value in (False,True)} for key in
                   ('wave_valid','wave1_range_candidate','range_probe','upper_wick_cluster','lower_wick_cluster')}
    summary=dict(status='COMPLETED',verification_level='NAUTILUS_RAW_BIDASK_SIGNAL_GATE',
      version=nautilus_trader.__version__,source='Dukascopy via Nautilus ParquetDataCatalog',
      raw_ticks=len(ticks),zero_size_quotes_repaired=repaired_sizes,
      period=dict(start=manifest['start'],days=days,end_exclusive=manifest['end_exclusive']),
      config=dict(symbol='XAUUSD',signal_tf='M1',trend_tf='M15',size='1',initial_usd=1000,leverage='2000',
                  elliott_gate=args.elliott_gate,max_spread=args.max_spread,reward_risk=2.0),
      overall=m,by_setup=by,by_feature=feature_metrics,
      signals=dict(strat.signal_count),wave_scores=dict(strat.wave_scores),
      wave_valid_signals=strat.wave_valid_count,denials=dict(strat.denials),
      range_wave1_candidates=strat.wave_candidate_count,engine_tick_events=strat.tick_count,
      entry_spread_observed=dict(n=len(strat.entry_spreads),median=float(np.median(strat.entry_spreads)) if strat.entry_spreads else None,
                                 minimum=min(strat.entry_spreads) if strat.entry_spreads else None,
                                 maximum=max(strat.entry_spreads) if strat.entry_spreads else None),
      range_probes=strat.range_probe_count,wick_cluster_bars=dict(strat.wick_counts),
      wick_target_15bar=dict(strat.wick_target_outcomes),
      bar_tail_counts=dict(m1=len(strat.b1),m15=len(strat.b15)),
      limitations=['MQ5 port comparison pending; signal feature definitions aligned but MT5 compiled parity not verified',
                   'Realized closed-equity MaxDD; synchronized floating MaxDD not implemented',
                   'Raw Bid/Ask native spread; explicit fee, latency and slippage models not implemented',
                   'Broker-specific stop/freeze levels unavailable; assumed zero',
                   'Wick target 15-bar hit rates are descriptive; no distance-matched control or causal attraction claim',
                   'Fixed 1-unit quantity; MQ5 lot-step and risk sizing parity pending',
                   'No OOS claim; repeatability and setup mapping require separate validation'])
    pd.DataFrame(trades).to_csv(out/'trades.csv',index=False)
    def finite(value):
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    summary=finite(summary)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    (out/'catalog_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    engine.dispose()


if __name__=='__main__':main()
