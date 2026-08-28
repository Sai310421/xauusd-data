from __future__ import annotations

"""Raw Bid/Ask Nautilus A/B: BASE vs AE multivariable-conformal proxy.

The EDGE arm is deliberately labelled DERIVED_PROXY, not the exact source-paper
construction. It uses only information available before each entry. A positive
result is a screening pass, not proof of conformal edge.
"""

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

from research.math_edge_candidates import empirical_joint_box, ae_safe_edge
from research.minimumspike_raw6x3_bt import PARAMS, TF_MIN, TRADE_SIZE, extract_trades, metrics

SIM = Venue('SIM')


class MVCConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    tf_minutes: int
    edge_enabled: bool = False
    alpha: float = 0.10
    calibration_window: int = 160
    min_calibration: int = 40
    edge_threshold: float = 0.0
    min_scale: float = 0.35


class MVCStrategy(Strategy):
    def __init__(self, config: MVCConfig):
        super().__init__(config)
        self.bars = deque(maxlen=256)
        self.outcomes = deque(maxlen=config.calibration_window)
        self.pending = None
        self.armed = None
        self.entry_ref = self.stop_ref = self.tp_ref = self.trail_ref = None
        self.hold_bars = 0
        self.exit_pending = False
        self.entries = 0
        self.blocked = 0
        self.scales = []

    @staticmethod
    def _f(x):
        return float(x.as_double()) if hasattr(x, 'as_double') else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def _atr14(self):
        if len(self.bars) < 15:
            return None
        xs = list(self.bars); tr=[]
        for i in range(-14,0):
            c,p=xs[i],xs[i-1]
            tr.append(max(c['h']-c['l'],abs(c['h']-p['c']),abs(c['l']-p['c'])))
        a=float(np.mean(tr))
        return a if math.isfinite(a) and a>0 else None

    def _update_outcome(self, b):
        # One-bar delayed, strictly past-data label. Components are normalized by
        # previous ATR: MFE, MAE, proxy tau-to-BE, downside-tail depth.
        if not self.bars:
            return
        prev=self.bars[-1]
        a=prev.get('atr')
        if not a or a<=0:
            return
        mfe=max(0.0,(b['h']-prev['c'])/a)
        mae=max(0.0,(prev['c']-b['l'])/a)
        tau=0.0 if b['h']>=prev['c'] else 1.0
        tail=max(0.0,(prev['c']-b['l'])/a)
        self.outcomes.append([mfe,mae,tau,tail])

    def _edge_scale(self):
        if not self.config.edge_enabled or len(self.outcomes)<self.config.min_calibration:
            return 1.0
        x=np.asarray(self.outcomes,float)
        # Expanding/rolling historical mean is the point forecast. Residuals are
        # historical only; no future leakage.
        point=x.mean(axis=0)
        residuals=x-point
        b=empirical_joint_box(residuals,point,alpha=self.config.alpha)
        score=ae_safe_edge(b,lambda_mae=0.65,lambda_tau=0.10,lambda_tail=0.35)
        if score<=self.config.edge_threshold:
            return 0.0
        # Smooth governor; never lever above BASE in screening stage.
        scale=max(self.config.min_scale,min(1.0,1.0-math.exp(-score)))
        return float(scale)

    def on_bar(self, bar: Bar):
        b={'o':self._f(bar.open),'h':self._f(bar.high),'l':self._f(bar.low),'c':self._f(bar.close),'ts':int(bar.ts_event)}
        self._update_outcome(b)
        a_before=self._atr14()
        b['atr']=a_before
        self.bars.append(b)
        if self.entry_ref is not None:self.hold_bars+=1
        if self.pending is not None:
            p=self.pending
            if b['ts']>p['ts']:
                if b['c']-p['low']>=PARAMS['rebound_atr']*p['atr'] and b['c']>b['o']:
                    self.armed=dict(p); self.pending=None
                else:
                    p['remaining']-=1
                    if p['remaining']<=0:self.pending=None
        a=self._atr14()
        if a is not None:
            body=b['o']-b['c']; rng=b['h']-b['l']
            if b['c']<b['o'] and body>=PARAMS['spike_body_atr']*a and rng>=PARAMS['spike_range_atr']*a:
                self.pending={'low':b['l'],'atr':a,'remaining':PARAMS['confirm_bars'],'ts':b['ts']}
        max_hold=max(1,int(PARAMS['max_hold_minutes']/self.config.tf_minutes))
        if self.entry_ref is not None and self.hold_bars>=max_hold and not self.exit_pending:
            self.close_all_positions(self.config.instrument_id); self.exit_pending=True

    def on_quote_tick(self,tick:QuoteTick):
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price)
        if self.armed is not None and self.entry_ref is None and self.portfolio.is_net_flat(self.config.instrument_id):
            scale=self._edge_scale()
            if scale<=0:
                self.blocked+=1; self.armed=None; return
            inst=self.cache.instrument(self.config.instrument_id)
            qty=Decimal(str(float(self.config.trade_size)*scale))
            # Quantize through instrument rules; if too small, skip.
            try:q=inst.make_qty(qty)
            except Exception:
                self.blocked+=1; self.armed=None; return
            if float(q)<=0:
                self.blocked+=1; self.armed=None; return
            order=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY,quantity=q)
            self.submit_order(order)
            p=self.armed; self.entry_ref=ask; self.stop_ref=p['low']-PARAMS['guard_ext']*p['atr']; self.tp_ref=ask+PARAMS['tp_atr']*p['atr']
            self.trail_ref=None; self.hold_bars=0; self.exit_pending=False; self.entries+=1; self.scales.append(scale); self.armed=None; return
        if self.entry_ref is None or self.exit_pending:return
        atr=(self.tp_ref-self.entry_ref)/PARAMS['tp_atr']
        if bid>=self.entry_ref+PARAMS['trail_act']*atr:
            nt=bid-PARAMS['trail_dist']*atr; self.trail_ref=nt if self.trail_ref is None else max(self.trail_ref,nt)
        active=max(self.stop_ref,self.trail_ref) if self.trail_ref is not None else self.stop_ref
        if self.hold_bars>=PARAMS['min_hold'] and (bid<=active or bid>=self.tp_ref):
            self.close_all_positions(self.config.instrument_id); self.exit_pending=True

    def on_position_closed(self,event):
        self.entry_ref=self.stop_ref=self.tp_ref=self.trail_ref=None; self.hold_bars=0; self.exit_pending=False
    def on_stop(self):self.close_all_positions(self.config.instrument_id)


def run_arm(catalog, inst_by_plain, symbols, tfs, days, edge):
    trades=[]; cells={}; raw_counts={}
    for symbol in symbols:
        inst=inst_by_plain.get(symbol)
        if inst is None: raise SystemExit(f'instrument missing: {symbol}')
        ticks=catalog.query_quote_ticks(identifiers=[inst.id.value]); raw_counts[symbol]=len(ticks)
        if not ticks: raise SystemExit(f'no raw QuoteTicks: {symbol}')
        for tf in tfs:
            m=TF_MIN[tf]
            engine=BacktestEngine(config=BacktestEngineConfig(trader_id=f"{'E' if edge else 'B'}-{symbol}-{tf}",logging=LoggerConfig(stdout_level=LogLevel.ERROR),risk_engine=RiskEngineConfig(bypass=True)))
            engine.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
            engine.add_instrument(inst); engine.add_data(ticks)
            bt=BarType.from_str(f'{inst.id.value}-{m}-MINUTE-BID-INTERNAL')
            s=MVCStrategy(MVCConfig(instrument_id=inst.id,bar_type=bt,trade_size=TRADE_SIZE[symbol],tf_minutes=m,edge_enabled=edge))
            engine.add_strategy(s); engine.run(); r=engine.generate_positions_report(); tt=extract_trades(r,symbol,tf); trades.extend(tt)
            cells[f'{symbol}:{tf}']={**metrics(tt,days=days),'signals_submitted':s.entries,'blocked':s.blocked,'mean_size_scale':float(np.mean(s.scales)) if s.scales else None,'raw_ticks':len(ticks)}
            engine.dispose()
    trades.sort(key=lambda z:(z['ts_closed'],z['symbol'],z['tf']))
    return trades, metrics(trades,days=days), cells, raw_counts


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--symbols',nargs='+',required=True); ap.add_argument('--timeframes',nargs='+',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--raw-bidask-only',action='store_true'); args=ap.parse_args()
    if not args.raw_bidask_only: raise SystemExit('raw-bidask-only is mandatory')
    cp=Path(args.catalog); manifest=json.loads((cp/'catalog_manifest.json').read_text()); days=int(manifest['days']); catalog=ParquetDataCatalog(str(cp)); inst={x.id.symbol.value.replace('/',''):x for x in catalog.instruments()}
    out=Path('results/ae-bt')/args.experiment_id; out.mkdir(parents=True,exist_ok=True)
    bt,bm,bc,rc=run_arm(catalog,inst,args.symbols,args.timeframes,days,False)
    et,em,ec,_=run_arm(catalog,inst,args.symbols,args.timeframes,days,True)
    result={'verification_level':'NAUTILUS_BT_RAW_BIDASK','candidate':'MULTIVARIABLE_CONFORMAL_DERIVED_PROXY','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'dataset_sha256':manifest.get('catalog_sha256'),'ohlc_resample_used':False,'baseline':bm,'edge':em,'delta':{k:(em.get(k)-bm.get(k) if isinstance(em.get(k),(int,float)) and isinstance(bm.get(k),(int,float)) else None) for k in set(bm)|set(em)},'baseline_cells':bc,'edge_cells':ec,'raw_tick_counts':rc,'causality':'rolling historical one-bar outcomes only; no future leakage','method_status':'DERIVED_PROXY_SCREENING','limitations':['Not the exact multi-variable conformal source construction.','Independent symbol/TF engines; portfolio DD reconstructed from realized closes.','Spread native from raw Bid/Ask; explicit commission/slippage model not yet added.']}
    pd.DataFrame(bt).to_csv(out/'baseline_trades.csv',index=False); pd.DataFrame(et).to_csv(out/'edge_trades.csv',index=False); (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)); (out/'catalog_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)); print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
