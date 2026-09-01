from __future__ import annotations

from nautilus_trader.persistence.catalog import ParquetDataCatalog

# NautilusTrader catalog reader names differ by release. Preserve Raw QuoteTick
# semantics and fail closed rather than ever falling back to OHLC/resampled data.
if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    if hasattr(ParquetDataCatalog, "quote_ticks"):
        def _query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quote_ticks(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = _query_quote_ticks
    elif hasattr(ParquetDataCatalog, "quotes"):
        def _query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.quotes(instrument_ids=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = _query_quote_ticks
    elif hasattr(ParquetDataCatalog, "query"):
        from nautilus_trader.model.data import QuoteTick as _CompatQuoteTick
        def _query_quote_ticks(self, identifiers=None, start=None, end=None, **kwargs):
            return self.query(data_cls=_CompatQuoteTick, identifiers=identifiers, start=start, end=end, **kwargs)
        ParquetDataCatalog.query_quote_ticks = _query_quote_ticks
    else:
        raise RuntimeError("No Raw QuoteTick reader found on ParquetDataCatalog")

import argparse, json, math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from research.minimumspike_raw6x3_bt import SIM, TF_MIN, TRADE_SIZE, extract_trades, metrics
from research.amos_math_ict_v1_4 import (
    MarketState, ICTState, FVGZone, ZoneLifecycle, EntryContext,
    LifecycleAwareEntryRouter, ExitRouter, ExitState, activate_ifvg, make_bpr,
)

class AmosICTConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    tf_minutes: int
    entry_variant: str
    exit_variant: str

class AmosMathICTStrategy(Strategy):
    def __init__(self, config: AmosICTConfig):
        super().__init__(config)
        self.bars=deque(maxlen=64); self.fvgs=[]; self.ifvgs=[]; self.bprs=[]
        self.bias=0; self.last_swing_high=None; self.last_swing_low=None
        self.entry_px=None; self.entry_ts=None; self.mfe=0.0; self.mae=0.0; self.last_atr=0.0
        self.exit_pending=False; self.exit_reason=''; self.entries=0; self.last_ict=ICTState()
        self.entry_router=LifecycleAwareEntryRouter(enable_liquidity=(config.entry_variant=='E04'),enable_regime=False,enable_ai=False,enable_harmonic=False,min_effective_groups=2,min_liquidity_score=1.0 if config.entry_variant=='E04' else 0.0)
        self.exit_router=ExitRouter(enable_structure_exit=True,enable_liquidity_take=(config.exit_variant=='X06'),enable_breakeven=config.exit_variant in ('X04','X06'),enable_atr_trail=config.exit_variant in ('X04','X06'),enable_time_stop=(config.exit_variant=='X06'),enable_ai_exit=False)
    @staticmethod
    def _f(px): return float(px.as_double()) if hasattr(px,'as_double') else float(px)
    @staticmethod
    def _is_long(pos):
        v=getattr(pos,'is_long',False); return bool(v() if callable(v) else v)
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id); self.subscribe_bars(self.config.bar_type)
    def _atr14(self):
        if len(self.bars)<15: return None
        xs=list(self.bars); trs=[]
        for i in range(-14,0):
            cur,prev=xs[i],xs[i-1]; trs.append(max(cur['h']-cur['l'],abs(cur['h']-prev['c']),abs(cur['l']-prev['c'])))
        a=float(np.mean(trs)); return a if math.isfinite(a) and a>0 else None
    def _update_swings(self):
        if len(self.bars)<3: return
        a,b,c=list(self.bars)[-3:]
        if b['h']>a['h'] and b['h']>c['h']: self.last_swing_high=b['h']
        if b['l']<a['l'] and b['l']<c['l']: self.last_swing_low=b['l']
    def _detect(self,b):
        ict=ICTState(direction=self.bias); xs=list(self.bars)
        if len(xs)>=3:
            a,_,c=xs[-3],xs[-2],xs[-1]
            if c['l']>a['h']: self.fvgs.append(FVGZone(direction=1,lower=a['h'],upper=c['l'],created_ts_ns=c['ts'],lifecycle=ZoneLifecycle.ACTIVE)); ict.fvg=True
            if c['h']<a['l']: self.fvgs.append(FVGZone(direction=-1,lower=c['h'],upper=a['l'],created_ts_ns=c['ts'],lifecycle=ZoneLifecycle.ACTIVE)); ict.fvg=True
        prev_bias=self.bias
        if self.last_swing_high is not None:
            if b['h']>self.last_swing_high and b['c']<self.last_swing_high: ict.sweep=True; ict.direction=-1
            if b['c']>self.last_swing_high:
                self.bias=1; ict.direction=1; ict.bos=True
                if prev_bias<0: ict.mss=True; ict.choch=True
        if self.last_swing_low is not None:
            if b['l']<self.last_swing_low and b['c']>self.last_swing_low: ict.sweep=True; ict.direction=1
            if b['c']<self.last_swing_low:
                self.bias=-1; ict.direction=-1; ict.bos=True
                if prev_bias>0: ict.mss=True; ict.choch=True
        for z in self.fvgs[-20:]:
            if z.lifecycle!=ZoneLifecycle.ACTIVE: continue
            if z.direction==1 and b['c']<z.lower: z.lifecycle=ZoneLifecycle.BROKEN; z.break_direction=-1; z.broken_ts_ns=b['ts']; self.ifvgs.append(activate_ifvg(z,b['ts']))
            elif z.direction==-1 and b['c']>z.upper: z.lifecycle=ZoneLifecycle.BROKEN; z.break_direction=1; z.broken_ts_ns=b['ts']; self.ifvgs.append(activate_ifvg(z,b['ts']))
        active_ifvg=None
        for inv in self.ifvgs[-20:]:
            z=inv.source_fvg
            if inv.active and z and b['h']>=z.lower and b['l']<=z.upper: inv.retested=True; active_ifvg=inv; ict.direction=inv.direction
        bull=[z for z in self.fvgs[-20:] if z.direction==1 and z.lifecycle!=ZoneLifecycle.INVALIDATED]; bear=[z for z in self.fvgs[-20:] if z.direction==-1 and z.lifecycle!=ZoneLifecycle.INVALIDATED]
        if bull and bear:
            bz=make_bpr(bull[-1],bear[-1],b['ts'])
            if bz is not None: self.bprs.append(bz)
        active_bpr=None; bpr_retested=False; bpr_rejected=False
        for bz in self.bprs[-10:]:
            if b['h']>=bz.lower and b['l']<=bz.upper:
                bpr_retested=True; active_bpr=bz
                if b['c']>bz.upper: bz.rejection_direction=1; bpr_rejected=True; ict.direction=1
                elif b['c']<bz.lower: bz.rejection_direction=-1; bpr_rejected=True; ict.direction=-1
        self.last_ict=ict; return EntryContext(ict=ict,ifvg=active_ifvg,bpr=active_bpr,bpr_retested=bpr_retested,bpr_rejected=bpr_rejected)
    def on_bar(self,bar:Bar):
        b={'o':self._f(bar.open),'h':self._f(bar.high),'l':self._f(bar.low),'c':self._f(bar.close),'ts':int(bar.ts_event)}; self.bars.append(b); a=self._atr14(); self.last_atr=a or self.last_atr; ctx=self._detect(b); self._update_swings()
        if self.entry_px is not None or not self.portfolio.is_net_flat(self.config.instrument_id): return
        g=ctx.effective_groups(); v=self.config.entry_variant
        if v=='E00': ctx.ifvg=None; ctx.bpr=None; ctx.bpr_retested=False; ctx.bpr_rejected=False
        elif v=='E01' and not g['ifvg_transition']: return
        elif v=='E02' and not g['bpr_location']: return
        elif v=='E03' and not (g['ifvg_transition'] and g['bpr_location']): return
        elif v=='E04' and not g['liquidity']: return
        d=self.entry_router.decide(MarketState(b['ts'],b['c'],b['c'],0.0,timeframe=str(self.config.tf_minutes),liquidity_score=1.0 if g['liquidity'] else 0.0),ctx)
        if d.action in ('BUY','SELL'):
            inst=self.cache.instrument(self.config.instrument_id); side=OrderSide.BUY if d.action=='BUY' else OrderSide.SELL; self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=inst.make_qty(self.config.trade_size))); self.entry_px=b['c']; self.entry_ts=b['ts']; self.mfe=0.0; self.mae=0.0; self.exit_pending=False; self.entries+=1
    def on_quote_tick(self,tick:QuoteTick):
        if self.entry_px is None or self.exit_pending: return
        bid,ask=self._f(tick.bid_price),self._f(tick.ask_price); mid=(bid+ask)/2; positions=self.cache.positions_open(instrument_id=self.config.instrument_id)
        if not positions: return
        long=self._is_long(positions[0]); move=(mid-self.entry_px) if long else (self.entry_px-mid); self.mfe=max(self.mfe,move); self.mae=min(self.mae,move); structure_bad=(long and self.bias<0) or ((not long) and self.bias>0); opposite=(-1 if long else 1)
        s=ExitState(entry_price=self.entry_px,current_price=mid,pnl=move,mfe=self.mfe,mae=self.mae,atr=self.last_atr,seconds_held=max(0,(int(tick.ts_event)-self.entry_ts)/1e9),structure_invalidated=structure_bad,opposite_mss=self.last_ict.mss and self.last_ict.direction==opposite,opposite_choch=self.last_ict.choch and self.last_ict.direction==opposite,liquidity_target_hit=self.last_ict.sweep and move>0,spread=ask-bid); d=self.exit_router.decide(s)
        if d.action=='CLOSE': self.exit_reason=d.reason; self.close_all_positions(self.config.instrument_id); self.exit_pending=True
    def on_position_closed(self,event): self.entry_px=None; self.entry_ts=None; self.mfe=0.0; self.mae=0.0; self.exit_pending=False
    def on_stop(self): self.close_all_positions(self.config.instrument_id)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--symbols',nargs='+',required=True); ap.add_argument('--timeframes',nargs='+',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--entry-variant',default='E00',choices=['E00','E01','E02','E03','E04']); ap.add_argument('--exit-variant',default='X00',choices=['X00','X04','X06']); args=ap.parse_args()
    if not args.raw_bidask_only: raise SystemExit('raw-bidask-only is mandatory')
    catalog_path=Path(args.catalog); manifest=json.loads((catalog_path/'catalog_manifest.json').read_text()); days=int(manifest['days']); catalog=ParquetDataCatalog(str(catalog_path)); inst_by_plain={x.id.symbol.value.replace('/',''):x for x in catalog.instruments()}; outdir=Path('results/ae-bt')/args.experiment_id; outdir.mkdir(parents=True,exist_ok=True); all_trades=[]; cells={}; counts={}
    for symbol in args.symbols:
        inst=inst_by_plain.get(symbol)
        if inst is None: raise SystemExit(f'instrument missing: {symbol}')
        ticks=catalog.query_quote_ticks(identifiers=[inst.id.value]); counts[symbol]=len(ticks)
        if not ticks: raise SystemExit(f'no raw QuoteTicks: {symbol}')
        for tf in args.timeframes:
            engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True))); engine.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000')); engine.add_instrument(inst); engine.add_data(ticks); bt=BarType.from_str(f'{inst.id.value}-{TF_MIN[tf]}-MINUTE-BID-INTERNAL'); strat=AmosMathICTStrategy(AmosICTConfig(instrument_id=inst.id,bar_type=bt,trade_size=TRADE_SIZE[symbol],tf_minutes=TF_MIN[tf],entry_variant=args.entry_variant,exit_variant=args.exit_variant)); engine.add_strategy(strat); engine.run(); trades=extract_trades(engine.generate_positions_report(),symbol,tf); all_trades.extend(trades); cells[f'{symbol}:{tf}']={**metrics(trades,days=days),'raw_ticks':len(ticks),'signals_submitted':strat.entries}; engine.dispose()
    all_trades.sort(key=lambda x:(x['ts_closed'],x['symbol'],x['tf'])); summary={'verification_level':'NAUTILUS_BT_RAW_BIDASK','strategy':'AMOS_MATH_ICT_V1_4','entry_variant':args.entry_variant,'exit_variant':args.exit_variant,'data_kind':'RAW_BIDASK QuoteTick','ohlc_resample_used':False,'symbols':args.symbols,'timeframes':args.timeframes,'period':manifest,'portfolio_realized_close_ordered':metrics(all_trades,days=days),'cell_metrics':cells,'raw_tick_counts':counts,'limitations':['First gate E00-E04 only; synchronized MTF E05-E09 deferred.','Spread is native Bid/Ask; explicit commission/probabilistic slippage model still pending.','Portfolio DD is realized-close ordered across independent cells.']}; pd.DataFrame(all_trades).to_csv(outdir/'trades.csv',index=False); (outdir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)); print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
