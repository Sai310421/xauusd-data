from __future__ import annotations

import argparse, json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Quantity
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from armada_nautilus_raw_cell import ArmadaCandidate, ArmadaConfig, TF_MIN, _status_counts, _tick_diag


def _positive_size_ticks(ticks):
    fixed=[]; replaced=0
    for t in ticks:
        bid_sz=float(t.bid_size.as_double()) if hasattr(t.bid_size,'as_double') else float(t.bid_size)
        ask_sz=float(t.ask_size.as_double()) if hasattr(t.ask_size,'as_double') else float(t.ask_size)
        if bid_sz>0 and ask_sz>0:
            fixed.append(t); continue
        fixed.append(QuoteTick(
            instrument_id=t.instrument_id,
            bid_price=t.bid_price,
            ask_price=t.ask_price,
            bid_size=t.bid_size if bid_sz>0 else Quantity.from_int(1),
            ask_size=t.ask_size if ask_sz>0 else Quantity.from_int(1),
            ts_event=t.ts_event,
            ts_init=t.ts_init,
        ))
        replaced += 1
    return fixed,replaced


class ArmadaApexStage3(ArmadaCandidate):
    """State-dependent protection without changing the clean-room R2 entry logic.

    The controller varies the adverse threshold by trade state:
      * stale/no-MFE trades are cut earlier after N bars,
      * regime deterioration tightens the stop,
      * account drawdown tightens protection further,
      * otherwise a wider hard stop preserves winners.

    This is intentionally deterministic and causal: only information available at
    the current bar/tick is used.
    """
    def __init__(
        self,
        config,
        hard_stop_atr: float,
        stale_stop_atr: float,
        stale_after_bars: int,
        min_mfe_atr: float,
        regime_stop_atr: float,
        regime_floor: float,
        dd_soft_pct: float,
        dd_stop_atr: float,
    ):
        super().__init__(config)
        self.hard_stop_atr=float(hard_stop_atr)
        self.stale_stop_atr=float(stale_stop_atr)
        self.stale_after_bars=int(stale_after_bars)
        self.min_mfe_atr=float(min_mfe_atr)
        self.regime_stop_atr=float(regime_stop_atr)
        self.regime_floor=float(regime_floor)
        self.dd_soft_pct=float(dd_soft_pct)
        self.dd_stop_atr=float(dd_stop_atr)
        self.regime_score=0.0
        self.exit_reasons=Counter()
        self.state_samples=Counter()
        self.current_dd_pct=0.0
        self.dynamic_threshold_last=self.hard_stop_atr

    def _equity_dd(self):
        eq=self.config.initial_balance+self.realized
        self.peak=max(self.peak,eq)
        self.current_dd_pct=(self.peak-eq)/max(self.peak,1e-9)*100.0
        self.max_dd=max(self.max_dd,self.current_dd_pct)
        return self.current_dd_pct

    def _update_regime(self):
        if len(self.closes)<60 or not self.active:
            self.regime_score=0.0
            return
        c=np.asarray(self.closes,float)
        ema20=float(c[-20:].mean()); ema60=float(c[-60:].mean())
        atr=max(self._atr(),1e-9)
        self.regime_score=((ema20-ema60)/atr)*self.side

    def on_bar(self, bar):
        super().on_bar(bar)
        self._update_regime()
        self._equity_dd()

    def _close(self,bid,ask,reason):
        if self.active:
            self.exit_reasons[reason]+=1
        super()._close(bid,ask,reason)
        self._equity_dd()

    def on_quote_tick(self, tick):
        # Base handles entry + trailing + max-hold first.
        was_active=self.active
        super().on_quote_tick(tick)
        if was_active and not self.active:
            return
        if not self.active:
            return

        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price)
        mark=bid if self.side>0 else ask
        atr=max(self._atr(),1e-9)
        favorable=(mark-self.entry)*self.side
        mfe=((self.best-self.entry)*self.side) if self.side>0 else ((self.entry-self.best))
        mfe=max(0.0,mfe)
        hold_bars=max(0,self.bar_i-self.entry_bar)
        dd=self._equity_dd()

        threshold=self.hard_stop_atr
        state='NORMAL'

        # No follow-through: breakout failed to produce enough MFE after N bars.
        if hold_bars>=self.stale_after_bars and mfe < self.min_mfe_atr*atr:
            threshold=min(threshold,self.stale_stop_atr)
            state='STALE'

        # Directional structure has weakened or flipped relative to the trade.
        if self.regime_score < self.regime_floor:
            threshold=min(threshold,self.regime_stop_atr)
            state='REGIME_WEAK' if state=='NORMAL' else state+'+REGIME_WEAK'

        # Drawdown governor: preserve the entry edge, reduce tail exposure.
        if dd>=self.dd_soft_pct:
            threshold=min(threshold,self.dd_stop_atr)
            state='DD_GOV' if state=='NORMAL' else state+'+DD_GOV'

        self.dynamic_threshold_last=threshold
        self.state_samples[state]+=1

        if favorable <= -threshold*atr:
            self._close(bid,ask,'S3_'+state)

    def summary(self):
        x=super().summary()
        x.update({
            'stage3':True,
            'hard_stop_atr':self.hard_stop_atr,
            'stale_stop_atr':self.stale_stop_atr,
            'stale_after_bars':self.stale_after_bars,
            'min_mfe_atr':self.min_mfe_atr,
            'regime_stop_atr':self.regime_stop_atr,
            'regime_floor':self.regime_floor,
            'dd_soft_pct':self.dd_soft_pct,
            'dd_stop_atr':self.dd_stop_atr,
            'state_exit_counts':dict(self.exit_reasons),
            'state_tick_counts':dict(self.state_samples),
            'controller_max_realized_dd_pct':self.max_dd,
        })
        return x


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',required=True)
    ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--variant',required=True)
    ap.add_argument('--tf',default='M15',choices=list(TF_MIN))
    ap.add_argument('--family',default='R2',choices=['R1','R2','R3','R4'])
    ap.add_argument('--trail-atr',type=float,default=1.20)
    ap.add_argument('--protect-atr',type=float,default=0.60)
    ap.add_argument('--max-hold-minutes',type=int,default=480)
    ap.add_argument('--hard-stop-atr',type=float,default=2.40)
    ap.add_argument('--stale-stop-atr',type=float,default=1.45)
    ap.add_argument('--stale-after-bars',type=int,default=2)
    ap.add_argument('--min-mfe-atr',type=float,default=0.35)
    ap.add_argument('--regime-stop-atr',type=float,default=1.60)
    ap.add_argument('--regime-floor',type=float,default=0.0)
    ap.add_argument('--dd-soft-pct',type=float,default=3.5)
    ap.add_argument('--dd-stop-atr',type=float,default=1.35)
    ap.add_argument('--raw-bidask-only',action='store_true')
    a=ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit('raw-bidask-only mandatory')

    catalog=ParquetDataCatalog(str(Path(a.catalog)))
    instrument=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if instrument is None: raise SystemExit('XAUUSD missing')
    if hasattr(catalog,'query_quote_ticks'):
        raw_ticks=catalog.query_quote_ticks(identifiers=[instrument.id.value])
    else:
        raw_ticks=catalog.query(data_cls=QuoteTick,identifiers=[instrument.id.value])
    if not raw_ticks: raise SystemExit('no raw XAUUSD QuoteTicks')
    ticks,replaced=_positive_size_ticks(raw_ticks)

    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    exec_venue=instrument.id.venue
    engine.add_venue(venue=exec_venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(instrument); engine.add_data(ticks)
    bt=BarType.from_str(f'{instrument.id.value}-{TF_MIN[a.tf]}-MINUTE-BID-INTERNAL')
    cfg=ArmadaConfig(instrument_id=instrument.id,bar_type=bt,family=a.family,max_hold_minutes=a.max_hold_minutes,trail_atr=a.trail_atr,protect_atr=a.protect_atr)
    st=ArmadaApexStage3(
        cfg,
        hard_stop_atr=a.hard_stop_atr,
        stale_stop_atr=a.stale_stop_atr,
        stale_after_bars=a.stale_after_bars,
        min_mfe_atr=a.min_mfe_atr,
        regime_stop_atr=a.regime_stop_atr,
        regime_floor=a.regime_floor,
        dd_soft_pct=a.dd_soft_pct,
        dd_stop_atr=a.dd_stop_atr,
    )
    engine.add_strategy(st); engine.run()

    pos=engine.trader.generate_positions_report(); fills=engine.trader.generate_order_fills_report(); orders=engine.trader.generate_orders_report()
    s=st.summary()
    obj={
      'verification_level':'NAUTILUS_BT_RAW_BIDASK_ARMADA_APEX_STAGE3_STATE_NATIVE',
      'variant':a.variant,'engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
      'raw_ticks':len(raw_ticks),'execution_ticks':len(ticks),'raw_zero_size_quotes_replaced':replaced,'ohlc_resample_used':False,
      'tf':a.tf,'family':a.family,
      'params':{
        'trail_atr':a.trail_atr,'protect_atr':a.protect_atr,'max_hold_minutes':a.max_hold_minutes,
        'hard_stop_atr':a.hard_stop_atr,'stale_stop_atr':a.stale_stop_atr,'stale_after_bars':a.stale_after_bars,
        'min_mfe_atr':a.min_mfe_atr,'regime_stop_atr':a.regime_stop_atr,'regime_floor':a.regime_floor,
        'dd_soft_pct':a.dd_soft_pct,'dd_stop_atr':a.dd_stop_atr,
      },
      **s,
      'native_orders':int(len(orders)) if orders is not None else 0,'native_order_status_counts':_status_counts(orders),
      'native_positions':int(len(pos)) if pos is not None else 0,'native_fills':int(len(fills)) if fills is not None else 0,
      'native_fill_gate_pass':bool(fills is not None and len(fills)>0),
      'execution_size_policy':'Preserve raw bid/ask prices/timestamps; replace nonpositive L1 sizes with Quantity(1) solely for native matching.',
      'first_raw_quote':_tick_diag(raw_ticks[0]),'first_execution_quote':_tick_diag(ticks[0]),
      'disclaimer':'Clean-room APEX Stage3 state-controller hypothesis; not original Armada source.'
    }
    out=Path('results/armada-apex')/a.experiment_id/'cells'; out.mkdir(parents=True,exist_ok=True)
    path=out/f'{a.variant}.json'; path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
    print(json.dumps(obj,indent=2,default=str)); engine.dispose()

if __name__=='__main__': main()
