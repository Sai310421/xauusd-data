from __future__ import annotations

import argparse, json, math
from decimal import Decimal
from pathlib import Path

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


class ArmadaApexCandidate(ArmadaCandidate):
    def __init__(self, config, adverse_stop_atr: float=0.0, min_atr: float=0.0):
        super().__init__(config)
        self.adverse_stop_atr=float(adverse_stop_atr)
        self.min_atr=float(min_atr)
        self.apex_stops=0
        self.filtered_signals=0

    def on_bar(self, bar):
        # Preserve the clean-room signal logic, but allow a volatility floor to veto only new signals.
        before_pending=self.pending
        super().on_bar(bar)
        if before_pending==0 and self.pending!=0 and self.min_atr>0 and self._atr()<self.min_atr:
            self.pending=0
            self.filtered_signals += 1

    def on_quote_tick(self, tick):
        super().on_quote_tick(tick)
        if not self.active or self.adverse_stop_atr<=0:
            return
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price)
        mark=bid if self.side>0 else ask
        atr=max(self._atr(),1e-9)
        favorable=(mark-self.entry)*self.side
        if favorable <= -self.adverse_stop_atr*atr:
            self.apex_stops += 1
            self._close(bid,ask,'APEX_ATR_STOP')

    def summary(self):
        x=super().summary()
        x.update({'adverse_stop_atr':self.adverse_stop_atr,'min_atr':self.min_atr,'apex_stops':self.apex_stops,'filtered_signals':self.filtered_signals})
        return x


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',required=True)
    ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--variant',required=True)
    ap.add_argument('--tf',default='M15',choices=list(TF_MIN))
    ap.add_argument('--family',default='R2',choices=['R1','R2','R3','R4'])
    ap.add_argument('--trail-atr',type=float,default=1.5)
    ap.add_argument('--protect-atr',type=float,default=0.8)
    ap.add_argument('--max-hold-minutes',type=int,default=720)
    ap.add_argument('--adverse-stop-atr',type=float,default=0.0)
    ap.add_argument('--min-atr',type=float,default=0.0)
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
    st=ArmadaApexCandidate(cfg,adverse_stop_atr=a.adverse_stop_atr,min_atr=a.min_atr)
    engine.add_strategy(st); engine.run()

    pos=engine.trader.generate_positions_report(); fills=engine.trader.generate_order_fills_report(); orders=engine.trader.generate_orders_report()
    s=st.summary()
    obj={
      'verification_level':'NAUTILUS_BT_RAW_BIDASK_ARMADA_APEX_NATIVE',
      'variant':a.variant,'engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
      'raw_ticks':len(raw_ticks),'execution_ticks':len(ticks),'raw_zero_size_quotes_replaced':replaced,'ohlc_resample_used':False,
      'tf':a.tf,'family':a.family,'params':{'trail_atr':a.trail_atr,'protect_atr':a.protect_atr,'max_hold_minutes':a.max_hold_minutes,'adverse_stop_atr':a.adverse_stop_atr,'min_atr':a.min_atr},
      **s,
      'native_orders':int(len(orders)) if orders is not None else 0,'native_order_status_counts':_status_counts(orders),
      'native_positions':int(len(pos)) if pos is not None else 0,'native_fills':int(len(fills)) if fills is not None else 0,
      'native_fill_gate_pass':bool(fills is not None and len(fills)>0),
      'execution_size_policy':'Preserve raw bid/ask prices/timestamps; replace nonpositive L1 sizes with Quantity(1) solely for native matching.',
      'first_raw_quote':_tick_diag(raw_ticks[0]),'first_execution_quote':_tick_diag(ticks[0]),
      'disclaimer':'Clean-room APEX hypothesis; not original Armada source.'
    }
    out=Path('results/armada-apex')/a.experiment_id/'cells'; out.mkdir(parents=True,exist_ok=True)
    path=out/f'{a.variant}.json'; path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
    print(json.dumps(obj,indent=2,default=str)); engine.dispose()

if __name__=='__main__': main()
