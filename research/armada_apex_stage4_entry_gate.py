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
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from armada_nautilus_raw_cell import ArmadaCandidate, ArmadaConfig, TF_MIN, _status_counts, _tick_diag, _ensure_l1_liquidity


class ArmadaEntryGateCandidate(ArmadaCandidate):
    """E2 exit preserved; selectively veto low-quality R2 breakouts before entry."""
    def __init__(self, config, min_trend_strength=0.0, min_breakout_atr=0.0,
                 min_range_expansion=0.0, max_breakout_atr=999.0):
        super().__init__(config)
        self.min_trend_strength=float(min_trend_strength)
        self.min_breakout_atr=float(min_breakout_atr)
        self.min_range_expansion=float(min_range_expansion)
        self.max_breakout_atr=float(max_breakout_atr)
        self.raw_signals=0
        self.filtered_signals=0
        self.gate_reasons=Counter()

    def _signal(self):
        sig=super()._signal()
        if sig == 0:
            return 0
        self.raw_signals += 1
        if len(self.closes) < 65 or len(self.trs) < 2:
            self.filtered_signals += 1
            self.gate_reasons['warmup'] += 1
            return 0
        c=np.asarray(self.closes,float); h=np.asarray(self.highs,float); l=np.asarray(self.lows,float)
        atr=max(self._atr(),1e-9)
        ema20=float(c[-20:].mean()); ema60=float(c[-60:].mean())
        trend_strength=abs(ema20-ema60)/atr
        hh=float(h[-21:-1].max()); ll=float(l[-21:-1].min())
        impulse=((c[-1]-hh)/atr) if sig>0 else ((ll-c[-1])/atr)
        range_expansion=float(self.trs[-1])/atr
        reasons=[]
        if trend_strength < self.min_trend_strength: reasons.append('trend')
        if impulse < self.min_breakout_atr: reasons.append('impulse_low')
        if impulse > self.max_breakout_atr: reasons.append('impulse_high')
        if range_expansion < self.min_range_expansion: reasons.append('range')
        if reasons:
            self.filtered_signals += 1
            for r in reasons: self.gate_reasons[r] += 1
            return 0
        return sig

    def summary(self):
        x=super().summary()
        x.update({
            'stage4': True,
            'raw_signals': self.raw_signals,
            'filtered_signals': self.filtered_signals,
            'accepted_signals': self.raw_signals-self.filtered_signals,
            'gate_reasons': dict(self.gate_reasons),
            'entry_gate': {
                'min_trend_strength':self.min_trend_strength,
                'min_breakout_atr':self.min_breakout_atr,
                'min_range_expansion':self.min_range_expansion,
                'max_breakout_atr':self.max_breakout_atr,
            },
        })
        return x


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--variant',required=True)
    ap.add_argument('--tf',default='M15',choices=list(TF_MIN)); ap.add_argument('--family',default='R2',choices=['R2'])
    ap.add_argument('--min-trend-strength',type=float,default=0.0)
    ap.add_argument('--min-breakout-atr',type=float,default=0.0)
    ap.add_argument('--min-range-expansion',type=float,default=0.0)
    ap.add_argument('--max-breakout-atr',type=float,default=999.0)
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
    ticks,replaced=_ensure_l1_liquidity(raw_ticks)

    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=instrument.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
                     base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(instrument); engine.add_data(ticks)
    bt=BarType.from_str(f'{instrument.id.value}-{TF_MIN[a.tf]}-MINUTE-BID-INTERNAL')
    # Exact Stage1 E2 exit baseline: trail=1.00, protect=0.50, hold=720, no adverse stop.
    cfg=ArmadaConfig(instrument_id=instrument.id,bar_type=bt,family='R2',max_hold_minutes=720,trail_atr=1.00,protect_atr=0.50)
    st=ArmadaEntryGateCandidate(cfg,a.min_trend_strength,a.min_breakout_atr,a.min_range_expansion,a.max_breakout_atr)
    engine.add_strategy(st); engine.run()

    pos=engine.trader.generate_positions_report(); fills=engine.trader.generate_order_fills_report(); orders=engine.trader.generate_orders_report()
    obj={
        'verification_level':'NAUTILUS_BT_RAW_BIDASK_ARMADA_APEX_STAGE4_ENTRY_GATE_NATIVE',
        'variant':a.variant,'engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'raw_ticks':len(raw_ticks),'execution_ticks':len(ticks),'raw_zero_size_quotes_replaced':replaced,'ohlc_resample_used':False,
        'tf':a.tf,'family':'R2','base_exit':'E2: trail=1.00 protect=0.50 hold=720 stop=none',**st.summary(),
        'native_orders':int(len(orders)) if orders is not None else 0,'native_order_status_counts':_status_counts(orders),
        'native_positions':int(len(pos)) if pos is not None else 0,'native_fills':int(len(fills)) if fills is not None else 0,
        'native_fill_gate_pass':bool(fills is not None and len(fills)>0),
        'execution_size_policy':'Preserve raw bid/ask prices/timestamps; replace nonpositive L1 sizes with Quantity(1) solely for native matching.',
        'first_raw_quote':_tick_diag(raw_ticks[0]),'first_execution_quote':_tick_diag(ticks[0]),
        'disclaimer':'Clean-room APEX Stage4 entry-gate hypothesis; not original Armada source.'
    }
    out=Path('results/armada-apex')/a.experiment_id/'cells'; out.mkdir(parents=True,exist_ok=True)
    (out/f'{a.variant}.json').write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
    print(json.dumps(obj,indent=2,default=str)); engine.dispose()

if __name__=='__main__': main()
