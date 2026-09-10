from __future__ import annotations

import argparse, json, math
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


class FingerprintCandidate(ArmadaCandidate):
    """E2 baseline with entry-feature capture only. No filtering: diagnostic run."""
    def __init__(self, config):
        super().__init__(config)
        self.pending_feature = None
        self.active_feature = None

    def _signal(self):
        sig = super()._signal()
        if sig == 0:
            return 0
        c=np.asarray(self.closes,float); h=np.asarray(self.highs,float); l=np.asarray(self.lows,float)
        atr=max(self._atr(),1e-9)
        ema20=float(c[-20:].mean()); ema60=float(c[-60:].mean())
        slope=(ema20-ema60)/atr
        trend_strength=abs(slope)
        hh=float(h[-21:-1].max()); ll=float(l[-21:-1].min())
        impulse=((c[-1]-hh)/atr) if sig>0 else ((ll-c[-1])/atr)
        range_expansion=float(self.trs[-1])/atr
        close_pos=(c[-1]-l[-1])/max(h[-1]-l[-1],1e-9)
        directional_close_pos=close_pos if sig>0 else 1.0-close_pos
        body=abs(c[-1]-float(c[-2]))/atr
        self.pending_feature={
            'trend_strength':trend_strength,
            'signed_slope':slope*sig,
            'breakout_atr':impulse,
            'range_expansion':range_expansion,
            'directional_close_pos':directional_close_pos,
            'body_atr':body,
            'atr':atr,
            'side':sig,
        }
        return sig

    def _open(self,side,bid,ask):
        super()._open(side,bid,ask)
        if self.active:
            self.active_feature=dict(self.pending_feature or {})
        self.pending_feature=None

    def _close(self,bid,ask,reason):
        if not self.active:
            return
        feat=dict(self.active_feature or {})
        super()._close(bid,ask,reason)
        if self.trades:
            self.trades[-1].update(feat)
        self.active_feature=None


def summarize(trades):
    n=len(trades)
    wins=[t for t in trades if t['pnl']>0]
    losses=[t for t in trades if t['pnl']<0]
    gw=sum(t['pnl'] for t in wins); gl=-sum(t['pnl'] for t in losses)
    eq=1000.0; peak=eq; dd=0.0
    for t in trades:
        eq+=t['pnl']; peak=max(peak,eq); dd=max(dd,(peak-eq)/max(peak,1e-9)*100)
    return {'N':n,'WR_pct':100*len(wins)/max(n,1),'PF':gw/gl if gl>0 else (math.inf if gw>0 else 0.0),
            'expectancy':sum(t['pnl'] for t in trades)/max(n,1),'max_DD_pct':dd,'net':sum(t['pnl'] for t in trades)}


def dist(rows,key):
    x=np.asarray([r.get(key,np.nan) for r in rows],float); x=x[np.isfinite(x)]
    if len(x)==0:return {}
    return {'n':int(len(x)),'mean':float(x.mean()),'p10':float(np.quantile(x,.10)),'p25':float(np.quantile(x,.25)),
            'p50':float(np.quantile(x,.50)),'p75':float(np.quantile(x,.75)),'p90':float(np.quantile(x,.90))}


def scan(trades):
    """Counterfactual one-feature vetoes on the exact realized baseline sequence."""
    specs={
      'trend_strength': [('min',x) for x in [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50]],
      'signed_slope': [('min',x) for x in [-0.10,0.00,0.05,0.10,0.15,0.20,0.30]],
      'breakout_atr': [('min',x) for x in [0.01,0.03,0.05,0.08,0.10,0.15]],
      'range_expansion': [('min',x) for x in [0.7,0.8,0.9,1.0,1.1,1.2,1.3]],
      'directional_close_pos': [('min',x) for x in [0.50,0.60,0.70,0.80,0.90]],
      'body_atr': [('min',x) for x in [0.05,0.10,0.15,0.20,0.30,0.40]],
    }
    out=[]
    for key,tests in specs.items():
      for mode,thr in tests:
        kept=[t for t in trades if float(t.get(key,-1e99))>=thr]
        s=summarize(kept); s.update({'feature':key,'mode':mode,'threshold':thr,'retention_pct':100*len(kept)/max(len(trades),1)})
        out.append(s)
      # also remove extreme high tail for features where overextension may hurt
      vals=[0.20,0.30,0.40,0.50,0.75,1.00] if key=='breakout_atr' else []
      for thr in vals:
        kept=[t for t in trades if float(t.get(key,1e99))<=thr]
        s=summarize(kept); s.update({'feature':key,'mode':'max','threshold':thr,'retention_pct':100*len(kept)/max(len(trades),1)})
        out.append(s)
    out.sort(key=lambda z: (z['PF'], -z['max_DD_pct'], z['N']), reverse=True)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--raw-bidask-only',action='store_true')
    a=ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit('raw-bidask-only mandatory')
    catalog=ParquetDataCatalog(str(Path(a.catalog)))
    instrument=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if instrument is None: raise SystemExit('XAUUSD missing')
    raw_ticks=catalog.query_quote_ticks(identifiers=[instrument.id.value]) if hasattr(catalog,'query_quote_ticks') else catalog.query(data_cls=QuoteTick,identifiers=[instrument.id.value])
    if not raw_ticks: raise SystemExit('no raw XAUUSD QuoteTicks')
    ticks,replaced=_ensure_l1_liquidity(raw_ticks)
    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=instrument.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
                     base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(instrument); engine.add_data(ticks)
    bt=BarType.from_str(f'{instrument.id.value}-15-MINUTE-BID-INTERNAL')
    cfg=ArmadaConfig(instrument_id=instrument.id,bar_type=bt,family='R2',max_hold_minutes=720,trail_atr=1.00,protect_atr=0.50)
    st=FingerprintCandidate(cfg); engine.add_strategy(st); engine.run()
    orders=engine.trader.generate_orders_report(); fills=engine.trader.generate_order_fills_report(); pos=engine.trader.generate_positions_report()
    keys=['trend_strength','signed_slope','breakout_atr','range_expansion','directional_close_pos','body_atr','mae','mfe','hold_bars']
    wins=[t for t in st.trades if t['pnl']>0]; losses=[t for t in st.trades if t['pnl']<0]
    fingerprints={k:{'win':dist(wins,k),'loss':dist(losses,k)} for k in keys}
    candidates=scan(st.trades)
    eligible=[x for x in candidates if x['N']>=80 and x['PF']>=1.5]
    obj={
      'verification_level':'NAUTILUS_BT_RAW_BIDASK_ARMADA_APEX_STAGE5_LOSER_FINGERPRINT_NATIVE',
      'engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
      'raw_ticks':len(raw_ticks),'raw_zero_size_quotes_replaced':replaced,'ohlc_resample_used':False,
      'baseline':summarize(st.trades),'trade_count':len(st.trades),'fingerprints':fingerprints,
      'top_counterfactual_gates':candidates[:20],'eligible_pf15_n80':eligible[:20],
      'native_orders':int(len(orders)) if orders is not None else 0,'native_order_status_counts':_status_counts(orders),
      'native_positions':int(len(pos)) if pos is not None else 0,'native_fills':int(len(fills)) if fills is not None else 0,
      'native_fill_gate_pass':bool(fills is not None and len(fills)>0),
      'first_raw_quote':_tick_diag(raw_ticks[0]),'first_execution_quote':_tick_diag(ticks[0]),
      'disclaimer':'Diagnostic clean-room loser-fingerprint analysis on E2 baseline; counterfactual gates are in-sample diagnostics, not final OOS evidence.'
    }
    out=Path('results/armada-apex')/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    (out/'stage5_loser_fingerprint.json').write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
    # compact CSV for later controller use
    import csv
    with (out/'stage5_trade_features.csv').open('w',newline='',encoding='utf-8') as f:
      fields=['pnl','side','hold_bars','mfe','mae','reason','trend_strength','signed_slope','breakout_atr','range_expansion','directional_close_pos','body_atr','atr']
      w=csv.DictWriter(f,fieldnames=fields); w.writeheader();
      for t in st.trades:w.writerow({k:t.get(k) for k in fields})
    print(json.dumps(obj,indent=2,default=str)); engine.dispose()

if __name__=='__main__': main()
