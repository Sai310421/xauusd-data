from __future__ import annotations

import argparse, json, math
from collections import deque, Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, BookType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

if not hasattr(ParquetDataCatalog, 'query_quote_ticks'):
    def _query_quote_ticks(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks

TF_MIN = {'M1':1,'M5':5,'M15':15}

class ArmadaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    family: str
    initial_balance: float = 1000.0
    unit_qty: Decimal = Decimal('1')
    max_hold_minutes: int = 720
    trail_atr: float = 1.50
    protect_atr: float = 0.80

class ArmadaCandidate(Strategy):
    def __init__(self, config: ArmadaConfig):
        super().__init__(config)
        self.closes=deque(maxlen=120); self.highs=deque(maxlen=120); self.lows=deque(maxlen=120)
        self.trs=deque(maxlen=60); self.prev_close=None
        self.active=False; self.side=0; self.entry=None; self.entry_tick=0; self.entry_bar=0
        self.best=None; self.worst=None; self.trail_stop=None; self.pending=0
        self.tick_i=0; self.bar_i=0; self.last_bid=None; self.last_ask=None
        self.realized=0.0; self.peak=config.initial_balance; self.max_dd=0.0
        self.gw=0.0; self.gl=0.0; self.wins=0; self.losses=0; self.trades=[]
        self.submitted_orders=0
        self.order_events=Counter()
        self.last_order_problem=None

    @staticmethod
    def _f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def on_order_submitted(self, event): self.order_events['submitted'] += 1
    def on_order_accepted(self, event): self.order_events['accepted'] += 1
    def on_order_filled(self, event): self.order_events['filled'] += 1
    def on_order_denied(self, event):
        self.order_events['denied'] += 1
        self.last_order_problem = str(event)
    def on_order_rejected(self, event):
        self.order_events['rejected'] += 1
        self.last_order_problem = str(event)
    def on_order_canceled(self, event): self.order_events['canceled'] += 1

    def _atr(self):
        return float(np.mean(self.trs)) if self.trs else 0.0

    def _signal(self):
        n=len(self.closes)
        if n < 65: return 0
        c=np.asarray(self.closes,float); h=np.asarray(self.highs,float); l=np.asarray(self.lows,float)
        ema20=float(c[-20:].mean()); ema60=float(c[-60:].mean())
        slope=ema20-ema60
        atr=max(self._atr(),1e-9)
        trend_strength=abs(slope)/atr
        fam=self.config.family
        if fam=='R1':
            if slope>0 and c[-1]>ema20 and c[-2]<=ema20: return 1
            if slope<0 and c[-1]<ema20 and c[-2]>=ema20: return -1
            return 0
        if fam=='R2':
            hh=float(h[-21:-1].max()); ll=float(l[-21:-1].min())
            if c[-1]>hh: return 1
            if c[-1]<ll: return -1
            return 0
        if fam=='R3':
            m=float(c[-40:].mean()); s=float(c[-40:].std(ddof=0))
            if s<=0:return 0
            z=(c[-1]-m)/s
            if z<=-1.8:return 1
            if z>=1.8:return -1
            return 0
        if trend_strength>=0.8:
            hh=float(h[-21:-1].max()); ll=float(l[-21:-1].min())
            if c[-1]>hh or (slope>0 and c[-1]>ema20 and c[-2]<=ema20): return 1
            if c[-1]<ll or (slope<0 and c[-1]<ema20 and c[-2]>=ema20): return -1
            return 0
        m=float(c[-40:].mean()); s=float(c[-40:].std(ddof=0))
        if s>0:
            z=(c[-1]-m)/s
            if z<=-2.0:return 1
            if z>=2.0:return -1
        return 0

    def _submit(self, side:int):
        inst=self.cache.instrument(self.config.instrument_id)
        qty=inst.make_qty(self.config.unit_qty)
        o=self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if side>0 else OrderSide.SELL,
            quantity=qty,
        )
        self.submitted_orders += 1
        self.submit_order(o)

    def _open(self,side:int,bid:float,ask:float):
        if self.active:return
        px=ask if side>0 else bid
        self._submit(side); self.active=True; self.side=side; self.entry=px
        self.entry_tick=self.tick_i; self.entry_bar=self.bar_i; self.best=px; self.worst=px; self.trail_stop=None

    def _close(self,bid:float,ask:float,reason:str):
        if not self.active:return
        px=bid if self.side>0 else ask
        pnl=(px-self.entry)*self.side
        self._submit(-self.side)
        hold_bars=max(0,self.bar_i-self.entry_bar)
        mfe=(self.best-self.entry)*self.side if self.side>0 else (self.entry-self.best)
        mae=(self.entry-self.worst) if self.side>0 else (self.worst-self.entry)
        self.trades.append({'pnl':pnl,'hold_bars':hold_bars,'mfe':max(0.0,mfe),'mae':max(0.0,mae),'reason':reason,'side':self.side})
        self.realized+=pnl
        if pnl>0:self.wins+=1;self.gw+=pnl
        elif pnl<0:self.losses+=1;self.gl+=abs(pnl)
        self.active=False; self.side=0; self.entry=None; self.trail_stop=None; self.best=None; self.worst=None

    def on_bar(self,bar:Bar):
        self.bar_i+=1
        o,h,l,c=map(self._f,[bar.open,bar.high,bar.low,bar.close])
        tr=max(h-l,abs(h-self.prev_close) if self.prev_close is not None else 0,abs(l-self.prev_close) if self.prev_close is not None else 0)
        self.trs.append(tr); self.prev_close=c; self.closes.append(c); self.highs.append(h); self.lows.append(l)
        if not self.active and self.pending==0:self.pending=self._signal()

    def on_quote_tick(self,tick:QuoteTick):
        self.tick_i+=1; bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); self.last_bid=bid; self.last_ask=ask
        if self.pending and not self.active:
            s=self.pending; self.pending=0; self._open(s,bid,ask)
        if not self.active:return
        mark=bid if self.side>0 else ask
        if self.side>0:
            self.best=max(self.best,mark); self.worst=min(self.worst,mark)
        else:
            self.best=min(self.best,mark); self.worst=max(self.worst,mark)
        atr=max(self._atr(),1e-9)
        favorable=(mark-self.entry)*self.side
        if favorable>=self.config.protect_atr*atr:
            candidate=(self.best-self.config.trail_atr*atr) if self.side>0 else (self.best+self.config.trail_atr*atr)
            self.trail_stop=candidate if self.trail_stop is None else (max(self.trail_stop,candidate) if self.side>0 else min(self.trail_stop,candidate))
        if self.trail_stop is not None:
            hit=mark<=self.trail_stop if self.side>0 else mark>=self.trail_stop
            if hit:self._close(bid,ask,'TRAIL');return
        tf_minutes=TF_MIN.get(next((name for name,m in TF_MIN.items() if f'-{m}-MINUTE-' in str(self.config.bar_type)), 'M1'),1)
        if (self.bar_i-self.entry_bar)*tf_minutes>=self.config.max_hold_minutes:
            self._close(bid,ask,'TIME')

    def on_stop(self):
        if self.active and self.last_bid is not None:self._close(self.last_bid,self.last_ask,'EOD')

    def summary(self):
        n=len(self.trades); pf=self.gw/self.gl if self.gl>0 else (math.inf if self.gw>0 else 0.0)
        wr=self.wins/max(n,1)*100.0
        eq=self.config.initial_balance; peak=eq; mdd=0.0
        for t in self.trades:
            eq+=t['pnl']; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/max(peak,1e-9)*100)
        hs=[t['hold_bars'] for t in self.trades]; maes=[t['mae'] for t in self.trades]; mfes=[t['mfe'] for t in self.trades]
        return {
            'family':self.config.family,'N':n,'WR_pct':wr,'PF':pf,'net_virtual':self.realized,'max_DD_pct':mdd,
            'avg_win':self.gw/max(self.wins,1),'avg_loss':self.gl/max(self.losses,1),'expectancy':self.realized/max(n,1),
            'hold_bars_mean':float(np.mean(hs)) if hs else 0.0,'MAE_mean':float(np.mean(maes)) if maes else 0.0,'MFE_mean':float(np.mean(mfes)) if mfes else 0.0,
            'submitted_orders':self.submitted_orders,'order_events':dict(self.order_events),'last_order_problem':self.last_order_problem,
        }

def _status_counts(df):
    if df is None or len(df)==0:
        return {}
    for col in ('status','order_status'):
        if col in df.columns:
            return {str(k):int(v) for k,v in df[col].astype(str).value_counts().to_dict().items()}
    return {'columns': [str(c) for c in df.columns]}

def _tick_diag(tick):
    def val(name):
        x=getattr(tick,name,None)
        if x is None:return None
        if hasattr(x,'as_double'):return float(x.as_double())
        try:return float(x)
        except Exception:return str(x)
    return {
        'bid_price':val('bid_price'),'ask_price':val('ask_price'),
        'bid_size':val('bid_size'),'ask_size':val('ask_size'),
        'instrument_id':str(getattr(tick,'instrument_id','')),
    }

def _ensure_l1_liquidity(ticks):
    """Dukascopy cache currently carries price-valid QuoteTicks with 0 bid/ask size.
    Nautilus L1 matching correctly treats zero size as 'no market'. For execution
    validation only, preserve every raw bid/ask price and timestamp while replacing
    zero/negative sizes with one executable unit. This is not fabricated price data.
    """
    one=Quantity.from_int(1)
    out=[]; replaced=0
    for t in ticks:
        bs=t.bid_size; aqs=t.ask_size
        bsv=float(bs.as_double()) if hasattr(bs,'as_double') else float(bs)
        asv=float(aqs.as_double()) if hasattr(aqs,'as_double') else float(aqs)
        if bsv<=0.0 or asv<=0.0:
            out.append(QuoteTick(
                instrument_id=t.instrument_id,
                bid_price=t.bid_price,
                ask_price=t.ask_price,
                bid_size=one if bsv<=0.0 else bs,
                ask_size=one if asv<=0.0 else aqs,
                ts_event=t.ts_event,
                ts_init=t.ts_init,
            ))
            replaced+=1
        else:
            out.append(t)
    return out,replaced

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--tf',choices=list(TF_MIN),required=True); ap.add_argument('--family',choices=['R1','R2','R3','R4'],required=True); ap.add_argument('--raw-bidask-only',action='store_true')
    a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(a.catalog); catalog=ParquetDataCatalog(str(cp))
    instrument=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if instrument is None:raise SystemExit('XAUUSD missing')
    raw_ticks=catalog.query_quote_ticks(identifiers=[instrument.id.value])
    if not raw_ticks:raise SystemExit('no raw XAUUSD QuoteTicks')
    first_raw_tick=_tick_diag(raw_ticks[0]); last_raw_tick=_tick_diag(raw_ticks[-1])
    ticks,size_replacements=_ensure_l1_liquidity(raw_ticks)
    first_exec_tick=_tick_diag(ticks[0]); last_exec_tick=_tick_diag(ticks[-1])
    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    exec_venue=instrument.id.venue
    engine.add_venue(
        venue=exec_venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        book_type=BookType.L1_MBP,
        base_currency=USD,
        starting_balances=[Money(1000,USD)],
        default_leverage=Decimal('2000'),
    )
    engine.add_instrument(instrument); engine.add_data(ticks)
    bt=BarType.from_str(f'{instrument.id.value}-{TF_MIN[a.tf]}-MINUTE-BID-INTERNAL')
    st=ArmadaCandidate(ArmadaConfig(instrument_id=instrument.id,bar_type=bt,family=a.family))
    engine.add_strategy(st); engine.run()
    pos=engine.trader.generate_positions_report(); fills=engine.trader.generate_order_fills_report(); orders=engine.trader.generate_orders_report()
    obj={
        'verification_level':'NAUTILUS_BT_RAW_BIDASK_ARMADA_CLEANROOM_L1_SIZEFIX',
        'engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'raw_ticks':len(raw_ticks),'execution_ticks':len(ticks),'ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL from raw QuoteTicks','tf':a.tf,**st.summary(),
        'native_orders':int(len(orders)) if orders is not None else 0,
        'native_order_status_counts':_status_counts(orders),
        'native_positions':int(len(pos)) if pos is not None else 0,
        'native_fills':int(len(fills)) if fills is not None else 0,
        'execution_venue':str(exec_venue),
        'book_type':'L1_MBP',
        'instrument_id':str(instrument.id),
        'unit_qty':str(st.config.unit_qty),
        'instrument_size_precision':getattr(instrument,'size_precision',None),
        'raw_zero_size_quotes_replaced':size_replacements,
        'execution_size_policy':'Preserve raw bid/ask prices and timestamps; replace nonpositive L1 sizes with Quantity(1) solely to permit native matching.',
        'first_raw_quote':first_raw_tick,
        'last_raw_quote':last_raw_tick,
        'first_execution_quote':first_exec_tick,
        'last_execution_quote':last_exec_tick,
        'native_fill_gate_pass':bool(fills is not None and len(fills)>0),
        'disclaimer':'Clean-room behavioral hypothesis, not original Armada source.',
    }
    out=Path('results/armada-nautilus')/a.experiment_id/'cells'; out.mkdir(parents=True,exist_ok=True); (out/f'{a.tf}_{a.family}.json').write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8'); print(json.dumps(obj,indent=2,default=str)); engine.dispose()

if __name__=='__main__':main()
