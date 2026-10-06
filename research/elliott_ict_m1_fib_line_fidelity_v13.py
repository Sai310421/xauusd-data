from __future__ import annotations
import argparse, json, math, os, hashlib, datetime
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

TF_MIN=(1,)
INITIAL=1000.0
LEVERAGE=2000.0
BASE_QTY=1.0
DEEP=0.804
TIMEOUT_NS=240*60*1_000_000_000
COMMISSION_RT_PER_LOT=7.0
CASHBACK_RT_PER_LOT=6.0

def ff(x):
    return float(x.as_double()) if hasattr(x,'as_double') else float(x)

@dataclass
class BarState:
    start_ns:int=0; o:float=0.0; h:float=0.0; l:float=0.0; c:float=0.0
    hist:list=field(default_factory=list)

@dataclass
class Setup:
    tf:int; direction:int; origin:float; end:float; atr:float; signal_ns:int
    touched:bool=False; touch_ns:int=0; invalid:bool=False; entered:bool=False
    pending_ns:int=0; mss_close:float=0.0
    sl:float=0.0; tp:float=0.0

@dataclass
class Leg:
    tf:int; direction:int; qty:float; entry:float; entry_ns:int; sl:float; tp:float; expiry_ns:int
    active:bool=True

class Cfg(StrategyConfig, frozen=True):
    instrument_id: object
    confirm_mode: str='fib-line'
    body_mult: float=1.6
    range_atr_mult: float=0.9

class M1FibLineFidelityV13(Strategy):
    def __init__(self,c):
        super().__init__(c)
        self.bars={m:BarState() for m in TF_MIN}
        self.setups={m:[] for m in TF_MIN}
        self.legs=[]; self.closed=[]
        self.stats={m:{'candidates':0,'touched':0,'invalid_before_entry':0,'mss':0,'entries':0,'buy_entries':0,'sell_entries':0} for m in TF_MIN}
        self.last_bid=None; self.last_ask=None; self.last_ns=0; self.submitted=0
        self.realized=0.0; self.peak_equity=INITIAL; self.max_dd=0.0; self.min_ml=math.inf; self.max_gross=0.0
        self.first_ns=None; self.last_seen_ns=None

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def _submit(self,d,qty):
        inst=self.cache.instrument(self.config.instrument_id)
        q=inst.make_qty(Decimal(str(qty)))
        o=self.order_factory.market(instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if d>0 else OrderSide.SELL,quantity=q)
        self.submit_order(o); self.submitted+=1

    def _bar_update(self,m,ts,px):
        span=m*60*1_000_000_000; bucket=(ts//span)*span; b=self.bars[m]
        if b.start_ns==0:
            b.start_ns=bucket; b.o=b.h=b.l=b.c=px; return False
        if bucket==b.start_ns:
            b.h=max(b.h,px); b.l=min(b.l,px); b.c=px; return False
        b.hist.append((b.start_ns,b.o,b.h,b.l,b.c))
        if len(b.hist)>512:b.hist=b.hist[-512:]
        b.start_ns=bucket; b.o=b.h=b.l=b.c=px
        return True

    def _atr(self,hist):
        if len(hist)<16:return 0.0
        tr=[]
        for i in range(1,len(hist)):
            _,o,h,l,c=hist[i]; pc=hist[i-1][4]
            tr.append(max(h-l,abs(h-pc),abs(l-pc)))
        # Wilder-style EWM alpha 1/14 over available history, matching the prior unified diagnostic.
        a=tr[0]
        for x in tr[1:]: a=a+(x-a)/14.0
        return float(a)

    def _detect(self,m):
        # M1 fidelity rule: preserve the original Wave1 -> Fib -> MSS -> entry sequence,
        # but read M1 structure from the line-chart Close series to suppress wick noise.
        hist=self.bars[m].hist
        if len(hist)<35:return
        i=len(hist)-1
        ts=hist[i][0]
        closes=[x[4] for x in hist]
        c=closes[i]; pc=closes[i-1]

        prev20_moves=[abs(closes[j]-closes[j-1]) for j in range(i-20,i)]
        avg_move=float(np.mean(prev20_moves))
        atr=self._atr(hist)
        if avg_move<=0 or atr<=0:return

        # Same displacement idea as before, now measured on the line-chart move.
        if abs(c-pc)<self.config.body_mult*avg_move:return
        if abs(c-pc)<self.config.range_atr_mult*atr:return

        prev3=closes[i-3:i]
        bull=c>pc and c>max(prev3)
        bear=c<pc and c<min(prev3)

        bull_origin=None; bear_origin=None
        # Same sweep/reclaim concept, with wick highs/lows replaced only by Close points.
        for off in range(2,8):
            reclaim_idx=len(hist)-off
            sweep_idx=reclaim_idx-1
            if sweep_idx<12:continue
            reclaim_close=closes[reclaim_idx]
            sweep_close=closes[sweep_idx]
            # IMPORTANT: the prior-12 reference must exclude the sweep point itself.
            # v12 accidentally included sweep_close inside 'older', making
            # sweep_close < min(older) / > max(older) impossible and producing N=0.
            older=closes[sweep_idx-12:sweep_idx]
            old_lo=min(older); old_hi=max(older)
            if bull_origin is None and sweep_close<old_lo and reclaim_close>old_lo:
                bull_origin=sweep_close
            if bear_origin is None and sweep_close>old_hi and reclaim_close<old_hi:
                bear_origin=sweep_close

        setup=None
        if bull and bull_origin is not None and c>bull_origin:
            setup=Setup(m,1,bull_origin,c,atr,ts)
        elif bear and bear_origin is not None and bear_origin>c:
            setup=Setup(m,-1,bear_origin,c,atr,ts)

        if setup:
            span=abs(setup.end-setup.origin)
            setup.sl=setup.origin-setup.direction*0.2*setup.atr
            setup.tp=setup.end+setup.direction*0.618*span
            self.setups[m].append(setup)
            self.stats[m]['candidates']+=1

    def _mss(self,m,s):
        # Original MSS idea retained; only candle wick reference is removed.
        # BUY: current line Close breaks the previous line swing high.
        # SELL: current line Close breaks the previous line swing low.
        hist=self.bars[m].hist
        if len(hist)<3:return False
        c0=hist[-1][4]
        c1=hist[-2][4]
        c2=hist[-3][4]
        pivot_hi=max(c1,c2)
        pivot_lo=min(c1,c2)
        return c0>pivot_hi if s.direction>0 else c0<pivot_lo

    def _after_close(self,m,ts):
        self._detect(m)
        line_close=self.bars[m].hist[-1][4] if self.bars[m].hist else None
        for s in self.setups[m]:
            if s.invalid or s.entered or s.pending_ns:continue
            if ts-s.signal_ns>TIMEOUT_NS:continue

            # Preserve Fib entry logic. The only change is wick-noise suppression:
            # the M1 line Close must actually enter the 0.786-0.822 retracement zone.
            span=abs(s.end-s.origin)
            z786=s.end-s.direction*span*0.786
            z822=s.end-s.direction*span*0.822
            zone_lo=min(z786,z822); zone_hi=max(z786,z822)
            if not s.touched and line_close is not None and zone_lo<=line_close<=zone_hi:
                s.touched=True
                s.touch_ns=ts
                self.stats[m]['touched']+=1

            if not s.touched:continue
            if self._mss(m,s):
                s.pending_ns=ts
                s.mss_close=line_close
                self.stats[m]['mss']+=1

    def _setup_tick(self,m,bid,ask,ts):
        keep=[]
        for s in self.setups[m]:
            if s.entered:continue
            if ts-s.signal_ns>TIMEOUT_NS:continue
            inv = bid<=s.sl if s.direction>0 else ask>=s.sl
            if inv:
                s.invalid=True; self.stats[m]['invalid_before_entry']+=1; continue
            if s.pending_ns and ts>=s.pending_ns:
                px=ask if s.direction>0 else bid
                # enforce logical stop relation at the actual fill
                if (s.direction>0 and px<=s.sl) or (s.direction<0 and px>=s.sl):
                    s.invalid=True; self.stats[m]['invalid_before_entry']+=1; continue
                self._submit(s.direction,BASE_QTY)
                self.legs.append(Leg(m,s.direction,BASE_QTY,px,ts,s.sl,s.tp,s.signal_ns+TIMEOUT_NS,True))
                s.entered=True; self.stats[m]['entries']+=1
                if s.direction>0:self.stats[m]['buy_entries']+=1
                else:self.stats[m]['sell_entries']+=1
                continue
            keep.append(s)
        self.setups[m]=keep

    def _close_leg(self,x,bid,ask,ts,reason):
        if not x.active:return
        px=bid if x.direction>0 else ask
        lots=x.qty/100.0
        gross=(px-x.entry)*x.direction*x.qty
        cost=lots*COMMISSION_RT_PER_LOT
        cb=lots*CASHBACK_RT_PER_LOT
        pnl=gross-cost+cb
        self._submit(-x.direction,x.qty); x.active=False; self.realized+=pnl
        self.closed.append({'tf':x.tf,'direction':x.direction,'qty':x.qty,'entry':x.entry,'exit':px,'gross':gross,
            'commission_rt':cost,'cashback_rt':cb,'pnl':pnl,'entry_ns':x.entry_ns,'exit_ns':ts,'reason':reason})

    def _manage_legs(self,bid,ask,ts):
        for x in list(self.legs):
            if not x.active:continue
            mark=bid if x.direction>0 else ask
            if (mark<=x.sl if x.direction>0 else mark>=x.sl):
                self._close_leg(x,bid,ask,ts,'SL')
            elif (mark>=x.tp if x.direction>0 else mark<=x.tp):
                self._close_leg(x,bid,ask,ts,'TP')
            elif ts>=x.expiry_ns:
                self._close_leg(x,bid,ask,ts,'TIMEOUT')

    def _risk(self,bid,ask):
        floating=0.0; gross=0.0
        for x in self.legs:
            if not x.active:continue
            mark=bid if x.direction>0 else ask
            floating+=(mark-x.entry)*x.direction*x.qty; gross+=x.qty
        eq=INITIAL+self.realized+floating; self.peak_equity=max(self.peak_equity,eq); self.max_dd=max(self.max_dd,self.peak_equity-eq)
        self.max_gross=max(self.max_gross,gross)
        if gross>0:
            margin=gross*((bid+ask)/2)/LEVERAGE
            if margin>0:self.min_ml=min(self.min_ml,eq/margin*100.0)

    def on_quote_tick(self,t):
        bid,ask=ff(t.bid_price),ff(t.ask_price); ts=int(t.ts_event)
        self.last_bid,self.last_ask,self.last_ns=bid,ask,ts
        if self.first_ns is None:self.first_ns=ts
        self.last_seen_ns=ts
        closed=[]
        for m in TF_MIN:
            if self._bar_update(m,ts,bid):closed.append(m)
            self._setup_tick(m,bid,ask,ts)
        for m in closed:
            self._after_close(m,ts)
        # Second pass is intentional: if MSS is confirmed by this first tick of
        # the new minute, execute immediately at this same raw Bid/Ask tick.
        for m in closed:
            self._setup_tick(m,bid,ask,ts)
        self._manage_legs(bid,ask,ts); self._risk(bid,ask)

    def on_stop(self):
        if self.last_bid is None:return
        for x in list(self.legs):
            if x.active:self._close_leg(x,self.last_bid,self.last_ask,self.last_ns,'EOD')

    def result(self):
        def met(rows):
            p=np.array([r['pnl'] for r in rows],float)
            gp=p[p>0].sum() if len(p) else 0.0; gl=-p[p<0].sum() if len(p) else 0.0
            return {'N':int(len(p)),'WR_pct':float((p>0).mean()*100) if len(p) else 0.0,
                'PF':float(gp/gl) if gl>0 else (math.inf if gp>0 else 0.0),'Net_USD':float(p.sum())}
        p=np.array([r['pnl'] for r in self.closed],float)
        base=met(self.closed); net=float(p.sum()) if len(p) else 0.0
        start=pd.Timestamp(self.first_ns,unit='ns',tz='UTC').date(); end=pd.Timestamp(self.last_seen_ns,unit='ns',tz='UTC').date()
        bdays=max(1,len(pd.bdate_range(start,end))); scale=21.0/bdays
        return {**base,'RF':float(net/self.max_dd) if self.max_dd>0 else None,
            'Return_pct':net/INITIAL*100.0,'MaxFloatingDD_USD':float(self.max_dd),'MaxFloatingDD_pct_initial':float(self.max_dd/INITIAL*100),
            'MinMarginLevel_pct_approx':None if math.isinf(self.min_ml) else float(self.min_ml),'MaxGrossQty_oz':float(self.max_gross),
            'MaxGrossLots_approx':float(self.max_gross/100.0),'BusinessDays':bdays,'Net21_USD_linearized':net*scale,
            'Monthly21_pct_linearized':net*scale/INITIAL*100.0,'N21_linearized':len(p)*scale,
            'per_tf':{str(m):met([r for r in self.closed if r['tf']==m]) for m in TF_MIN},
            'setup_stats':self.stats,'submitted_orders':self.submitted}

def executable(xs):
    one=Quantity.from_int(1); out=[]; repl=0
    for t in xs:
        b=ff(t.bid_size); a=ff(t.ask_size)
        if b<=0 or a<=0:
            out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,
                bid_size=one if b<=0 else t.bid_size,ask_size=one if a<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init)); repl+=1
        else:out.append(t)
    return out,repl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',action='append',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--body-mult',type=float,default=1.6); ap.add_argument('--range-atr-mult',type=float,default=0.9)
    a=ap.parse_args(); cats=[ParquetDataCatalog(x) for x in a.catalog]
    inst=next(x for x in cats[0].instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    raw=[]
    for cat in cats:
        ci=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
        raw.extend(cat.query(data_cls=QuoteTick,identifiers=[ci.id.value]))
    raw.sort(key=lambda t:int(t.ts_event)); ticks,repl=executable(raw)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
        base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks)
    st=M1FibLineFidelityV13(Cfg(instrument_id=inst.id,confirm_mode='fib-line',body_mult=a.body_mult,range_atr_mult=a.range_atr_mult)); eng.add_strategy(st); eng.run(); eng.end()
    out=Path('results/elliott-ict-m1-fib-line-v13')/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(st.closed).to_csv(out/'trades.csv',index=False)
    result={'verification_level':'NAUTILUS_BT_M1_FIB_LINE_FIDELITY_V13','engine':'NautilusTrader BacktestEngine',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'raw_ticks':len(raw),'execution_ticks':len(ticks),
        'ohlc_input_used':False,'signal_bars':'M1 original Elliott/Fib entry sequence preserved; wick-sensitive structure and Fib-zone qualification use line-chart Close only',
        'execution':'same raw Bid/Ask quote tick after the closed M1 line bar confirms MSS; zero artificial delay','commission_rt_per_lot':COMMISSION_RT_PER_LOT,
        'cashback_rt_per_lot':CASHBACK_RT_PER_LOT,'fib_zone':[0.786,0.804,0.822],'deep_fib_center':DEEP,'pre_entry_invalidation':'ENFORCED','direction_mapping':'bull=BUY,bear=SELL','confirm_mode':'fib-line','body_mult':a.body_mult,'range_atr_mult':a.range_atr_mult,
        'note':'Strategy logic is unchanged: Wave1 -> Fib 0.786/0.804/0.822 retracement zone -> MSS -> entry -> original SL/TP projection. Only M1 wick-sensitive structural readings use Close-line values. v13 fixes the v12 sweep-reference bug which incorrectly included the sweep point in its own prior-12 reference. Bull=BUY, bear=SELL, zero artificial delay.',
        **st.result()}
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str)); eng.dispose()

if __name__=='__main__':main()
