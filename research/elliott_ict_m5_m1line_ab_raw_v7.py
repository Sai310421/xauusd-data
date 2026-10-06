from __future__ import annotations
import argparse, json, math
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

INITIAL=1000.0
LEVERAGE=2000.0
BASE_QTY=1.0
DEEP=0.804
TIMEOUT_NS=240*60*1_000_000_000
LATENCY_NS=100*1_000_000
COMMISSION_RT_PER_LOT=7.0
CASHBACK_RT_PER_LOT=6.0
LINE_LOOKBACK=3

def ff(x):
    return float(x.as_double()) if hasattr(x,'as_double') else float(x)

@dataclass
class BarState:
    start_ns:int=0
    o:float=0.0
    h:float=0.0
    l:float=0.0
    c:float=0.0
    hist:list=field(default_factory=list)

@dataclass
class Setup:
    direction:int
    origin:float
    end:float
    atr:float
    signal_ns:int
    touched:bool=False
    touch_ns:int=0
    pending_ns:int=0
    invalid:bool=False
    entered:bool=False
    sl:float=0.0
    tp:float=0.0

@dataclass
class Leg:
    direction:int
    qty:float
    entry:float
    entry_ns:int
    sl:float
    tp:float
    expiry_ns:int
    active:bool=True

class Cfg(StrategyConfig, frozen=True):
    instrument_id: object
    confirm_mode: str='m5-candle'

class M5M1LineAB(Strategy):
    def __init__(self,c):
        super().__init__(c)
        self.mode=c.confirm_mode
        self.bars={1:BarState(),5:BarState()}
        self.setups=[]
        self.legs=[]
        self.closed=[]
        self.stats={'candidates':0,'touched':0,'invalid_before_entry':0,'confirmations':0,'entries':0}
        self.last_bid=None; self.last_ask=None; self.last_ns=0; self.submitted=0
        self.realized=0.0; self.peak_equity=INITIAL; self.max_dd=0.0
        self.min_ml=math.inf; self.max_gross=0.0
        self.first_ns=None; self.last_seen_ns=None

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def _submit(self,d,qty):
        inst=self.cache.instrument(self.config.instrument_id)
        q=inst.make_qty(Decimal(str(qty)))
        order=self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if d>0 else OrderSide.SELL,
            quantity=q,
        )
        self.submit_order(order); self.submitted+=1

    def _bar_update(self,m,ts,px):
        span=m*60*1_000_000_000
        bucket=(ts//span)*span
        b=self.bars[m]
        if b.start_ns==0:
            b.start_ns=bucket; b.o=b.h=b.l=b.c=px
            return False
        if bucket==b.start_ns:
            b.h=max(b.h,px); b.l=min(b.l,px); b.c=px
            return False
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
        a=tr[0]
        for x in tr[1:]:
            a=a+(x-a)/14.0
        return float(a)

    def _detect_m5(self):
        hist=self.bars[5].hist
        if len(hist)<35:return
        i=len(hist)-1
        ts,o,h,l,c=hist[i]
        prev20=hist[i-20:i]
        avg_body=float(np.mean([abs(x[4]-x[1]) for x in prev20]))
        atr=self._atr(hist)
        if avg_body<=0 or atr<=0:return
        if abs(c-o)<1.6*avg_body or (h-l)<0.9*atr:return
        ph=max(x[2] for x in hist[i-3:i]); pl=min(x[3] for x in hist[i-3:i])
        bull=c>o and c>ph
        bear=c<o and c<pl
        bull_origin=None; bear_origin=None
        for off in range(2,8):
            idx=len(hist)-off
            if idx<12:continue
            bar=hist[idx]; older=hist[idx-12:idx]
            ol=min(x[3] for x in older); oh=max(x[2] for x in older)
            if bull_origin is None and bar[3]<ol and bar[4]>ol:
                bull_origin=bar[3]
            if bear_origin is None and bar[2]>oh and bar[4]<oh:
                bear_origin=bar[2]
        s=None
        if bull and bull_origin is not None and h>bull_origin:
            s=Setup(1,bull_origin,h,atr,ts)
        elif bear and bear_origin is not None and bear_origin>l:
            s=Setup(-1,bear_origin,l,atr,ts)
        if s:
            span=abs(s.end-s.origin)
            s.sl=s.origin-s.direction*0.2*s.atr
            s.tp=s.end+s.direction*0.618*span
            self.setups.append(s)
            self.stats['candidates']+=1

    def _m5_candle_confirm(self,s):
        hist=self.bars[5].hist
        if len(hist)<2:return False
        a,b=hist[-1],hist[-2]
        return a[4]>b[2] if s.direction>0 else a[4]<b[3]

    def _m1_line_confirm(self,s):
        hist=self.bars[1].hist
        if len(hist)<LINE_LOOKBACK+1:return False
        closes=[x[4] for x in hist]
        cur=closes[-1]
        prev=closes[-1-LINE_LOOKBACK:-1]
        return cur>max(prev) if s.direction>0 else cur<min(prev)

    def _confirm_on_close(self,m,ts):
        if m==5:
            self._detect_m5()
        if self.mode=='m5-candle' and m!=5:return
        if self.mode=='m1-line' and m!=1:return
        for s in self.setups:
            if s.invalid or s.entered or not s.touched or s.pending_ns:continue
            if ts-s.signal_ns>TIMEOUT_NS:continue
            ok=self._m5_candle_confirm(s) if self.mode=='m5-candle' else self._m1_line_confirm(s)
            if ok:
                # Confirmation bar has just closed at current raw tick timestamp.
                s.pending_ns=ts+LATENCY_NS
                self.stats['confirmations']+=1

    def _setup_tick(self,bid,ask,ts):
        keep=[]
        for s in self.setups:
            if s.entered:continue
            if ts-s.signal_ns>TIMEOUT_NS:continue
            inv=bid<=s.sl if s.direction>0 else ask>=s.sl
            if inv:
                s.invalid=True; self.stats['invalid_before_entry']+=1
                continue
            level=s.end-s.direction*abs(s.end-s.origin)*DEEP
            if not s.touched:
                touch=bid<=level if s.direction>0 else ask>=level
                if touch:
                    s.touched=True; s.touch_ns=ts; self.stats['touched']+=1
            if s.pending_ns and ts>=s.pending_ns:
                px=ask if s.direction>0 else bid
                if (s.direction>0 and px<=s.sl) or (s.direction<0 and px>=s.sl):
                    s.invalid=True; self.stats['invalid_before_entry']+=1
                    continue
                self._submit(s.direction,BASE_QTY)
                self.legs.append(Leg(s.direction,BASE_QTY,px,ts,s.sl,s.tp,s.signal_ns+TIMEOUT_NS,True))
                s.entered=True; self.stats['entries']+=1
                continue
            keep.append(s)
        self.setups=keep

    def _close_leg(self,x,bid,ask,ts,reason):
        if not x.active:return
        px=bid if x.direction>0 else ask
        lots=x.qty/100.0
        gross=(px-x.entry)*x.direction*x.qty
        cost=lots*COMMISSION_RT_PER_LOT
        cb=lots*CASHBACK_RT_PER_LOT
        pnl=gross-cost+cb
        self._submit(-x.direction,x.qty)
        x.active=False
        self.realized+=pnl
        self.closed.append({
            'direction':x.direction,'qty':x.qty,'entry':x.entry,'exit':px,
            'gross':gross,'commission_rt':cost,'cashback_rt':cb,'pnl':pnl,
            'entry_ns':x.entry_ns,'exit_ns':ts,'reason':reason,'confirm_mode':self.mode,
        })

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
            floating+=(mark-x.entry)*x.direction*x.qty
            gross+=x.qty
        eq=INITIAL+self.realized+floating
        self.peak_equity=max(self.peak_equity,eq)
        self.max_dd=max(self.max_dd,self.peak_equity-eq)
        self.max_gross=max(self.max_gross,gross)
        if gross>0:
            margin=gross*((bid+ask)/2.0)/LEVERAGE
            if margin>0:self.min_ml=min(self.min_ml,eq/margin*100.0)

    def on_quote_tick(self,t):
        bid,ask=ff(t.bid_price),ff(t.ask_price)
        ts=int(t.ts_event)
        self.last_bid,self.last_ask,self.last_ns=bid,ask,ts
        if self.first_ns is None:self.first_ns=ts
        self.last_seen_ns=ts
        closed=[]
        for m in (1,5):
            if self._bar_update(m,ts,bid):closed.append(m)
        self._setup_tick(bid,ask,ts)
        for m in closed:
            self._confirm_on_close(m,ts)
        self._manage_legs(bid,ask,ts)
        self._risk(bid,ask)

    def on_stop(self):
        if self.last_bid is None:return
        for x in list(self.legs):
            if x.active:self._close_leg(x,self.last_bid,self.last_ask,self.last_ns,'EOD')

    def result(self):
        p=np.array([r['pnl'] for r in self.closed],float)
        gp=p[p>0].sum() if len(p) else 0.0
        gl=-p[p<0].sum() if len(p) else 0.0
        pf=float(gp/gl) if gl>0 else (math.inf if gp>0 else 0.0)
        wr=float((p>0).mean()*100.0) if len(p) else 0.0
        net=float(p.sum()) if len(p) else 0.0
        start=pd.Timestamp(self.first_ns,unit='ns',tz='UTC').date()
        end=pd.Timestamp(self.last_seen_ns,unit='ns',tz='UTC').date()
        bdays=max(1,len(pd.bdate_range(start,end)))
        scale=21.0/bdays
        return {
            'N':int(len(p)),'WR_pct':wr,'PF':pf,'Net_USD':net,'Return_pct':net/INITIAL*100.0,
            'RF':float(net/self.max_dd) if self.max_dd>0 else None,
            'MaxFloatingDD_USD':float(self.max_dd),'MaxFloatingDD_pct_initial':float(self.max_dd/INITIAL*100.0),
            'MinMarginLevel_pct_approx':None if math.isinf(self.min_ml) else float(self.min_ml),
            'MaxGrossQty_oz':float(self.max_gross),'MaxGrossLots_approx':float(self.max_gross/100.0),
            'BusinessDays':bdays,'Net21_USD_linearized':float(net*scale),
            'Monthly21_pct_linearized':float(net*scale/INITIAL*100.0),'N21_linearized':float(len(p)*scale),
            'setup_stats':self.stats,'submitted_orders':self.submitted,
        }

def executable(xs):
    one=Quantity.from_int(1); out=[]; repl=0
    for t in xs:
        b=ff(t.bid_size); a=ff(t.ask_size)
        if b<=0 or a<=0:
            out.append(QuoteTick(
                instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,
                bid_size=one if b<=0 else t.bid_size,ask_size=one if a<=0 else t.ask_size,
                ts_event=t.ts_event,ts_init=t.ts_init,
            ))
            repl+=1
        else:
            out.append(t)
    return out,repl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',action='append',required=True)
    ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--confirm-mode',choices=['m5-candle','m1-line'],required=True)
    a=ap.parse_args()
    cats=[ParquetDataCatalog(x) for x in a.catalog]
    inst=next(x for x in cats[0].instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    raw=[]
    for cat in cats:
        ci=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
        raw.extend(cat.query(data_cls=QuoteTick,identifiers=[ci.id.value]))
    raw.sort(key=lambda t:int(t.ts_event))
    ticks,repl=executable(raw)

    eng=BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level='ERROR'),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    eng.add_venue(
        venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
        base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal('2000'),
    )
    eng.add_instrument(inst); eng.add_data(ticks)
    st=M5M1LineAB(Cfg(instrument_id=inst.id,confirm_mode=a.confirm_mode))
    eng.add_strategy(st); eng.run(); eng.end()

    out=Path('results/elliott-ict-m5-m1line-v7')/a.experiment_id
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(st.closed).to_csv(out/'trades.csv',index=False)
    result={
        'verification_level':'NAUTILUS_BT_M5_M1_LINE_AB_V7',
        'engine':'NautilusTrader BacktestEngine',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'confirm_mode':a.confirm_mode,'raw_ticks':len(raw),'execution_ticks':len(ticks),
        'ohlc_input_used':False,
        'signal':'M5 Wave1 + Fib 0.804 touch; confirmation is M5 candle MSS or M1 close-only line MSS',
        'm1_line_definition':f'current M1 close breaks previous {LINE_LOOKBACK} M1 closes',
        'execution':'first raw Bid/Ask QuoteTick at/after confirmation close + 100ms',
        'commission_rt_per_lot':COMMISSION_RT_PER_LOT,'cashback_rt_per_lot':CASHBACK_RT_PER_LOT,
        'deep_fib':DEEP,'pre_entry_invalidation':'ENFORCED','raw_zero_size_quotes_replaced':repl,
        **st.result(),
    }
    result['PF_gate']='PASS' if result['PF']>1.0 else 'FAIL'
    result['PF_1_20_gate']='PASS' if result['PF']>=1.2 else 'FAIL'
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str))
    eng.dispose()

if __name__=='__main__':
    main()
