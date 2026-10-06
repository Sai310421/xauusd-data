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
BASE_QTY_OZ=1.0       # 0.01 lot when 1 lot = 100 oz
MAX_CONCURRENT=3
MAX_GROSS_QTY_OZ=3.0  # 0.03 lot
DEEP_FIB=0.804
SWING_LOOKBACK=12
SWEEP_SEARCH=6
MSS_LOOKBACK=3
AVG_BODY_LOOKBACK=20
ATR_PERIOD=14
BODY_MULT=1.60
RANGE_ATR_MULT=0.90
INVALID_ATR=0.20
TIMEOUT_NS=240*60*1_000_000_000
PROJECTION=0.618
M1_CLOSE_LOOKBACK=3
M15_CLOSE_LOOKBACK=3
COMMISSION_RT_PER_LOT=7.0
CASHBACK_RT_PER_LOT=6.0

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
    endp:float
    atr:float
    signal_ns:int
    fib804:float
    sl:float
    tp:float
    touched:bool=False
    pending:bool=False
    confirm_ns:int=0
    active:bool=True

@dataclass
class Leg:
    direction:int
    qty:float
    entry:float
    entry_ns:int
    sl:float
    tp:float
    expiry_ns:int
    m1_ok:bool
    m15_ok:bool
    active:bool=True

class Cfg(StrategyConfig, frozen=True):
    instrument_id: object
    gate_mode: str='current'

class OneEAv040Parity(Strategy):
    def __init__(self,c):
        super().__init__(c)
        self.mode=c.gate_mode
        self.require_m1=self.mode in ('m1-gate','both-gates')
        self.require_m15=self.mode in ('m15-gate','both-gates')
        self.bars={1:BarState(),5:BarState(),15:BarState()}
        self.setups=[]
        self.legs=[]
        self.closed=[]
        self.last_bid=None; self.last_ask=None; self.last_ns=0
        self.first_ns=None; self.last_seen_ns=None
        self.realized=0.0; self.peak_equity=INITIAL; self.max_dd=0.0
        self.min_ml=math.inf; self.max_gross=0.0
        self.submitted=0
        self.stats={
            'candidates':0,'fib_touched':0,'invalid_before_entry':0,
            'm5_confirmations':0,'entries':0,'blocked_capacity_ticks':0,
            'blocked_m1_ticks':0,'blocked_m15_ticks':0,
        }

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
        self.submit_order(order)
        self.submitted+=1

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
            _,o,h,l,c=hist[i]
            pc=hist[i-1][4]
            tr.append(max(h-l,abs(h-pc),abs(l-pc)))
        a=tr[0]
        for x in tr[1:]:
            a += (x-a)/ATR_PERIOD
        return float(a)

    def _detect_m5(self):
        hist=self.bars[5].hist
        if len(hist)<64:return
        i=len(hist)-1
        ts,o,h,l,c=hist[i]
        prev20=hist[i-AVG_BODY_LOOKBACK:i]
        avg_body=float(np.mean([abs(x[4]-x[1]) for x in prev20]))
        atr=self._atr(hist)
        if avg_body<=0 or atr<=0:return
        if abs(c-o)<BODY_MULT*avg_body:return
        if (h-l)<RANGE_ATR_MULT*atr:return

        prev_hi=max(x[2] for x in hist[i-MSS_LOOKBACK:i])
        prev_lo=min(x[3] for x in hist[i-MSS_LOOKBACK:i])
        bull_disp=c>o and c>prev_hi
        bear_disp=c<o and c<prev_lo
        if not bull_disp and not bear_disp:return

        bull_origin=None; bear_origin=None
        for off in range(2,SWEEP_SEARCH+2):
            idx=len(hist)-off
            if idx<SWING_LOOKBACK:continue
            bar=hist[idx]
            older=hist[idx-SWING_LOOKBACK:idx]
            older_lo=min(x[3] for x in older)
            older_hi=max(x[2] for x in older)
            if bull_origin is None and bar[3]<older_lo and bar[4]>older_lo:
                bull_origin=bar[3]
            if bear_origin is None and bar[2]>older_hi and bar[4]<older_hi:
                bear_origin=bar[2]

        direction=0; origin=0.0; endp=0.0
        if bull_disp and bull_origin is not None and h>bull_origin:
            direction=1; origin=bull_origin; endp=h
        elif bear_disp and bear_origin is not None and bear_origin>l:
            direction=-1; origin=bear_origin; endp=l
        if direction==0:return

        span=abs(endp-origin)
        fib=endp-direction*span*DEEP_FIB
        sl=origin-direction*INVALID_ATR*atr
        tp=endp+direction*PROJECTION*span
        self.setups.append(Setup(direction,origin,endp,atr,ts,fib,sl,tp))
        self.stats['candidates']+=1

    def _m5_confirm(self,s):
        hist=self.bars[5].hist
        if len(hist)<2:return False
        cur,prev=hist[-1],hist[-2]
        return cur[4]>prev[2] if s.direction>0 else cur[4]<prev[3]

    def _line_aligned(self,m,lookback,direction):
        hist=self.bars[m].hist
        if len(hist)<lookback+1:return False
        cur=hist[-1][4]
        prev=[x[4] for x in hist[-1-lookback:-1]]
        return cur>max(prev) if direction>0 else cur<min(prev)

    def _m1_ok(self,d):
        return self._line_aligned(1,M1_CLOSE_LOOKBACK,d)

    def _m15_ok(self,d):
        return self._line_aligned(15,M15_CLOSE_LOOKBACK,d)

    def _confirm_touched_on_m5_close(self,ts):
        for s in self.setups:
            if not s.active or not s.touched or s.pending:continue
            if ts-s.signal_ns>TIMEOUT_NS:continue
            if self._m5_confirm(s):
                s.pending=True
                s.confirm_ns=ts
                self.stats['m5_confirmations']+=1

    def _active_count(self):
        return sum(1 for x in self.legs if x.active)

    def _active_qty(self):
        return sum(x.qty for x in self.legs if x.active)

    def _manage_setups_tick(self,bid,ask,ts):
        keep=[]
        for s in self.setups:
            if not s.active:continue
            if ts-s.signal_ns>TIMEOUT_NS:
                s.active=False
                continue

            invalid=(bid<=s.sl) if s.direction>0 else (ask>=s.sl)
            if invalid:
                s.active=False
                self.stats['invalid_before_entry']+=1
                continue

            if not s.touched:
                touch=(bid<=s.fib804) if s.direction>0 else (ask>=s.fib804)
                if touch:
                    s.touched=True
                    self.stats['fib_touched']+=1

            if s.pending and ts>s.confirm_ns:
                m1ok=self._m1_ok(s.direction)
                m15ok=self._m15_ok(s.direction)
                if self.require_m1 and not m1ok:
                    self.stats['blocked_m1_ticks']+=1
                    keep.append(s); continue
                if self.require_m15 and not m15ok:
                    self.stats['blocked_m15_ticks']+=1
                    keep.append(s); continue
                if self._active_count()>=MAX_CONCURRENT or self._active_qty()+BASE_QTY_OZ>MAX_GROSS_QTY_OZ+1e-12:
                    self.stats['blocked_capacity_ticks']+=1
                    keep.append(s); continue

                px=ask if s.direction>0 else bid
                if (s.direction>0 and px<=s.sl) or (s.direction<0 and px>=s.sl):
                    s.active=False
                    self.stats['invalid_before_entry']+=1
                    continue
                self._submit(s.direction,BASE_QTY_OZ)
                self.legs.append(Leg(
                    s.direction,BASE_QTY_OZ,px,ts,s.sl,s.tp,s.signal_ns+TIMEOUT_NS,m1ok,m15ok,True
                ))
                self.stats['entries']+=1
                s.active=False
                continue
            keep.append(s)
        self.setups=keep

    def _close_leg(self,x,bid,ask,ts,reason):
        if not x.active:return
        px=bid if x.direction>0 else ask
        gross=(px-x.entry)*x.direction*x.qty
        lots=x.qty/100.0
        commission=lots*COMMISSION_RT_PER_LOT
        cashback=lots*CASHBACK_RT_PER_LOT
        pnl=gross-commission+cashback
        self._submit(-x.direction,x.qty)
        x.active=False
        self.realized+=pnl
        self.closed.append({
            'direction':x.direction,'qty_oz':x.qty,'entry':x.entry,'exit':px,
            'gross_usd':gross,'commission_rt_usd':commission,'cashback_rt_usd':cashback,
            'pnl':pnl,'entry_ns':x.entry_ns,'exit_ns':ts,'reason':reason,
            'm1_aligned_at_entry':x.m1_ok,'m15_aligned_at_entry':x.m15_ok,
            'both_aligned_at_entry':bool(x.m1_ok and x.m15_ok),'gate_mode':self.mode,
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
        floating=0.0; gross_qty=0.0
        for x in self.legs:
            if not x.active:continue
            mark=bid if x.direction>0 else ask
            floating+=(mark-x.entry)*x.direction*x.qty
            gross_qty+=x.qty
        eq=INITIAL+self.realized+floating
        self.peak_equity=max(self.peak_equity,eq)
        self.max_dd=max(self.max_dd,self.peak_equity-eq)
        self.max_gross=max(self.max_gross,gross_qty)
        if gross_qty>0:
            margin=gross_qty*((bid+ask)/2.0)/LEVERAGE
            if margin>0:self.min_ml=min(self.min_ml,eq/margin*100.0)

    def on_quote_tick(self,t):
        bid,ask=ff(t.bid_price),ff(t.ask_price)
        ts=int(t.ts_event)
        self.last_bid,self.last_ask,self.last_ns=bid,ask,ts
        if self.first_ns is None:self.first_ns=ts
        self.last_seen_ns=ts

        # Match MT5 v0.40 order: manage existing setup/legs first, then process new bars.
        self._manage_setups_tick(bid,ask,ts)
        self._manage_legs(bid,ask,ts)

        closed=[]
        for m in (1,5,15):
            if self._bar_update(m,ts,bid):
                closed.append(m)
        if 5 in closed:
            self._detect_m5()
            self._confirm_touched_on_m5_close(ts)

        self._risk(bid,ask)

    def on_stop(self):
        if self.last_bid is None:return
        for x in list(self.legs):
            if x.active:
                self._close_leg(x,self.last_bid,self.last_ask,self.last_ns,'EOD')

    def _metrics(self,rows):
        p=np.array([r['pnl'] for r in rows],float)
        gp=p[p>0].sum() if len(p) else 0.0
        gl=-p[p<0].sum() if len(p) else 0.0
        pf=float(gp/gl) if gl>0 else (math.inf if gp>0 else 0.0)
        return {
            'N':int(len(p)),
            'WR_pct':float((p>0).mean()*100.0) if len(p) else 0.0,
            'PF':pf,
            'Net_USD':float(p.sum()) if len(p) else 0.0,
        }

    def result(self):
        base=self._metrics(self.closed)
        net=base['Net_USD']
        start=pd.Timestamp(self.first_ns,unit='ns',tz='UTC').date()
        end=pd.Timestamp(self.last_seen_ns,unit='ns',tz='UTC').date()
        bdays=max(1,len(pd.bdate_range(start,end)))
        scale=21.0/bdays
        base.update({
            'RF':float(net/self.max_dd) if self.max_dd>0 else None,
            'Return_pct':net/INITIAL*100.0,
            'MaxFloatingDD_USD':float(self.max_dd),
            'MaxFloatingDD_pct_initial':float(self.max_dd/INITIAL*100.0),
            'MinMarginLevel_pct_approx':None if math.isinf(self.min_ml) else float(self.min_ml),
            'MaxGrossQty_oz':float(self.max_gross),
            'MaxGrossLots_approx':float(self.max_gross/100.0),
            'BusinessDays':bdays,
            'Net21_USD_linearized':float(net*scale),
            'Monthly21_pct_linearized':float(net*scale/INITIAL*100.0),
            'N21_linearized':float(base['N']*scale),
            'setup_stats':self.stats,
            'subgroups':{
                'm1_aligned':self._metrics([r for r in self.closed if r['m1_aligned_at_entry']]),
                'm1_not_aligned':self._metrics([r for r in self.closed if not r['m1_aligned_at_entry']]),
                'm15_aligned':self._metrics([r for r in self.closed if r['m15_aligned_at_entry']]),
                'm15_not_aligned':self._metrics([r for r in self.closed if not r['m15_aligned_at_entry']]),
                'both_aligned':self._metrics([r for r in self.closed if r['both_aligned_at_entry']]),
                'not_both_aligned':self._metrics([r for r in self.closed if not r['both_aligned_at_entry']]),
            },
            'submitted_orders':self.submitted,
        })
        return base

def executable(xs):
    one=Quantity.from_int(1); out=[]; repl=0
    for t in xs:
        b=ff(t.bid_size); a=ff(t.ask_size)
        if b<=0 or a<=0:
            out.append(QuoteTick(
                instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,
                bid_size=one if b<=0 else t.bid_size,
                ask_size=one if a<=0 else t.ask_size,
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
    ap.add_argument('--gate-mode',choices=['current','m1-gate','m15-gate','both-gates'],default='current')
    a=ap.parse_args()

    cats=[ParquetDataCatalog(p) for p in a.catalog]
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
    st=OneEAv040Parity(Cfg(instrument_id=inst.id,gate_mode=a.gate_mode))
    eng.add_strategy(st); eng.run(); eng.end()

    out=Path('results/elliott-ict-oneea-v040')/a.experiment_id
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(st.closed).to_csv(out/'trades.csv',index=False)
    result={
        'verification_level':'NAUTILUS_BT_ONEEA_V040_PARITY',
        'engine':'NautilusTrader BacktestEngine',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'ea_profile':'XAUUSD_ElliottICT_MTF_OneEA_v0_40',
        'gate_mode':a.gate_mode,
        'raw_ticks':len(raw),'execution_ticks':len(ticks),'ohlc_input_used':False,
        'signal':'M5 Wave1 + sweep + Fib 0.804 + M5 one-bar MSS; M1/M15 close-line context',
        'entry_timing':'first raw quote tick strictly after M5 confirmation, matching EA OnTick order; no artificial 100ms delay',
        'cost_assumption':'raw spread + $7/lot round-trip commission - $6/lot round-trip cashback',
        'slippage_note':'MT5 SlippagePoints is a max deviation setting, not deterministic slippage; no extra slippage penalty applied here.',
        'max_concurrent_positions':MAX_CONCURRENT,'max_total_lots':MAX_GROSS_QTY_OZ/100.0,
        'm1_hard_gate':a.gate_mode in ('m1-gate','both-gates'),
        'm15_hard_gate':a.gate_mode in ('m15-gate','both-gates'),
        'confluence_boost':False,
        'raw_zero_size_quotes_replaced':repl,
        **st.result(),
    }
    result['PF_gate']='PASS' if result['PF']>1.0 else 'FAIL'
    result['PF_1_20_gate']='PASS' if result['PF']>=1.2 else 'FAIL'
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str))
    eng.dispose()

if __name__=='__main__':
    main()
