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

EDGE_IDS = [
    'T1_EMA21','T2_MTF_ALIGN','T3_BREAKOUT_CONT','T4_ACCEL',
    'R1_RANGE_BIAS','R2_COMP_EXP','R3_MICRO_BREAK',
    'V1_SWEEP','V2_MEAN_EXTREME','V3_REJECTION',
]
COMBOS = {
    'TREND_CORE': ['T1_EMA21','T2_MTF_ALIGN','T3_BREAKOUT_CONT','T4_ACCEL'],
    'RANGE_CORE': ['R2_COMP_EXP','R3_MICRO_BREAK'],
    'REVERSAL_CORE': ['V1_SWEEP','V3_REJECTION'],
}

class MultiEdgeConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    m1: BarType
    m5: BarType
    m15: BarType
    selected: str
    initial_balance: float = 1000.0
    unit_qty: Decimal = Decimal('1')
    horizon_minutes: int = 60
    tp_atr: float = 0.75
    sl_atr: float = 0.75

class MultiEdgeRawStrategy(Strategy):
    def __init__(self, config: MultiEdgeConfig):
        super().__init__(config)
        self.c = {1:deque(maxlen=240),5:deque(maxlen=240),15:deque(maxlen=240)}
        self.h = {1:deque(maxlen=240),5:deque(maxlen=240),15:deque(maxlen=240)}
        self.l = {1:deque(maxlen=240),5:deque(maxlen=240),15:deque(maxlen=240)}
        self.tr = deque(maxlen=240); self.prev_close=None
        self.m1_i=0; self.tick_i=0; self.pending=0
        self.active=False; self.side=0; self.entry=0.0; self.entry_m1=0; self.entry_atr=0.0
        self.best=0.0; self.worst=0.0; self.last_bid=None; self.last_ask=None
        self.trades=[]; self.gw=0.0; self.gl=0.0; self.wins=0; self.losses=0
        self.submitted=0; self.order_events=Counter(); self.fire_count=0

    @staticmethod
    def _f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.m1); self.subscribe_bars(self.config.m5); self.subscribe_bars(self.config.m15)

    def on_order_submitted(self,e): self.order_events['submitted']+=1
    def on_order_accepted(self,e): self.order_events['accepted']+=1
    def on_order_filled(self,e): self.order_events['filled']+=1
    def on_order_denied(self,e): self.order_events['denied']+=1
    def on_order_rejected(self,e): self.order_events['rejected']+=1

    def _atr(self): return float(np.mean(self.tr[-20:])) if self.tr else 0.0
    def _ema(self, arr, n):
        a=np.asarray(list(arr)[-n:],float)
        if len(a)<n:return None
        alpha=2.0/(n+1.0); v=float(a[0])
        for x in a[1:]: v=alpha*float(x)+(1-alpha)*v
        return v
    def _rsi(self,n=14):
        a=np.asarray(self.c[1],float)
        if len(a)<n+1:return 50.0
        d=np.diff(a[-(n+1):]); up=np.maximum(d,0).mean(); dn=np.maximum(-d,0).mean()
        if dn<=1e-12:return 100.0
        rs=up/dn; return 100.0-100.0/(1.0+rs)
    def _atr_pct(self):
        if len(self.tr)<40:return 0.5
        vals=np.asarray(self.tr,float); cur=float(np.mean(vals[-20:])); hist=np.asarray([np.mean(vals[max(0,i-19):i+1]) for i in range(19,len(vals))])
        return float(np.mean(hist<=cur)) if len(hist) else 0.5
    def _tf_bias(self):
        out=[]
        for tf in (1,5,15):
            e21=self._ema(self.c[tf],21); e50=self._ema(self.c[tf],50)
            if e21 is None or e50 is None: continue
            out.append(1 if e21>e50 else -1 if e21<e50 else 0)
        return float(np.mean(out)) if out else 0.0
    def _edge_scores(self):
        if len(self.c[1])<70:return {}
        c=np.asarray(self.c[1],float); h=np.asarray(self.h[1],float); l=np.asarray(self.l[1],float)
        atr=max(self._atr(),1e-9); e21=self._ema(self.c[1],21); e50=self._ema(self.c[1],50)
        e21_prev=float(np.mean(c[-22:-1])) if len(c)>=22 else e21
        loc=np.clip((c[-1]-e21)/atr,-1,1); slope=np.clip((e21-e21_prev)/atr*8,-1,1)
        t1=float(np.clip(.65*loc+.35*slope,-1,1))
        t2=float(np.clip(self._tf_bias(),-1,1))
        hh=float(h[-21:-1].max()); ll=float(l[-21:-1].min())
        t3=float(np.clip((c[-1]-hh)/atr*2,-1,1)) if c[-1]>hh else float(-np.clip((ll-c[-1])/atr*2,-1,1)) if c[-1]<ll else 0.0
        mom=float(c[-1]-c[-4]); mom0=float(c[-4]-c[-7]); den=max(abs(mom0),atr*.05,1e-9)
        t4=float(np.clip((mom-mom0)/den,-1,1))
        r1=float(np.clip(.75*np.clip((c[-1]-e21)/atr*.8,-1,1)+.25*np.clip((e21-e50)/atr,-1,1),-1,1))
        comp=float(np.clip(1-self._atr_pct(),0,1)); r2=(1 if mom>0 else -1 if mom<0 else 0)*comp
        rh=float(h[-8:-1].max()); rl=float(l[-8:-1].min())
        r3=float(np.clip((c[-1]-rh)/atr*3,-1,1)) if c[-1]>rh else float(-np.clip((rl-c[-1])/atr*3,-1,1)) if c[-1]<rl else 0.0
        swept_high = 1.0 if h[-1]>hh and c[-1]<hh else 0.0
        swept_low = 1.0 if l[-1]<ll and c[-1]>ll else 0.0
        v1=float(swept_low-swept_high)
        rsi=self._rsi(); v2=float(-np.clip((rsi-50)/35,0,1)) if rsi>=50 else float(np.clip((50-rsi)/35,0,1))
        rng=max(h[-1]-l[-1],1e-9); body=abs(c[-1]-c[-2]); upper=h[-1]-max(c[-1],c[-2]); lower=min(c[-1],c[-2])-l[-1]
        v3=float(np.clip((lower-upper)/rng,-1,1)) if body/rng<.8 else 0.0
        return {'T1_EMA21':t1,'T2_MTF_ALIGN':t2,'T3_BREAKOUT_CONT':t3,'T4_ACCEL':t4,
                'R1_RANGE_BIAS':r1,'R2_COMP_EXP':r2,'R3_MICRO_BREAK':r3,
                'V1_SWEEP':v1,'V2_MEAN_EXTREME':v2,'V3_REJECTION':v3}
    def _signal(self):
        s=self._edge_scores()
        if not s:return 0
        selected = COMBOS.get(self.config.selected,[self.config.selected])
        vals=[s[e] for e in selected]
        if len(vals)==1:
            v=vals[0]
            if abs(v)>=0.25:self.fire_count+=1; return 1 if v>0 else -1
            return 0
        nz=[v for v in vals if abs(v)>=0.15]
        if len(nz)<2:return 0
        pos=sum(v>0 for v in nz); neg=sum(v<0 for v in nz)
        if pos and neg:return 0
        comp=float(np.mean(nz))
        if abs(comp)>=0.25:self.fire_count+=1; return 1 if comp>0 else -1
        return 0
    def _submit(self,side):
        inst=self.cache.instrument(self.config.instrument_id); qty=inst.make_qty(self.config.unit_qty)
        o=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=qty)
        self.submitted+=1; self.submit_order(o)
    def _open(self,side,bid,ask):
        if self.active:return
        self._submit(side); self.active=True; self.side=side; self.entry=ask if side>0 else bid; self.entry_m1=self.m1_i; self.entry_atr=max(self._atr(),1e-9); self.best=self.entry; self.worst=self.entry
    def _close(self,bid,ask,reason):
        if not self.active:return
        px=bid if self.side>0 else ask; pnl=(px-self.entry)*self.side
        self._submit(-self.side)
        mfe=max(0.0,(self.best-self.entry)*self.side if self.side>0 else (self.entry-self.best))
        mae=max(0.0,(self.entry-self.worst) if self.side>0 else (self.worst-self.entry))
        self.trades.append({'pnl':pnl,'mfe':mfe,'mae':mae,'reason':reason,'side':self.side,'hold_m1':self.m1_i-self.entry_m1})
        if pnl>0:self.wins+=1;self.gw+=pnl
        elif pnl<0:self.losses+=1;self.gl+=abs(pnl)
        self.active=False; self.side=0
    def on_bar(self,bar:Bar):
        s=str(bar.bar_type)
        tf=15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
        if tf is None:return
        o,h,l,c=map(self._f,[bar.open,bar.high,bar.low,bar.close]); self.c[tf].append(c); self.h[tf].append(h); self.l[tf].append(l)
        if tf==1:
            self.m1_i+=1
            tr=max(h-l,abs(h-self.prev_close) if self.prev_close is not None else 0,abs(l-self.prev_close) if self.prev_close is not None else 0)
            self.tr.append(tr); self.prev_close=c
            if not self.active and self.pending==0:self.pending=self._signal()
    def on_quote_tick(self,tick:QuoteTick):
        self.tick_i+=1; bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); self.last_bid=bid; self.last_ask=ask
        if self.pending and not self.active:
            x=self.pending; self.pending=0; self._open(x,bid,ask)
        if not self.active:return
        mark=bid if self.side>0 else ask
        if self.side>0:self.best=max(self.best,mark);self.worst=min(self.worst,mark)
        else:self.best=min(self.best,mark);self.worst=max(self.worst,mark)
        move=(mark-self.entry)*self.side
        if move>=self.config.tp_atr*self.entry_atr:self._close(bid,ask,'TP_FIRST');return
        if move<=-self.config.sl_atr*self.entry_atr:self._close(bid,ask,'SL_FIRST');return
        if self.m1_i-self.entry_m1>=self.config.horizon_minutes:self._close(bid,ask,'HORIZON');return
    def on_stop(self):
        if self.active and self.last_bid is not None:self._close(self.last_bid,self.last_ask,'EOD')
    def summary(self):
        n=len(self.trades); net=sum(x['pnl'] for x in self.trades); pf=self.gw/self.gl if self.gl>0 else (math.inf if self.gw>0 else 0.0)
        eq=self.config.initial_balance; peak=eq; mdd=0.0
        for t in self.trades:
            eq+=t['pnl']; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/max(peak,1e-9)*100)
        reasons=Counter(t['reason'] for t in self.trades)
        return {'selected':self.config.selected,'N':n,'fire_count':self.fire_count,'WR_pct':100*self.wins/max(n,1),'PF':pf,'net_virtual':net,
                'expectancy':net/max(n,1),'max_DD_pct':mdd,'MFE_mean':float(np.mean([t['mfe'] for t in self.trades])) if n else 0.0,
                'MAE_mean':float(np.mean([t['mae'] for t in self.trades])) if n else 0.0,'first_touch':dict(reasons),'submitted_orders':self.submitted,'order_events':dict(self.order_events)}

def ensure_l1(ticks):
    one=Quantity.from_int(1); out=[]; rep=0
    for t in ticks:
        bs=float(t.bid_size.as_double()) if hasattr(t.bid_size,'as_double') else float(t.bid_size); az=float(t.ask_size.as_double()) if hasattr(t.ask_size,'as_double') else float(t.ask_size)
        if bs<=0 or az<=0:
            out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init));rep+=1
        else: out.append(t)
    return out,rep

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--selected',required=True,choices=EDGE_IDS+list(COMBOS)); ap.add_argument('--experiment-id',required=True); a=ap.parse_args()
    catalog=ParquetDataCatalog(a.catalog); inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD missing')
    raw=catalog.query_quote_ticks(identifiers=[inst.id.value]);
    if not raw:raise SystemExit('no raw XAUUSD QuoteTicks')
    ticks,replaced=ensure_l1(raw)
    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(inst); engine.add_data(ticks)
    cfg=MultiEdgeConfig(instrument_id=inst.id,m1=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{inst.id.value}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{inst.id.value}-15-MINUTE-BID-INTERNAL'),selected=a.selected)
    st=MultiEdgeRawStrategy(cfg); engine.add_strategy(st); engine.run()
    fills=engine.trader.generate_order_fills_report(); orders=engine.trader.generate_orders_report()
    obj={'verification_level':'NAUTILUS_RAW_BIDASK_MULTIEDGE_GATE_AB','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'raw_ticks':len(raw),'execution_ticks':len(ticks),'raw_zero_size_quotes_replaced':replaced,'ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL M1/M5/M15 from raw QuoteTicks','native_fills':int(len(fills)) if fills is not None else 0,'native_orders':int(len(orders)) if orders is not None else 0,**st.summary()}
    out=Path('results/multiedge')/a.experiment_id; out.mkdir(parents=True,exist_ok=True); (out/f'{a.selected}.json').write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8'); print(json.dumps(obj,indent=2,default=str)); engine.dispose()

if __name__=='__main__':main()
