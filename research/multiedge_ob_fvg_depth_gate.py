from __future__ import annotations

"""OB ∩ FVG five-depth diagnostic gate for XAUUSD M1 Raw BidAsk.

Purpose:
- Detect a confirmed breakout/displacement.
- Build a strict FVG and the last opposite M1 candle as the candidate OB.
- Require a non-empty OB ∩ FVG overlap.
- Evaluate independent entries at 0/25/50/75/100% of the overlap zone.
- Keep each depth independent so we can learn where the entry-price EDGE actually lives.

This is deliberately a diagnostic gate, not the final basket sizer.
"""

import argparse
import json
import math
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

DEPTHS = (0, 25, 50, 75, 100)

class GateConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    m1: BarType
    initial_balance: float = 1000.0
    unit_qty: Decimal = Decimal('1')
    breakout_lookback: int = 20
    ob_search_bars: int = 6
    max_wait_bars: int = 10
    displacement_body_atr: float = 0.60
    tp_atr: float = 0.75
    sl_atr: float = 0.75
    horizon_minutes: int = 60

class DepthGate(Strategy):
    def __init__(self, config: GateConfig):
        super().__init__(config)
        self.o=deque(maxlen=300); self.h=deque(maxlen=300); self.l=deque(maxlen=300); self.c=deque(maxlen=300)
        self.tr=deque(maxlen=300); self.prev_close=None; self.m1_i=0
        self.last_bid=None; self.last_ask=None
        self.zone=None
        self.active={d:None for d in DEPTHS}
        self.stats={d:{'trades':[],'gw':0.0,'gl':0.0,'wins':0,'losses':0,'fills':0} for d in DEPTHS}
        self.zone_count=0; self.rejected_no_overlap=0; self.rejected_no_ob=0; self.rejected_no_fvg=0

    @staticmethod
    def _f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id); self.subscribe_bars(self.config.m1)
    def _atr(self):
        return float(np.mean(list(self.tr)[-20:])) if len(self.tr)>=20 else 0.0
    def _submit(self, side):
        inst=self.cache.instrument(self.config.instrument_id); qty=inst.make_qty(self.config.unit_qty)
        o=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=qty)
        self.submit_order(o)
    def _find_ob(self, side):
        # Last opposite candle before the displacement bar. Full candle range is the OB zone.
        oo=list(self.o); cc=list(self.c); hh=list(self.h); ll=list(self.l)
        end=len(cc)-1
        for j in range(end-1, max(-1,end-1-self.config.ob_search_bars), -1):
            if j<0: break
            bearish=cc[j] < oo[j]; bullish=cc[j] > oo[j]
            if (side>0 and bearish) or (side<0 and bullish):
                return float(ll[j]), float(hh[j])
        return None
    def _try_arm_zone(self):
        n=self.config.breakout_lookback
        if len(self.c)<max(70,n+4) or self._atr()<=0: return
        o=np.asarray(self.o,float); h=np.asarray(self.h,float); l=np.asarray(self.l,float); c=np.asarray(self.c,float)
        atr=self._atr(); prior_hi=float(h[-(n+1):-1].max()); prior_lo=float(l[-(n+1):-1].min())
        body=abs(c[-1]-o[-1])
        side=1 if c[-1]>prior_hi else -1 if c[-1]<prior_lo else 0
        if side==0 or body < self.config.displacement_body_atr*atr: return

        # Strict 3-candle FVG using candle t-2 and current candle t.
        if side>0:
            if not (l[-1] > h[-3]): self.rejected_no_fvg+=1; return
            fvg=(float(h[-3]), float(l[-1]))
        else:
            if not (h[-1] < l[-3]): self.rejected_no_fvg+=1; return
            fvg=(float(h[-1]), float(l[-3]))

        ob=self._find_ob(side)
        if ob is None: self.rejected_no_ob+=1; return
        zlo=max(ob[0],fvg[0]); zhi=min(ob[1],fvg[1])
        if not (zlo < zhi): self.rejected_no_overlap+=1; return

        width=zhi-zlo
        # depth 0 is shallow edge in the direction of the breakout, 100 is deep edge.
        levels={d:(zhi-(d/100.0)*width if side>0 else zlo+(d/100.0)*width) for d in DEPTHS}
        self.zone={'side':side,'lo':zlo,'hi':zhi,'levels':levels,'armed_i':self.m1_i,'atr':atr,'filled':set()}
        self.zone_count+=1

    def on_bar(self, bar:Bar):
        o,h,l,c=map(self._f,[bar.open,bar.high,bar.low,bar.close])
        self.o.append(o); self.h.append(h); self.l.append(l); self.c.append(c); self.m1_i+=1
        tr=max(h-l,abs(h-self.prev_close) if self.prev_close is not None else 0.0,abs(l-self.prev_close) if self.prev_close is not None else 0.0)
        self.tr.append(tr); self.prev_close=c
        if self.zone and self.m1_i-self.zone['armed_i'] > self.config.max_wait_bars:
            self.zone=None
        if self.zone is None:
            self._try_arm_zone()

    def _open_depth(self,d,side,px,atr):
        if self.active[d] is not None:return
        self._submit(side)
        self.active[d]={'side':side,'entry':px,'entry_i':self.m1_i,'atr':atr,'best':px,'worst':px}
        self.stats[d]['fills']+=1
    def _close_depth(self,d,bid,ask,reason):
        a=self.active[d]
        if a is None:return
        side=a['side']; px=bid if side>0 else ask; pnl=(px-a['entry'])*side
        self._submit(-side)
        mfe=max(0.0,(a['best']-a['entry']) if side>0 else (a['entry']-a['best']))
        mae=max(0.0,(a['entry']-a['worst']) if side>0 else (a['worst']-a['entry']))
        rec={'pnl':pnl,'mfe':mfe,'mae':mae,'mfe_atr':mfe/max(a['atr'],1e-9),'mae_atr':mae/max(a['atr'],1e-9),'reason':reason}
        s=self.stats[d]; s['trades'].append(rec)
        if pnl>0:s['wins']+=1;s['gw']+=pnl
        elif pnl<0:s['losses']+=1;s['gl']+=abs(pnl)
        self.active[d]=None

    def on_quote_tick(self,tick:QuoteTick):
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); self.last_bid=bid; self.last_ask=ask
        if self.zone:
            side=self.zone['side']; probe=ask if side>0 else bid
            # Enter independently when price reaches each depth level while still inside/through the overlap zone.
            for d,level in self.zone['levels'].items():
                if d in self.zone['filled']: continue
                touched = probe <= level if side>0 else probe >= level
                if touched:
                    self.zone['filled'].add(d); self._open_depth(d,side,probe,self.zone['atr'])
            # Invalidate if price fully traverses the overlap and continues materially beyond it.
            if (side>0 and bid < self.zone['lo']-0.35*self.zone['atr']) or (side<0 and ask > self.zone['hi']+0.35*self.zone['atr']):
                self.zone=None
        for d,a in list(self.active.items()):
            if a is None: continue
            side=a['side']; mark=bid if side>0 else ask
            if side>0:a['best']=max(a['best'],mark);a['worst']=min(a['worst'],mark)
            else:a['best']=min(a['best'],mark);a['worst']=max(a['worst'],mark)
            move=(mark-a['entry'])*side
            if move>=self.config.tp_atr*a['atr']:
                self._close_depth(d,bid,ask,'TP_FIRST')
            elif move<=-self.config.sl_atr*a['atr']:
                self._close_depth(d,bid,ask,'SL_FIRST')
            elif self.m1_i-a['entry_i']>=self.config.horizon_minutes:
                self._close_depth(d,bid,ask,'HORIZON')

    def on_stop(self):
        if self.last_bid is not None:
            for d in DEPTHS:self._close_depth(d,self.last_bid,self.last_ask,'EOD')
    def summary(self):
        out={'zones':self.zone_count,'reject_no_fvg':self.rejected_no_fvg,'reject_no_ob':self.rejected_no_ob,'reject_no_overlap':self.rejected_no_overlap,'depths':{}}
        for d,s in self.stats.items():
            t=s['trades']; n=len(t); net=sum(x['pnl'] for x in t); pf=s['gw']/s['gl'] if s['gl']>0 else (math.inf if s['gw']>0 else 0.0)
            reasons=Counter(x['reason'] for x in t)
            out['depths'][str(d)]={'N':n,'WR_pct':100*s['wins']/max(n,1),'PF':pf,'net_virtual':net,'expectancy':net/max(n,1),
                'MFE_mean':float(np.mean([x['mfe'] for x in t])) if n else 0.0,'MAE_mean':float(np.mean([x['mae'] for x in t])) if n else 0.0,
                'MFE_ATR_mean':float(np.mean([x['mfe_atr'] for x in t])) if n else 0.0,'MAE_ATR_mean':float(np.mean([x['mae_atr'] for x in t])) if n else 0.0,
                'first_touch':dict(reasons),'fills':s['fills']}
        return out

def ensure_l1(ticks):
    one=Quantity.from_int(1); out=[]; rep=0
    for t in ticks:
        bs=float(t.bid_size.as_double()) if hasattr(t.bid_size,'as_double') else float(t.bid_size); az=float(t.ask_size.as_double()) if hasattr(t.ask_size,'as_double') else float(t.ask_size)
        if bs<=0 or az<=0:
            out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init)); rep+=1
        else: out.append(t)
    return out,rep

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); a=ap.parse_args()
    catalog=ParquetDataCatalog(a.catalog); inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None: raise SystemExit('XAUUSD missing')
    raw=catalog.query_quote_ticks(identifiers=[inst.id.value]);
    if not raw: raise SystemExit('no raw XAUUSD QuoteTicks')
    ticks,replaced=ensure_l1(raw)
    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(inst); engine.add_data(ticks)
    cfg=GateConfig(instrument_id=inst.id,m1=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL'))
    strat=DepthGate(cfg); engine.add_strategy(strat); engine.run(); engine.end()
    result={'verification_level':'NAUTILUS_RAW_BIDASK_OB_FVG_DEPTH_GATE','engine':'NautilusTrader BacktestEngine','nautilus_version':nautilus_trader.__version__,
            'raw_ticks':len(raw),'execution_ticks':len(ticks),'raw_zero_size_quotes_replaced':replaced,'ohlc_resample_used':False,
            'signal_bars':'Nautilus INTERNAL M1 from raw QuoteTicks','poi':'OB_INTERSECT_FVG','depths':[0,25,50,75,100],**strat.summary()}
    out=Path('results/multiedge')/a.experiment_id/'OB_FVG_DEPTH.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
