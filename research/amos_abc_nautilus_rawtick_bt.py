"""A01/B01/C01 research-only Nautilus Raw Bid/Ask backtest.

Uses existing XAUUSD ParquetDataCatalog and BacktestEngine. Signals from completed
M1 BID candles derived causally from Raw QuoteTicks; entry/exit marked on actual
Bid/Ask ticks. Fills are VIRTUAL (not Nautilus order matching), clearly labeled.
No live orders. Five completed M1 bars hold, fixed research-only exit.
"""
from __future__ import annotations
import argparse, json, math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float

def signals(bars):
    if len(bars)<31:return []
    b=list(bars); last=b[-1]; prev=b[-21:-1]
    sma20=mean(x.close for x in prev)
    sma5=mean(x.close for x in b[-6:-1])
    atr=mean(max(x.high-x.low,abs(x.high-y.close),abs(x.low-y.close))
             for y,x in zip(b[-15:-1],b[-14:]))
    if atr<=0:return []
    out=[]
    if last.close<sma20-1.5*atr:out.append(('A01',1))
    elif last.close>sma20+1.5*atr:out.append(('A01',-1))
    if last.close>max(x.high for x in prev) and sma5>=sma20:out.append(('B01',1))
    elif last.close<min(x.low for x in prev) and sma5<=sma20:out.append(('B01',-1))
    recent=b[-6:-1]
    if mean(x.high-x.low for x in recent)<.75*atr and last.high-last.low>1.3*atr:
        if last.close>last.open and last.close>max(x.high for x in recent):out.append(('C01',1))
        elif last.close<last.open and last.close<min(x.low for x in recent):out.append(('C01',-1))
    return out

class Config(StrategyConfig,frozen=True):
    instrument_id:object
    hold_bars:int=5
    units_per_lot:float=100.0
    lot:float=0.01
    commission_per_lot_roundtrip:float=0.0
    initial_usd:float=1000.0

class ABC(Strategy):
    def __init__(self,c):
        super().__init__(c)
        self.bars=deque(maxlen=64);self.bucket=None;self.cur=None
        self.positions={k:None for k in ('A01','B01','C01')}
        self.closed={k:[] for k in self.positions}
        self.last_bid=self.last_ask=None
        self.equity_peak=c.initial_usd;self.max_floating_dd=0.
        self.realized=0.;self.ticks=0;self.first_ns=None;self.last_ns=None
    def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
    def _mark(self,bid,ask):
        return sum(((bid if p['side']>0 else ask)-p['entry'])*p['side']*self.config.lot*self.config.units_per_lot
                   for p in self.positions.values() if p)
    def _dd(self,bid,ask):
        equity=self.config.initial_usd+self.realized+self._mark(bid,ask)
        self.equity_peak=max(self.equity_peak,equity)
        self.max_floating_dd=max(self.max_floating_dd,(self.equity_peak-equity)/max(self.equity_peak,1e-9)*100)
    def _exit(self,k,bid,ask):
        p=self.positions[k]
        if p is None:return
        px=bid if p['side']>0 else ask
        gross=(px-p['entry'])*p['side']*self.config.lot*self.config.units_per_lot
        net=gross-self.config.commission_per_lot_roundtrip*self.config.lot
        self.realized+=net
        self.closed[k].append({'entry_ns':p['entry_ns'],'exit_ns':self.last_ns,'side':p['side'],
            'gross_usd':gross,'net_usd':net,'mae_usd':p['mae'],'mfe_usd':p['mfe']})
        self.positions[k]=None
    def _closed_bar(self,bid,ask):
        if self.cur is None:return
        self.bars.append(self.cur)
        for k,p in self.positions.items():
            if p:
                p['age']+=1
                if p['age']>=self.config.hold_bars:self._exit(k,bid,ask)
        for k,side in signals(self.bars):
            if self.positions[k] is None:
                self.positions[k]={'side':side,'entry':ask if side>0 else bid,'entry_ns':self.last_ns,
                    'age':0,'mae':0.,'mfe':0.}
    def on_quote_tick(self,t:QuoteTick):
        bid=float(t.bid_price);ask=float(t.ask_price)
        if bid<=0 or ask<bid:raise ValueError('Invalid Raw Bid/Ask')
        ns=int(t.ts_event);bucket=ns//60_000_000_000
        if self.last_ns is not None and ns<self.last_ns:raise ValueError('Nonmonotonic ticks')
        if self.first_ns is None:self.first_ns=ns
        self.last_ns=ns;self.ticks+=1
        if self.bucket is not None and bucket!=self.bucket:
            self._closed_bar(bid,ask)
            self.cur=None
        self.bucket=bucket
        if self.cur is None:self.cur=Candle(bid,bid,bid,bid)
        else:
            self.cur.high=max(self.cur.high,bid)
            self.cur.low=min(self.cur.low,bid)
            self.cur.close=bid
        for p in self.positions.values():
            if p:
                move=((bid if p['side']>0 else ask)-p['entry'])*p['side']*self.config.lot*self.config.units_per_lot
                p['mae']=min(p['mae'],move);p['mfe']=max(p['mfe'],move)
        self.last_bid=bid;self.last_ask=ask;self._dd(bid,ask)
    def on_stop(self):
        if self.last_bid is not None:
            for k in self.positions:self._exit(k,self.last_bid,self.last_ask)
    def report(self):
        out={}
        for k,trades in self.closed.items():
            gains=sum(t['net_usd'] for t in trades if t['net_usd']>0)
            losses=-sum(t['net_usd'] for t in trades if t['net_usd']<0)
            n=len(trades)
            out[k]={'N':n,'WR_pct':100*sum(t['net_usd']>0 for t in trades)/n if n else None,
                'PF':gains/losses if losses else (None if not gains else 'inf'),
                'net_usd':sum(t['net_usd'] for t in trades),
                'EV_usd':sum(t['net_usd'] for t in trades)/n if n else None,
                'mean_MAE_usd':mean(t['mae_usd'] for t in trades) if n else None,
                'mean_MFE_usd':mean(t['mfe_usd'] for t in trades) if n else None}
        return {'verification':'NAUTILUS_RAW_QUOTETICK_VIRTUAL_EXECUTION','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
            'raw_ticks':self.ticks,'first_ns':self.first_ns,'last_ns':self.last_ns,
            'max_floating_dd_pct_combined':self.max_floating_dd,
            'execution_limitations':['Virtual Bid/Ask fills, not Nautilus order matching','Research exit after 5 completed M1 bars',
            'Commission configurable; no slippage modeled','No OOS/WFO/MC yet'],
            'edges':out}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',required=True);p.add_argument('--out',default='results/amos-abc/rawtick-kpi.json')
    p.add_argument('--commission-per-lot-roundtrip',type=float,default=0.)
    a=p.parse_args()
    cat=ParquetDataCatalog(a.catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    raw=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    if not raw:raise SystemExit('No QuoteTicks: refusing synthetic OHLC fallback')
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,
                  base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(raw)
    s=ABC(Config(instrument_id=inst.id,commission_per_lot_roundtrip=a.commission_per_lot_roundtrip))
    eng.add_strategy(s);eng.run();eng.end()
    result=s.report()
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    eng.dispose()

if __name__=='__main__':main()
