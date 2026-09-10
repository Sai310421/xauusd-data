from __future__ import annotations
import argparse, json, math
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM=Venue('SIM')

class EVG75Config(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    trigger: float
    add: float
    reversal: float
    trade_size: Decimal
    max_layers: int=10

class EVG75NativeStrategy(Strategy):
    def __init__(self, config: EVG75Config):
        super().__init__(config)
        self.anchor=None; self.side=0; self.last_add=None; self.extreme=None
        self.layers=0; self.entries=0; self.adds=0; self.exits=0
    @staticmethod
    def _f(px):
        return float(px.as_double()) if hasattr(px,'as_double') else float(px)
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)
    def _submit(self, side):
        inst=self.cache.instrument(self.config.instrument_id)
        order=self.order_factory.market(instrument_id=self.config.instrument_id, order_side=side, quantity=inst.make_qty(self.config.trade_size))
        self.submit_order(order)
    def on_quote_tick(self,tick:QuoteTick):
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); mid=(bid+ask)/2.0
        if self.anchor is None:
            self.anchor=mid; return
        if self.side==0:
            if mid>=self.anchor+self.config.trigger:
                self.side=1; self._submit(OrderSide.BUY); self.entries+=1; self.layers=1; self.last_add=ask; self.extreme=bid
            elif mid<=self.anchor-self.config.trigger:
                self.side=-1; self._submit(OrderSide.SELL); self.entries+=1; self.layers=1; self.last_add=bid; self.extreme=ask
            return
        px=bid if self.side>0 else ask
        self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        target=self.last_add+self.side*self.config.add
        if self.layers<self.config.max_layers and ((self.side>0 and px>=target) or (self.side<0 and px<=target)):
            self._submit(OrderSide.BUY if self.side>0 else OrderSide.SELL); self.adds+=1; self.layers+=1; self.last_add=target
        rev=(px<=self.extreme-self.config.reversal) if self.side>0 else (px>=self.extreme+self.config.reversal)
        if rev:
            self.close_all_positions(self.config.instrument_id); self.exits+=1
            self.anchor=mid; self.side=0; self.last_add=None; self.extreme=None; self.layers=0
    def on_stop(self): self.close_all_positions(self.config.instrument_id)

def parse_money(v):
    if v is None:return 0.0
    if isinstance(v,(int,float,np.number)):return float(v)
    try:return float(str(v).replace(',','').split()[0])
    except:return 0.0

def metrics(report, initial=1000.0, days=30):
    if report is None or report.empty:return {'N':0,'WR_pct':0.0,'PF':0.0,'NetProfit':0.0,'MaxDD_pct':0.0,'RF':0.0,'Monthly21_pct':0.0}
    pnl_col=next((c for c in report.columns if 'pnl' in str(c).lower()),None)
    a=np.array([parse_money(x) for x in report[pnl_col]],float) if pnl_col else np.array([],float)
    if len(a)==0:return {'N':0,'WR_pct':0.0,'PF':0.0,'NetProfit':0.0,'MaxDD_pct':0.0,'RF':0.0,'Monthly21_pct':0.0}
    wins=a[a>0]; losses=a[a<0]; pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (math.inf if len(wins) else 0.0)
    eq=initial; peak=initial; mdd=0.0
    for x in a: eq+=x; peak=max(peak,eq); mdd=max(mdd,peak-eq)
    net=float(a.sum()); mdd_pct=mdd/peak*100 if peak>0 else 0.0; monthly=((max(eq,1e-9)/initial)**(21/days)-1)*100
    return {'N':int(len(a)),'WR_pct':float((a>0).mean()*100),'PF':pf,'NetProfit':net,'MaxDD_pct':mdd_pct,'RF':(net/mdd if mdd>0 else None),'Monthly21_pct':monthly}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--trigger',type=float,required=True); ap.add_argument('--add',type=float,required=True); ap.add_argument('--reversal',type=float,required=True); ap.add_argument('--trade-size',default='1'); ap.add_argument('--raw-bidask-only',action='store_true'); a=ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit('RAW_BIDASK_ONLY_REQUIRED')
    cp=Path(a.catalog); man=json.loads((cp/'catalog_manifest.json').read_text()); days=int(man['days']); cat=ParquetDataCatalog(str(cp)); inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'); ticks=list(cat.query(data_cls=QuoteTick, identifiers=[inst.id.value]))
    if not ticks: raise SystemExit('NO_RAW_TICKS')
    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(inst); engine.add_data(ticks)
    strat=EVG75NativeStrategy(EVG75Config(instrument_id=inst.id,trigger=a.trigger,add=a.add,reversal=a.reversal,trade_size=Decimal(a.trade_size)))
    engine.add_strategy(strat); engine.run(); report=engine.trader.generate_positions_report(); m=metrics(report,days=days)
    out=Path('results/ae-bt')/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    report.to_csv(out/'positions.csv',index=False)
    summary={'verification_level':'NAUTILUS_NATIVE_ORDER_RAW_BIDASK','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'params':{'trigger':a.trigger,'add':a.add,'reversal':a.reversal,'trade_size':a.trade_size},'raw_ticks':len(ticks),'signals':{'entries':strat.entries,'adds':strat.adds,'exits':strat.exits},'metrics':m,'execution':'MARKET orders on raw Bid/Ask; observed spread native; commission/slippage not yet added','period':man}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary))
    engine.dispose()
if __name__=='__main__': main()
