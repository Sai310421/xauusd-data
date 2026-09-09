from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
SIM=Venue('SIM')
class Cfg(StrategyConfig,frozen=True):
    instrument_id:InstrumentId
    max_entries_per_day:int
    initial_balance:float=1000.0
    trigger:float=0.12
    reversal:float=0.20
    per_entry_risk_pct:float=0.05
    dd_limit:float=3.0
class S(Strategy):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.anchor=None;self.active=False;self.side=0;self.entry=None;self.extreme=None
        self.real=0.;self.peak=cfg.initial_balance;self.maxdd=0.;self.gw=self.gl=0.;self.wins=self.cycles=0
        self.day=None;self.entries_today=0;self.total_entries=0;self.rejected_by_cap=0;self.dd_stops=0
        self.last_bid=self.last_ask=None
        self.m1_bucket=None;self.m1_close=None
    @staticmethod
    def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
    def mark(self,bid,ask):
        if not self.active:return 0.
        px=bid if self.side>0 else ask
        return (px-self.entry)*self.side
    def dd(self,bid,ask):
        eq=self.config.initial_balance+self.real+self.mark(bid,ask)
        self.peak=max(self.peak,eq)
        d=max(0.,(self.peak-eq)/max(self.peak,1e-9)*100)
        self.maxdd=max(self.maxdd,d)
        return d
    def close(self,bid,ask):
        if not self.active:return
        px=bid if self.side>0 else ask
        p=(px-self.entry)*self.side
        self.real+=p;self.cycles+=1
        if p>0:self.wins+=1;self.gw+=p
        elif p<0:self.gl+=abs(p)
        self.active=False;self.side=0;self.entry=None;self.extreme=None
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);mid=(bid+ask)/2
        self.last_bid=bid;self.last_ask=ask
        sec=int(t.ts_event)//1_000_000_000
        day=sec//86400
        if self.day is None or day!=self.day:
            self.day=day;self.entries_today=0
        b=sec//60
        if self.m1_bucket is None:self.m1_bucket=b
        elif b!=self.m1_bucket:
            self.anchor=self.m1_close if self.m1_close is not None else mid
            self.m1_bucket=b
        self.m1_close=mid
        if self.anchor is None:return
        d=self.dd(bid,ask)
        if d>=self.config.dd_limit:
            if self.active:self.close(bid,ask)
            self.dd_stops+=1
            return
        if self.active:
            px=bid if self.side>0 else ask
            self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
            rev=px<=self.extreme-self.config.reversal if self.side>0 else px>=self.extreme+self.config.reversal
            if rev:self.close(bid,ask)
            return
        up=mid>=self.anchor+self.config.trigger;dn=mid<=self.anchor-self.config.trigger
        if not (up or dn):return
        if self.entries_today>=self.config.max_entries_per_day:
            self.rejected_by_cap+=1;return
        sig=1 if up else -1
        self.side=sig;self.entry=ask if sig>0 else bid;self.extreme=bid if sig>0 else ask
        self.active=True;self.entries_today+=1;self.total_entries+=1
    def on_stop(self):
        if self.active and self.last_bid is not None:self.close(self.last_bid,self.last_ask)
    def summary(self):
        pf=self.gw/self.gl if self.gl else (math.inf if self.gw else 0.)
        return dict(max_entries_per_day=self.config.max_entries_per_day,cycles=self.cycles,total_entries=self.total_entries,WR_pct=100*self.wins/max(1,self.cycles),PF=pf,realized_usd_0p01lot_equiv=self.real,return_pct_on_1000=self.real/10,max_DD_pct=self.maxdd,rejected_by_cap=self.rejected_by_cap,dd_stops=self.dd_stops)
def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--experiment-id',required=True);p.add_argument('--max-entries-per-day',type=int,required=True);p.add_argument('--raw-bidask-only',action='store_true');a=p.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks)
    s=S(Cfg(instrument_id=inst.id,max_entries_per_day=a.max_entries_per_day));eng.add_strategy(s);eng.run()
    r={**s.summary(),'raw_ticks':len(ticks),'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'period_start':man.get('start'),'period_days':man.get('days'),'period_end_exclusive':man.get('end_exclusive'),'chronology':'RAW_BIDASK_OPPORTUNITY_SATURATION_V1'}
    out=Path('results/ae-bt')/a.experiment_id/'cells';out.mkdir(parents=True,exist_ok=True);(out/f'N{a.max_entries_per_day}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));eng.dispose()
if __name__=='__main__':main()
