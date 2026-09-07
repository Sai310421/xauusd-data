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
    instrument_id:InstrumentId; gate:str; variant:str
    initial_balance:float=1000.;trigger:float=.12;add:float=.025;reversal:float=.20;max_layers:int=10
    confirm1:float=.025;confirm2:float=.05;pullback_max:float=.04
    soft_dd:float=2.5;lock_dd:float=3.5;hard_dd:float=4.5
class S(Strategy):
    def __init__(self,cfg):
        super().__init__(cfg);self.bucket=None;self.anchor=None;self.bar_close=None;self.started=False
        self.pending_side=0;self.trigger_px=None;self.pending_extreme=None;self.mids=deque(maxlen=16)
        self.active=False;self.side=0;self.entries=[];self.last_add=None;self.extreme=None
        self.real=0.;self.peak=cfg.initial_balance;self.maxdd=0.;self.cycles=self.wins=self.losses=self.adds=0;self.gw=self.gl=0.;self.maxlayers=0
        self.mode='NORMAL';self.debt=0.;self.maxdebt=0.;self.locks=self.rec_cycles=self.rec_success=0;self.recovery_earned=0.;self.soft_events=self.hard_events=0;self.stopped=False
        self.candidates=self.rejects=self.confirms=0;self.last_bid=self.last_ask=None
    @staticmethod
    def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
    def mark(self,bid,ask):
        if not self.active:return 0.
        px=bid if self.side>0 else ask
        return sum((px-e)*self.side for e in self.entries)
    def dd(self,bid,ask):
        eq=self.config.initial_balance+self.real+self.mark(bid,ask);self.peak=max(self.peak,eq);d=max(0.,(self.peak-eq)/max(self.peak,1e-9)*100);self.maxdd=max(self.maxdd,d);return d
    def close(self,bid,ask,reason):
        if not self.active:return
        px=bid if self.side>0 else ask;p=sum((px-e)*self.side for e in self.entries);self.real+=p;self.cycles+=1
        if p>0:self.wins+=1;self.gw+=p
        elif p<0:self.losses+=1;self.gl+=abs(p)
        if self.mode=='RECOVERY':
            self.rec_cycles+=1
            if p>0 and self.debt>0:
                pay=min(self.debt,p);self.debt-=pay;self.recovery_earned+=pay
                if self.debt<=1e-9:self.debt=0.;self.rec_success+=1;self.mode='NORMAL'
        self.active=False;self.side=0;self.entries=[];self.last_add=None;self.extreme=None
    def lock(self,bid,ask):
        loss=max(0.,-self.mark(bid,ask));self.debt+=loss;self.maxdebt=max(self.maxdebt,self.debt);self.locks+=1;self.close(bid,ask,'LOCK');self.mode='RECOVERY'
    def finish_bar(self,bid,ask):
        if self.bar_close is None:return
        if self.active:
            px=bid if self.side>0 else ask;rev=px<=self.extreme-self.config.reversal if self.side>0 else px>=self.extreme+self.config.reversal
            if rev:self.close(bid,ask,'REV')
        self.anchor=self.bar_close;self.bar_close=None;self.started=False;self.pending_side=0;self.trigger_px=None;self.pending_extreme=None
    def confirm_needed(self):return self.config.confirm1 if self.config.gate=='C1' else self.config.confirm2
    def velocity_ok(self,side):
        if len(self.mids)<6:return False
        return (self.mids[-1]-self.mids[-6])*side>0
    def pending_ok(self,mid):
        s=self.pending_side;need=self.confirm_needed()
        self.pending_extreme=max(self.pending_extreme,mid) if s>0 else min(self.pending_extreme,mid)
        extension=(self.pending_extreme-self.trigger_px)*s
        pullback=(self.pending_extreme-mid)*s
        if pullback>self.config.pullback_max:return -1
        if extension<need:return 0
        if self.config.gate=='C3' and not self.velocity_ok(s):return 0
        return 1
    def start(self,bid,ask):
        s=self.pending_side;entry=ask if s>0 else bid;self.side=s;self.active=True;self.entries=[entry];self.last_add=entry;self.extreme=bid if s>0 else ask;self.started=True;self.confirms+=1
        self.pending_side=0;self.trigger_px=None;self.pending_extreme=None;self.maxlayers=max(self.maxlayers,1)
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);mid=(bid+ask)/2;self.last_bid=bid;self.last_ask=ask;self.mids.append(mid);sec=int(t.ts_event)//1_000_000_000;b=sec//60
        if self.bucket is None:self.bucket=b;self.anchor=mid
        elif b!=self.bucket:self.finish_bar(bid,ask);self.bucket=b
        self.bar_close=mid
        if self.stopped or self.anchor is None:return
        d=self.dd(bid,ask)
        if d>=self.config.hard_dd:
            self.hard_events+=1
            if self.active:self.close(bid,ask,'HARD')
            self.stopped=True;self.mode='STOPPED';return
        if self.config.variant=='C' and self.active and self.mode!='RECOVERY' and d>=self.config.lock_dd:
            self.lock(bid,ask);return
        if self.mode not in ('RECOVERY','STOPPED'):
            if d>=self.config.soft_dd:
                if self.mode!='REDUCED':self.soft_events+=1
                self.mode='REDUCED'
            else:self.mode='NORMAL'
        if not self.active and not self.started:
            if self.pending_side==0:
                up=mid>=self.anchor+self.config.trigger;dn=mid<=self.anchor-self.config.trigger
                if up or dn:
                    self.candidates+=1;self.pending_side=1 if up else -1;self.trigger_px=mid;self.pending_extreme=mid
            else:
                q=self.pending_ok(mid)
                if q<0:self.rejects+=1;self.started=True;self.pending_side=0
                elif q>0:self.start(bid,ask)
        if not self.active:return
        px=bid if self.side>0 else ask;self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        cap=3 if self.mode=='RECOVERY' else (5 if self.mode=='REDUCED' else self.config.max_layers)
        while len(self.entries)<cap:
            target=self.last_add+self.side*self.config.add;cross=px>=target if self.side>0 else px<=target
            if not cross:break
            fill=ask if self.side>0 else bid;self.entries.append(fill);self.last_add=target;self.adds+=1;self.maxlayers=max(self.maxlayers,len(self.entries))
        self.dd(bid,ask)
    def on_stop(self):
        if self.active and self.last_bid is not None:self.close(self.last_bid,self.last_ask,'EOD')
    def summary(self):
        pf=self.gw/self.gl if self.gl else (math.inf if self.gw else 0.)
        return dict(gate=self.config.gate,variant=self.config.variant,cycles=self.cycles,WR_pct=100*self.wins/max(1,self.cycles),PF=pf,realized_usd_0p01lot_equiv=self.real,return_pct_on_1000=self.real/10,max_DD_pct=self.maxdd,adds=self.adds,max_layers=self.maxlayers,debt_final=self.debt,max_debt=self.maxdebt,lock_events=self.locks,recovery_cycles=self.rec_cycles,recovery_success=self.rec_success,recovery_earned=self.recovery_earned,soft_events=self.soft_events,hard_events=self.hard_events,trigger_candidates=self.candidates,direction_rejects=self.rejects,direction_confirms=self.confirms,stopped=self.stopped)
def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--experiment-id',required=True);p.add_argument('--gate',choices=['C1','C2','C3'],required=True);p.add_argument('--variant',choices=['B','C'],required=True);p.add_argument('--raw-bidask-only',action='store_true');a=p.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    s=S(Cfg(instrument_id=inst.id,gate=a.gate,variant=a.variant));eng.add_strategy(s);eng.run();r={**s.summary(),'raw_ticks':len(ticks),'chronology':'M1_RAW_TRIGGER_CONTINUATION_CONFIRM','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'period_start':man.get('start'),'period_days':man.get('days'),'period_end_exclusive':man.get('end_exclusive')};out=Path('results/ae-bt')/a.experiment_id/'cells';out.mkdir(parents=True,exist_ok=True);(out/f'{a.gate}_{a.variant}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));eng.dispose()
if __name__=='__main__':main()
