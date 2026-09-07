from __future__ import annotations
import argparse,json,math
from pathlib import Path
from decimal import Decimal
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId

SIM=Venue('SIM'); TF_SEC={'M1':60,'M5':300,'M15':900}

class Cfg(StrategyConfig,frozen=True):
    instrument_id:InstrumentId; tf_sec:int; variant:str
    initial_balance:float=1000.0; trigger:float=.12; add:float=.025; reversal:float=.20; max_layers:int=10
    soft_dd_pct:float=1.5; lock_dd_pct:float=3.0; hard_dd_pct:float=4.5; recovery_alloc:float=.70

class G75CausalRaw(Strategy):
    """Causal G75 translation.
    - anchor = previous completed bar close
    - trigger direction is the direction actually crossed first (no future close lookahead)
    - entries/adds use ordered raw Bid/Ask quotes
    - running extreme uses executable side mark (bid for longs, ask for shorts)
    - reversal decision only at completed bar close, preserving canonical exit cadence
    - at most one new basket start per bar, preserving canonical event cadence
    """
    def __init__(self,cfg:Cfg):
        super().__init__(cfg)
        self.bucket=None;self.anchor=None;self.bar_close_mid=None;self.started_bucket=False
        self.active=False;self.side=0;self.entries=[];self.last_add=None;self.extreme=None
        self.realized=0.;self.peak=cfg.initial_balance;self.maxdd=0.;self.cycles=self.wins=self.losses=self.adds=0;self.gw=self.gl=0.;self.maxlayers=0
        self.mode='NORMAL';self.debt=0.;self.maxdebt=0.;self.soft=self.locks=self.hard=self.rec_cycles=self.rec_success=0;self.rec_hist=[];self.stopped=False;self.clear_bars=[];self.lock_bucket=None
        self.last_bid=self.last_ask=None
    @staticmethod
    def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)
    def mark(self,bid,ask):
        if not self.active:return 0.
        px=bid if self.side>0 else ask
        return sum((px-e)*self.side for e in self.entries)
    def dd(self,bid,ask):
        eq=self.config.initial_balance+self.realized+self.mark(bid,ask);self.peak=max(self.peak,eq);d=max(0.,(self.peak-eq)/max(self.peak,1e-9)*100);self.maxdd=max(self.maxdd,d);return d
    def cap(self):return max(1,self.config.max_layers//2) if self.config.variant!='A' and self.mode in ('REDUCED','RECOVERY') else self.config.max_layers
    def rec_ok(self):
        if self.mode!='RECOVERY' or len(self.rec_hist)<5:return True
        h=self.rec_hist[-20:];return sum(x for x in h if x>0)>abs(sum(x for x in h if x<0))
    def close(self,bid,ask,reason):
        if not self.active:return
        px=bid if self.side>0 else ask;p=sum((px-e)*self.side for e in self.entries);self.realized+=p;self.cycles+=1
        if p>0:self.wins+=1;self.gw+=p
        elif p<0:self.losses+=1;self.gl+=abs(p)
        if self.mode=='RECOVERY':
            self.rec_cycles+=1;self.rec_hist.append(p)
            if p>0 and self.debt>0:
                self.debt=max(0.,self.debt-p*self.config.recovery_alloc)
                if self.debt<=1e-12:self.rec_success+=1;self.mode='NORMAL'
        self.active=False;self.side=0;self.entries=[];self.last_add=None;self.extreme=None
    def lock(self,bid,ask):
        locked=max(0.,-self.mark(bid,ask));self.debt+=locked;self.maxdebt=max(self.maxdebt,self.debt);self.locks+=1
        self.close(bid,ask,'LOCK');self.mode='RECOVERY'
    def finish_bar(self):
        if self.bar_close_mid is None:return
        bid=self.last_bid;ask=self.last_ask
        if self.active:
            close_mark=bid if self.side>0 else ask
            rev=(close_mark<=self.extreme-self.config.reversal) if self.side>0 else (close_mark>=self.extreme+self.config.reversal)
            if rev:self.close(bid,ask,'REVERSAL_CLOSE')
        self.anchor=self.bar_close_mid
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);mid=(bid+ask)/2;self.last_bid=bid;self.last_ask=ask
        ns=int(t.ts_event);b=(ns//1_000_000_000)//self.config.tf_sec
        if self.bucket is None:self.bucket=b;self.anchor=mid
        elif b!=self.bucket:
            self.finish_bar();self.bucket=b;self.started_bucket=False
        self.bar_close_mid=mid
        if self.stopped:return
        d=self.dd(bid,ask)
        if self.config.variant!='A':
            if d>=self.config.hard_dd_pct:
                self.hard+=1
                if self.active:self.close(bid,ask,'HARD')
                self.stopped=True;self.mode='STOPPED';return
            if self.config.variant=='C' and self.active and self.mode!='RECOVERY' and d>=self.config.lock_dd_pct:
                self.lock(bid,ask);return
            if self.mode not in ('RECOVERY','STOPPED'):
                if d>=self.config.soft_dd_pct:
                    if self.mode!='REDUCED':self.soft+=1
                    self.mode='REDUCED'
                else:self.mode='NORMAL'
        if not self.active and not self.started_bucket and self.anchor is not None:
            if self.mode=='RECOVERY' and not self.rec_ok():return
            if mid>=self.anchor+self.config.trigger:
                self.side=1;entry=ask
            elif mid<=self.anchor-self.config.trigger:
                self.side=-1;entry=bid
            else:return
            self.active=True;self.entries=[entry];self.last_add=entry;self.extreme=(bid if self.side>0 else ask);self.started_bucket=True;self.maxlayers=max(self.maxlayers,1)
        if not self.active:return
        px=bid if self.side>0 else ask
        self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        cap=self.cap()
        while len(self.entries)<cap:
            target=self.last_add+self.side*self.config.add;cross=(px>=target if self.side>0 else px<=target)
            if not cross:break
            fill=ask if self.side>0 else bid;self.entries.append(fill);self.last_add=target;self.adds+=1;self.maxlayers=max(self.maxlayers,len(self.entries))
        self.dd(bid,ask)
    def on_stop(self):
        self.finish_bar()
        if self.active and self.last_bid is not None:self.close(self.last_bid,self.last_ask,'EOD')
    def summary(self):
        pf=self.gw/self.gl if self.gl else (math.inf if self.gw else 0.)
        return {'variant':self.config.variant,'cycles':self.cycles,'WR_pct':100*self.wins/max(1,self.cycles),'PF':pf,'realized_usd_0p01lot_equiv':self.realized,'return_pct_on_1000':self.realized/10,'max_DD_pct':self.maxdd,'adds':self.adds,'max_layers':self.maxlayers,'mode_final':self.mode,'debt_final':self.debt,'max_debt':self.maxdebt,'soft_events':self.soft,'lock_events':self.locks,'hard_events':self.hard,'recovery_cycles':self.rec_cycles,'recovery_success':self.rec_success,'stopped':self.stopped}

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--experiment-id',required=True);p.add_argument('--tf',choices=TF_SEC,required=True);p.add_argument('--variant',choices=['A','B','C'],required=True);p.add_argument('--raw-bidask-only',action='store_true');a=p.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    s=G75CausalRaw(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant=a.variant));eng.add_strategy(s);eng.run();r={**s.summary(),'tf':a.tf,'raw_ticks':len(ticks),'chronology':'CAUSAL_RAW_BIDASK_TRIGGER_DIRECTION_CLOSE_REVERSAL','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'period_start':man.get('start'),'period_days':man.get('days'),'period_end_exclusive':man.get('end_exclusive')};out=Path('results/ae-bt')/a.experiment_id/'cells';out.mkdir(parents=True,exist_ok=True);(out/f'{a.tf}_{a.variant}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));eng.dispose()
if __name__=='__main__':main()
