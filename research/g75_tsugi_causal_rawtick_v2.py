from __future__ import annotations
import argparse, json, math
from collections import deque
from pathlib import Path
from decimal import Decimal
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId

SIM=Venue('SIM')
TF_SEC={'M1':60,'M5':300,'M15':900}

class Cfg(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    tf_sec: int
    variant: str
    initial_balance: float=1000.0
    trigger: float=.12
    confirm: float=.03
    add: float=.025
    reversal: float=.20
    max_layers: int=10
    soft_dd_pct: float=1.5
    lock_dd_pct: float=3.0
    hard_dd_pct: float=4.5
    cooldown_bars: int=6
    recovery_alloc: float=1.0
    velocity_ticks: int=8

class G75TsugiV2(Strategy):
    """Causal Raw Bid/Ask G75 + TSUGI v2.

    Frozen price core: trigger=.12, add=.025, reversal=.20.
    Changes are supervisory only:
    - causal direction confirmation (no future close)
    - hard-DD cooldown/re-arm instead of permanent stop
    - risk-epoch reset after hard event (account loss is NOT erased)
    - C lock -> debt ledger -> reduced-layer recovery -> debt clear
    - recovery cannot pyramid aggressively
    """
    def __init__(self,cfg:Cfg):
        super().__init__(cfg)
        self.bucket=None; self.anchor=None; self.bar_close_mid=None; self.started_bucket=False
        self.prev_closes=deque(maxlen=4); self.tick_mids=deque(maxlen=max(3,cfg.velocity_ticks))
        self.active=False; self.side=0; self.entries=[]; self.last_add=None; self.extreme=None
        self.realized=0.; self.account_peak=cfg.initial_balance; self.max_account_dd=0.
        self.risk_peak=cfg.initial_balance; self.max_epoch_dd=0.
        self.cycles=self.wins=self.losses=self.adds=0; self.gw=self.gl=0.; self.maxlayers=0
        self.mode='NORMAL'; self.debt=0.; self.maxdebt=0.; self.recovery_earned=0.
        self.soft=self.locks=self.hard=self.cooldowns=self.rearms=self.rec_cycles=self.rec_success=0
        self.cooldown_until_bucket=None; self.last_bid=self.last_ask=None
        self.direction_rejects=0; self.trigger_candidates=0
    @staticmethod
    def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)
    def mark(self,bid,ask):
        if not self.active:return 0.
        px=bid if self.side>0 else ask
        return sum((px-e)*self.side for e in self.entries)
    def equity(self,bid,ask): return self.config.initial_balance+self.realized+self.mark(bid,ask)
    def dd(self,bid,ask):
        eq=self.equity(bid,ask)
        self.account_peak=max(self.account_peak,eq)
        ad=max(0.,(self.account_peak-eq)/max(self.account_peak,1e-9)*100.)
        self.max_account_dd=max(self.max_account_dd,ad)
        self.risk_peak=max(self.risk_peak,eq)
        rd=max(0.,(self.risk_peak-eq)/max(self.risk_peak,1e-9)*100.)
        self.max_epoch_dd=max(self.max_epoch_dd,rd)
        return rd
    def reset_risk_epoch(self,bid,ask): self.risk_peak=max(1e-9,self.equity(bid,ask))
    def cap(self):
        if self.config.variant=='A': return self.config.max_layers
        if self.mode=='RECOVERY': return 3
        if self.mode=='REDUCED': return max(2,self.config.max_layers//2)
        return self.config.max_layers
    def direction_score(self,side,mid):
        score=0
        # 1) true causal overshoot beyond trigger: avoids touch-and-fade noise.
        if side>0 and mid>=self.anchor+self.config.trigger+self.config.confirm: score+=1
        if side<0 and mid<=self.anchor-self.config.trigger-self.config.confirm: score+=1
        # 2) tick velocity/continuation from already observed quotes only.
        if len(self.tick_mids)>=3:
            d=self.tick_mids[-1]-self.tick_mids[0]
            if d*side>0: score+=1
        # 3) prior completed-bar momentum only; never current future close.
        if len(self.prev_closes)>=2:
            d=self.prev_closes[-1]-self.prev_closes[-2]
            if d*side>0: score+=1
        return score
    def close(self,bid,ask,reason):
        if not self.active:return 0.
        px=bid if self.side>0 else ask
        p=sum((px-e)*self.side for e in self.entries)
        self.realized+=p; self.cycles+=1
        if p>0:self.wins+=1;self.gw+=p
        elif p<0:self.losses+=1;self.gl+=abs(p)
        if self.mode=='RECOVERY':
            self.rec_cycles+=1
            if p>0 and self.debt>0:
                credit=p*self.config.recovery_alloc
                self.recovery_earned+=credit
                old=self.debt; self.debt=max(0.,self.debt-credit)
                if old>0 and self.debt<=1e-12:
                    self.rec_success+=1; self.mode='NORMAL'; self.reset_risk_epoch(bid,ask)
        self.active=False;self.side=0;self.entries=[];self.last_add=None;self.extreme=None
        return p
    def lock(self,bid,ask):
        locked=max(0.,-self.mark(bid,ask))
        self.debt+=locked; self.maxdebt=max(self.maxdebt,self.debt); self.locks+=1
        self.close(bid,ask,'LOCK')
        self.mode='RECOVERY'
        self.reset_risk_epoch(bid,ask)
    def hard_reset(self,bid,ask):
        self.hard+=1
        if self.active:self.close(bid,ask,'HARD')
        self.mode='COOLDOWN'; self.cooldowns+=1
        self.cooldown_until_bucket=(self.bucket or 0)+self.config.cooldown_bars
        self.reset_risk_epoch(bid,ask)
    def finish_bar(self):
        if self.bar_close_mid is None:return
        bid=self.last_bid;ask=self.last_ask
        if self.active:
            close_mark=bid if self.side>0 else ask
            rev=(close_mark<=self.extreme-self.config.reversal) if self.side>0 else (close_mark>=self.extreme+self.config.reversal)
            if rev:self.close(bid,ask,'REVERSAL_CLOSE')
        self.prev_closes.append(self.bar_close_mid)
        self.anchor=self.bar_close_mid
    def maybe_rearm(self,bid,ask):
        if self.mode!='COOLDOWN':return
        if self.cooldown_until_bucket is not None and self.bucket>=self.cooldown_until_bucket:
            self.rearms+=1
            self.mode='RECOVERY' if self.debt>0 else 'NORMAL'
            self.reset_risk_epoch(bid,ask)
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);mid=(bid+ask)/2
        self.last_bid=bid;self.last_ask=ask;self.tick_mids.append(mid)
        ns=int(t.ts_event); b=(ns//1_000_000_000)//self.config.tf_sec
        if self.bucket is None:self.bucket=b;self.anchor=mid
        elif b!=self.bucket:
            self.finish_bar();self.bucket=b;self.started_bucket=False
        self.bar_close_mid=mid
        self.maybe_rearm(bid,ask)
        if self.mode=='COOLDOWN':
            self.dd(bid,ask);return
        rd=self.dd(bid,ask)
        if self.config.variant!='A':
            if rd>=self.config.hard_dd_pct:
                self.hard_reset(bid,ask);return
            if self.config.variant=='C' and self.active and self.mode not in ('RECOVERY','COOLDOWN') and rd>=self.config.lock_dd_pct:
                self.lock(bid,ask);return
            if self.mode not in ('RECOVERY','COOLDOWN'):
                if rd>=self.config.soft_dd_pct:
                    if self.mode!='REDUCED':self.soft+=1
                    self.mode='REDUCED'
                else:self.mode='NORMAL'
        if not self.active and not self.started_bucket and self.anchor is not None:
            side=0
            if mid>=self.anchor+self.config.trigger: side=1
            elif mid<=self.anchor-self.config.trigger: side=-1
            else:return
            self.trigger_candidates+=1
            needed=1 if self.config.variant=='A' else (3 if self.mode=='RECOVERY' else 2)
            if self.config.variant!='A' and self.direction_score(side,mid)<needed:
                self.direction_rejects+=1; return
            entry=ask if side>0 else bid
            self.side=side;self.active=True;self.entries=[entry];self.last_add=entry
            self.extreme=(bid if side>0 else ask);self.started_bucket=True;self.maxlayers=max(self.maxlayers,1)
        if not self.active:return
        px=bid if self.side>0 else ask
        self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        cap=self.cap()
        while len(self.entries)<cap:
            target=self.last_add+self.side*self.config.add
            cross=px>=target if self.side>0 else px<=target
            if not cross:break
            # In B/C, avoid adding if observed tick velocity has already flipped.
            if self.config.variant!='A' and len(self.tick_mids)>=3 and (self.tick_mids[-1]-self.tick_mids[0])*self.side<=0:break
            fill=ask if self.side>0 else bid
            self.entries.append(fill);self.last_add=target;self.adds+=1;self.maxlayers=max(self.maxlayers,len(self.entries))
        self.dd(bid,ask)
    def on_stop(self):
        self.finish_bar()
        if self.active and self.last_bid is not None:self.close(self.last_bid,self.last_ask,'EOD')
    def summary(self):
        pf=self.gw/self.gl if self.gl else (math.inf if self.gw else 0.)
        return {
            'variant':self.config.variant,'cycles':self.cycles,'WR_pct':100*self.wins/max(1,self.cycles),'PF':pf,
            'realized_usd_0p01lot_equiv':self.realized,'return_pct_on_1000':self.realized/10,
            'max_account_DD_pct':self.max_account_dd,'max_risk_epoch_DD_pct':self.max_epoch_dd,
            'adds':self.adds,'max_layers':self.maxlayers,'mode_final':self.mode,'debt_final':self.debt,'max_debt':self.maxdebt,
            'recovery_earned':self.recovery_earned,'soft_events':self.soft,'lock_events':self.locks,'hard_events':self.hard,
            'cooldowns':self.cooldowns,'rearms':self.rearms,'recovery_cycles':self.rec_cycles,'recovery_success':self.rec_success,
            'trigger_candidates':self.trigger_candidates,'direction_rejects':self.direction_rejects,
        }

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--experiment-id',required=True)
    p.add_argument('--tf',choices=TF_SEC,required=True);p.add_argument('--variant',choices=['A','B','C'],required=True)
    p.add_argument('--raw-bidask-only',action='store_true');a=p.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks)
    s=G75TsugiV2(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant=a.variant));eng.add_strategy(s);eng.run()
    r={**s.summary(),'tf':a.tf,'raw_ticks':len(ticks),'chronology':'CAUSAL_RAW_BIDASK_V2_DIRECTION_CONFIRM_COOLDOWN_REARM','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'period_start':man.get('start'),'period_days':man.get('days'),'period_end_exclusive':man.get('end_exclusive')}
    out=Path('results/ae-bt')/a.experiment_id/'cells';out.mkdir(parents=True,exist_ok=True)
    (out/f'{a.tf}_{a.variant}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));eng.dispose()
if __name__=='__main__':main()
