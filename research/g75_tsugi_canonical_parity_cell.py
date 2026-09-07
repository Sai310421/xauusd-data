from __future__ import annotations
import argparse,json,math
from pathlib import Path
from decimal import Decimal
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId

SIM=Venue('SIM'); TF_MIN={'M1':1,'M5':5,'M15':15}
class Cfg(StrategyConfig,frozen=True):
    instrument_id:InstrumentId; bar_type:BarType; variant:str
    initial_balance:float=1000.0; trigger:float=0.12; add:float=0.025; reversal:float=0.20; max_layers:int=10
    soft_dd_pct:float=1.5; lock_dd_pct:float=3.0; hard_dd_pct:float=4.5; recovery_alloc:float=.70

class CanonicalParity(Strategy):
    def __init__(self,cfg:Cfg):
        super().__init__(cfg); self.anchor=None; self.active=False; self.side=0; self.last_add=None; self.extreme=None; self.entries=[]
        self.realized=0.; self.peak=cfg.initial_balance; self.maxdd=0.; self.cycles=self.wins=self.losses=self.adds=0; self.gw=self.gl=0.; self.maxlayers=0
        self.mode='NORMAL'; self.debt=0.; self.maxdebt=0.; self.soft=self.locks=self.hard=self.rec_cycles=self.rec_success=0; self.debt_bars=0; self.lock_bar=None; self.bar_idx=0; self.clear_bars=[]; self.rec_hist=[]; self.stopped=False
    @staticmethod
    def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def on_start(self): self.subscribe_bars(self.config.bar_type)
    def mark(self,c): return sum((c-e)*self.side for e in self.entries) if self.active else 0.
    def dd(self,c):
        eq=self.config.initial_balance+self.realized+self.mark(c); self.peak=max(self.peak,eq); d=max(0.,(self.peak-eq)/max(self.peak,1e-9)*100); self.maxdd=max(self.maxdd,d); return d
    def cap(self): return max(1,self.config.max_layers//2) if self.variant!='A' and self.mode in ('REDUCED','RECOVERY') else self.config.max_layers
    def rec_ok(self):
        if self.mode!='RECOVERY' or len(self.rec_hist)<5:return True
        h=self.rec_hist[-20:]; return sum(x for x in h if x>0)>abs(sum(x for x in h if x<0))
    def close(self,c,reason):
        if not self.active:return
        p=sum((c-e)*self.side for e in self.entries); self.realized+=p; self.cycles+=1
        if p>0:self.wins+=1;self.gw+=p
        elif p<0:self.losses+=1;self.gl+=abs(p)
        if self.mode=='RECOVERY':
            self.rec_cycles+=1;self.rec_hist.append(p)
            if p>0 and self.debt>0:
                self.debt=max(0.,self.debt-p*self.config.recovery_alloc)
                if self.debt<=1e-12:
                    self.rec_success+=1
                    if self.lock_bar is not None:self.clear_bars.append(self.bar_idx-self.lock_bar)
                    self.mode='NORMAL';self.lock_bar=None
        self.active=False;self.side=0;self.entries=[];self.last_add=None;self.extreme=None;self.anchor=c
    def lock(self,c):
        locked=max(0.,-self.mark(c)); self.debt+=locked;self.maxdebt=max(self.maxdebt,self.debt);self.locks+=1;self.lock_bar=self.bar_idx
        self.close(c,'LOCK');self.mode='RECOVERY';self.anchor=c
    def on_bar(self,b:Bar):
        self.bar_idx+=1; o,h,l,c=map(self.f,(b.open,b.high,b.low,b.close))
        if self.debt>0:self.debt_bars+=1;self.maxdebt=max(self.maxdebt,self.debt)
        if self.anchor is None:self.anchor=c;return
        if self.stopped:return
        # Controller observes existing basket at prior state before this bar's canonical operations.
        d=self.dd(c)
        if self.variant!='A':
            if d>=self.config.hard_dd_pct:
                self.hard+=1
                if self.active:self.close(c,'HARD')
                self.stopped=True;self.mode='STOPPED';return
            if self.variant=='C' and self.active and self.mode!='RECOVERY' and d>=self.config.lock_dd_pct:
                self.lock(c);return
            if self.mode not in ('RECOVERY','STOPPED'):
                if d>=self.config.soft_dd_pct:
                    if self.mode!='REDUCED':self.soft+=1
                    self.mode='REDUCED'
                else:self.mode='NORMAL'
        if not self.active:
            if self.mode=='RECOVERY' and not self.rec_ok():return
            up=h>=self.anchor+self.config.trigger;dn=l<=self.anchor-self.config.trigger
            if not(up or dn):self.anchor=c;return
            self.side=1 if c>=self.anchor else -1; entry=self.anchor+self.side*self.config.trigger
            self.active=True;self.entries=[entry];self.last_add=entry;self.extreme=entry;self.maxlayers=max(self.maxlayers,1)
        cap=self.cap()
        if self.side==1:
            while len(self.entries)<cap and self.last_add+self.config.add<=h+1e-12:
                self.last_add+=self.config.add;self.entries.append(self.last_add);self.adds+=1;self.maxlayers=max(self.maxlayers,len(self.entries))
            self.extreme=max(self.extreme,h); rev=c<=self.extreme-self.config.reversal
        else:
            while len(self.entries)<cap and self.last_add-self.config.add>=l-1e-12:
                self.last_add-=self.config.add;self.entries.append(self.last_add);self.adds+=1;self.maxlayers=max(self.maxlayers,len(self.entries))
            self.extreme=min(self.extreme,l); rev=c>=self.extreme+self.config.reversal
        if rev:self.close(c,'REVERSAL')
        self.dd(c)
    def summary(self):
        pf=self.gw/self.gl if self.gl else (math.inf if self.gw else 0.)
        return {'variant':self.variant,'cycles':self.cycles,'WR_pct':100*self.wins/max(1,self.cycles),'PF':pf,'realized_usd_0p01lot_equiv':self.realized,'return_pct_on_1000':self.realized/10,'max_DD_pct':self.maxdd,'adds':self.adds,'max_layers':self.maxlayers,'mode_final':self.mode,'debt_final':self.debt,'max_debt':self.maxdebt,'soft_events':self.soft,'lock_events':self.locks,'hard_events':self.hard,'recovery_cycles':self.rec_cycles,'recovery_success':self.rec_success,'debt_clear_bars_p50':sorted(self.clear_bars)[len(self.clear_bars)//2] if self.clear_bars else None,'stopped':self.stopped}

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--experiment-id',required=True);p.add_argument('--tf',choices=TF_MIN,required=True);p.add_argument('--variant',choices=['A','B','C'],required=True);p.add_argument('--raw-bidask-only',action='store_true');a=p.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-{TF_MIN[a.tf]}-MINUTE-BID-INTERNAL');s=CanonicalParity(Cfg(instrument_id=inst.id,bar_type=bt,variant=a.variant));eng.add_strategy(s);eng.run();r={**s.summary(),'tf':a.tf,'raw_ticks':len(ticks),'chronology':'CANONICAL_G75_BAR_PARITY_FROM_RAW_TICKS','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'period_start':man.get('start'),'period_days':man.get('days'),'period_end_exclusive':man.get('end_exclusive')};out=Path('results/ae-bt')/a.experiment_id/'cells';out.mkdir(parents=True,exist_ok=True);(out/f'{a.tf}_{a.variant}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));eng.dispose()
if __name__=='__main__':main()
