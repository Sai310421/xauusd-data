from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from g75_tsugi_causal_rawtick_v2 import G75TsugiV2, Cfg, TF_SEC

SIM=Venue('SIM')

@dataclass
class Stat:
    n:int=0
    s:float=0.0
    s2:float=0.0
    def add(self,x): self.n+=1; self.s+=x; self.s2+=x*x
    @property
    def mean(self): return self.s/max(1,self.n)
    @property
    def var(self):
        if self.n<2:return 0.0
        return max(0.0,(self.s2-self.s*self.s/self.n)/(self.n-1))
    def lcb(self,z=1.0):
        return self.mean-z*math.sqrt(self.var/max(1,self.n))

VEL_CUTS=[-0.06,-0.03,-0.015,-0.005,0.005,0.015,0.03,0.06]
MOM_CUTS=[-0.20,-0.10,-0.05,-0.02,0.02,0.05,0.10,0.20]

def qbin(x,cuts):
    for i,c in enumerate(cuts):
        if x<c:return i
    return len(cuts)

def directional_key(strategy,side,mid):
    vel=0.0
    if len(strategy.tick_mids)>=3: vel=(strategy.tick_mids[-1]-strategy.tick_mids[0])*side
    mom=0.0
    if len(strategy.prev_closes)>=2: mom=(strategy.prev_closes[-1]-strategy.prev_closes[-2])*side
    ds=int(strategy.direction_score(side,mid))
    return (ds,qbin(vel,VEL_CUTS),qbin(mom,MOM_CUTS))

class Probe(G75TsugiV2):
    def __init__(self,cfg):
        super().__init__(cfg); self.open_key=None; self.rows=[]
    def close(self,bid,ask,reason):
        if not self.active:return 0.0
        k=self.open_key
        p=super().close(bid,ask,reason)
        if k is not None:self.rows.append({'key':list(k),'pnl':float(p)})
        self.open_key=None
        return p
    def on_quote_tick(self,t):
        was=self.active
        bid=self.f(t.bid_price); ask=self.f(t.ask_price); mid=(bid+ask)/2
        super().on_quote_tick(t)
        if (not was) and self.active:
            self.open_key=directional_key(self,self.side,mid)

class ExpectedAction(G75TsugiV2):
    """G75 opportunity trigger is frozen; entry direction is chosen by OOS-predicted action EV.
    For each trigger event, evaluate LONG and SHORT from the same observed state and choose
    the action with the highest positive conservative EV lower bound. Add/reversal thresholds stay frozen.
    """
    def __init__(self,cfg,table,min_samples=12,z=1.0,margin=0.0,sizing=False):
        super().__init__(cfg); self.table=table; self.min_samples=min_samples; self.z=z; self.margin=margin; self.sizing=sizing
        self.ev_accepts=self.ev_skips=self.ev_unknown=0; self.action_flips=0; self.current_lcb=0.0; self.current_mean=0.0
    def score_action(self,side,mid):
        st=self.table.get(directional_key(self,side,mid))
        if st is None or st.n<self.min_samples:return None
        return st.mean,st.lcb(self.z),st.n
    def cap(self):
        base=super().cap()
        if not self.sizing:return base
        if self.current_lcb<=0:return 1
        # conservative EV-ranked layer budget; price add spacing remains 0.025
        if self.current_lcb<0.05:return max(1,round(base*0.3))
        if self.current_lcb<0.15:return max(1,round(base*0.6))
        return base
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price); ask=self.f(t.ask_price); mid=(bid+ask)/2
        self.last_bid=bid; self.last_ask=ask; self.tick_mids.append(mid)
        ns=int(t.ts_event); b=(ns//1_000_000_000)//self.config.tf_sec
        if self.bucket is None:self.bucket=b;self.anchor=mid
        elif b!=self.bucket:
            self.finish_bar();self.bucket=b;self.started_bucket=False
        self.bar_close_mid=mid; self.maybe_rearm(bid,ask)
        if self.mode=='COOLDOWN': self.dd(bid,ask); return
        self.dd(bid,ask)
        if not self.active and not self.started_bucket and self.anchor is not None:
            trigger_side=1 if mid>=self.anchor+self.config.trigger else (-1 if mid<=self.anchor-self.config.trigger else 0)
            if trigger_side:
                self.trigger_candidates+=1
                long_s=self.score_action(1,mid); short_s=self.score_action(-1,mid)
                candidates=[]
                if long_s is not None:candidates.append((long_s[1],long_s[0],1,long_s[2]))
                if short_s is not None:candidates.append((short_s[1],short_s[0],-1,short_s[2]))
                if not candidates:
                    self.ev_unknown+=1; return
                best=max(candidates,key=lambda x:x[0])
                lcb,mean,side,n=best
                if lcb<=self.margin:
                    self.ev_skips+=1; return
                self.ev_accepts+=1; self.current_lcb=lcb; self.current_mean=mean
                if side!=trigger_side:self.action_flips+=1
                entry=ask if side>0 else bid
                self.side=side;self.active=True;self.entries=[entry];self.last_add=entry
                self.extreme=(bid if side>0 else ask);self.started_bucket=True;self.maxlayers=max(self.maxlayers,1)
        if not self.active:return
        px=bid if self.side>0 else ask
        self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        while len(self.entries)<self.cap():
            target=self.last_add+self.side*self.config.add
            cross=px>=target if self.side>0 else px<=target
            if not cross:break
            fill=ask if self.side>0 else bid
            self.entries.append(fill);self.last_add=target;self.adds+=1;self.maxlayers=max(self.maxlayers,len(self.entries))
        self.dd(bid,ask)
    def summary(self):
        x=super().summary(); x.update({'ev_accepts':self.ev_accepts,'ev_skips':self.ev_skips,'ev_unknown':self.ev_unknown,'action_flips':self.action_flips,'ev_sizing':self.sizing})
        return x

def load_slice(catalog,start,end):
    cat=ParquetDataCatalog(str(Path(catalog))); inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]); n=len(ticks)
    return inst,ticks[int(n*start):int(n*end)],n

def engine_run(inst,ticks,strat):
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks);eng.add_strategy(strat);eng.run();res=strat.summary();eng.dispose();return res

def train_worker(a):
    inst,ticks,total=load_slice(a.catalog,0.0,0.40); s=Probe(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'))
    res=engine_run(inst,ticks,s); table=defaultdict(Stat)
    for r in s.rows: table[tuple(r['key'])].add(float(r['pnl']))
    obj={'raw_ticks_total':total,'train_ticks':len(ticks),'train_summary':res,'table':[{'k':list(k),'n':v.n,'s':v.s,'s2':v.s2} for k,v in table.items()]}
    Path(a.out).write_text(json.dumps(obj),encoding='utf-8')

def eval_worker(a):
    inst,ticks,total=load_slice(a.catalog,0.40,1.0); raw=json.loads(Path(a.table).read_text()); table={tuple(r['k']):Stat(int(r['n']),float(r['s']),float(r['s2'])) for r in raw['table']}
    if a.variant=='PURE': s=Probe(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'))
    else: s=ExpectedAction(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'),table,min_samples=a.min_samples,z=a.z,margin=a.margin,sizing=(a.variant=='EV_ACTION_SIZER'))
    res=engine_run(inst,ticks,s); Path(a.out).write_text(json.dumps({**res,'test_ticks':len(ticks),'raw_ticks_total':total}),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--tf',choices=TF_SEC,required=True);ap.add_argument('--variant',choices=['PURE','EV_ACTION','EV_ACTION_SIZER'],default='PURE');ap.add_argument('--raw-bidask-only',action='store_true');ap.add_argument('--worker',choices=['train','eval']);ap.add_argument('--out');ap.add_argument('--table');ap.add_argument('--min-samples',type=int,default=12);ap.add_argument('--z',type=float,default=1.0);ap.add_argument('--margin',type=float,default=0.0);a=ap.parse_args()
    if a.worker=='train':return train_worker(a)
    if a.worker=='eval':return eval_worker(a)
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    d=Path('results/g75-expected-action')/a.experiment_id;d.mkdir(parents=True,exist_ok=True);tr=d/f'{a.tf}_train.json';ev=d/f'{a.tf}_{a.variant}_eval.json'
    base=[sys.executable,__file__,'--catalog',a.catalog,'--experiment-id',a.experiment_id,'--tf',a.tf,'--raw-bidask-only','--min-samples',str(a.min_samples),'--z',str(a.z),'--margin',str(a.margin)]
    subprocess.run(base+['--worker','train','--out',str(tr)],check=True,env=os.environ.copy())
    subprocess.run(base+['--worker','eval','--variant',a.variant,'--table',str(tr),'--out',str(ev)],check=True,env=os.environ.copy())
    train=json.loads(tr.read_text());out=json.loads(ev.read_text());final={**out,'tf':a.tf,'variant_eval':a.variant,'train_ticks':train['train_ticks'],'split':'40% chronological train / 60% causal OOS','min_samples':a.min_samples,'lcb_z':a.z,'ev_margin':a.margin,'frozen_opportunity_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},'entry_policy':'at each G75 trigger evaluate LONG and SHORT conservative EV; take max positive action or skip','verification_level':'CAUSAL_RAW_BIDASK_EXPECTED_ACTION_V3'}
    (d/f'{a.tf}_{a.variant}.json').write_text(json.dumps(final,indent=2),encoding='utf-8');print(json.dumps(final,indent=2))
if __name__=='__main__':main()
