from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from collections import defaultdict, deque
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
    n:int=0; s:float=0.0; s2:float=0.0
    def add(self,x:float): self.n+=1; self.s+=x; self.s2+=x*x
    @property
    def mean(self): return self.s/max(1,self.n)
    @property
    def var(self):
        if self.n<2:return 0.0
        return max(0.0,(self.s2-self.s*self.s/self.n)/(self.n-1))
    def lcb(self,z=0.5): return self.mean-z*math.sqrt(self.var/max(1,self.n))

# deliberately coarser bins than v3 to avoid sparse/overfit cells
VEL_CUTS=[-0.03,-0.01,0.01,0.03]
MOM_CUTS=[-0.10,-0.03,0.03,0.10]
OVR_CUTS=[0.01,0.03,0.06]

def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def qbin(x,cuts):
    for i,c in enumerate(cuts):
        if x<c:return i
    return len(cuts)

def load_all(catalog):
    cat=ParquetDataCatalog(str(Path(catalog)))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    if not ticks: raise SystemExit('no ticks')
    return inst,ticks

def make_feature(trigger_side, anchor, mid, mids, prev_closes):
    vel=(mids[-1]-mids[0]) if len(mids)>=3 else 0.0
    mom=(prev_closes[-1]-prev_closes[-2]) if len(prev_closes)>=2 else 0.0
    overshoot=max(0.0,abs(mid-anchor)-0.12)
    # features are state descriptors, not chosen-action descriptors
    return (int(trigger_side),qbin(vel,VEL_CUTS),qbin(mom,MOM_CUTS),qbin(overshoot,OVR_CUTS))

def collect_trigger_events(ticks,tf_sec,start_i,end_i):
    """First G75 trigger per TF bucket. Returns indices/features using only information available at trigger time."""
    bucket=None; anchor=None; started=False; bar_close=None
    prev_closes=deque(maxlen=4); mids=deque(maxlen=8); events=[]
    for i in range(start_i,end_i):
        t=ticks[i]; bid=f(t.bid_price); ask=f(t.ask_price); mid=(bid+ask)/2; mids.append(mid)
        b=(int(t.ts_event)//1_000_000_000)//tf_sec
        if bucket is None: bucket=b; anchor=mid
        elif b!=bucket:
            if bar_close is not None: prev_closes.append(bar_close); anchor=bar_close
            bucket=b; started=False
        bar_close=mid
        if (not started) and anchor is not None:
            side=1 if mid>=anchor+0.12 else (-1 if mid<=anchor-0.12 else 0)
            if side:
                events.append((i,make_feature(side,anchor,mid,mids,prev_closes),side))
                started=True
    return events

def build_counterfactual_table(ticks,tf_sec,start_i,end_i,horizon_mult=1.0):
    """Label BOTH LONG and SHORT at each train trigger with raw Bid/Ask forward PnL.
    Entry uses executable ask/bid and exit uses executable bid/ask, so spread is embedded.
    Labels are future-looking only inside the training slice; evaluation never sees them.
    """
    events=collect_trigger_events(ticks,tf_sec,start_i,end_i)
    ts=[int(t.ts_event) for t in ticks]
    horizon_ns=int(tf_sec*horizon_mult*1_000_000_000)
    tables={1:defaultdict(Stat),-1:defaultdict(Stat)}
    j=start_i
    for i,key,_ in events:
        target=ts[i]+horizon_ns
        j=max(j,i+1)
        while j<end_i and ts[j]<target: j+=1
        if j>=end_i: break
        t0=ticks[i]; t1=ticks[j]
        ask0=f(t0.ask_price); bid0=f(t0.bid_price); ask1=f(t1.ask_price); bid1=f(t1.bid_price)
        pnl_long=bid1-ask0
        pnl_short=bid0-ask1
        tables[1][key].add(pnl_long); tables[-1][key].add(pnl_short)
    return tables,len(events)

class ExpectedActionV4(G75TsugiV2):
    def __init__(self,cfg,tables,min_samples=20,z=0.5,margin=0.0,sizing=False):
        super().__init__(cfg); self.tables=tables; self.min_samples=min_samples; self.z=z; self.margin=margin; self.sizing=sizing
        self.ev_accepts=self.ev_skips=self.ev_unknown=self.action_flips=0
        self.current_lcb=0.0; self.current_mean=0.0
    def _feature(self,trigger_side,mid):
        vel=(self.tick_mids[-1]-self.tick_mids[0]) if len(self.tick_mids)>=3 else 0.0
        mom=(self.prev_closes[-1]-self.prev_closes[-2]) if len(self.prev_closes)>=2 else 0.0
        overshoot=max(0.0,abs(mid-self.anchor)-self.config.trigger)
        return (int(trigger_side),qbin(vel,VEL_CUTS),qbin(mom,MOM_CUTS),qbin(overshoot,OVR_CUTS))
    def score(self,action_side,key):
        st=self.tables.get(action_side,{}).get(key)
        if st is None or st.n<self.min_samples:return None
        return st.mean,st.lcb(self.z),st.n
    def cap(self):
        base=super().cap()
        if not self.sizing:return base
        if self.current_lcb<=0:return 1
        if self.current_lcb<0.03:return max(1,round(base*0.3))
        if self.current_lcb<0.08:return max(1,round(base*0.6))
        return base
    def on_quote_tick(self,t:QuoteTick):
        bid=f(t.bid_price); ask=f(t.ask_price); mid=(bid+ask)/2
        self.last_bid=bid; self.last_ask=ask; self.tick_mids.append(mid)
        b=(int(t.ts_event)//1_000_000_000)//self.config.tf_sec
        if self.bucket is None:self.bucket=b;self.anchor=mid
        elif b!=self.bucket:
            self.finish_bar();self.bucket=b;self.started_bucket=False
        self.bar_close_mid=mid
        if not self.active and not self.started_bucket and self.anchor is not None:
            trigger_side=1 if mid>=self.anchor+self.config.trigger else (-1 if mid<=self.anchor-self.config.trigger else 0)
            if trigger_side:
                self.trigger_candidates+=1; key=self._feature(trigger_side,mid)
                c=[]
                for a_side in (1,-1):
                    sc=self.score(a_side,key)
                    if sc is not None:c.append((sc[1],sc[0],a_side,sc[2]))
                if not c:self.ev_unknown+=1;return
                lcb,mean,side,n=max(c,key=lambda x:x[0])
                if lcb<=self.margin:self.ev_skips+=1;return
                self.ev_accepts+=1;self.current_lcb=lcb;self.current_mean=mean
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
        x=super().summary();x.update({'ev_accepts':self.ev_accepts,'ev_skips':self.ev_skips,'ev_unknown':self.ev_unknown,'action_flips':self.action_flips,'ev_sizing':self.sizing})
        return x

def serialize_tables(tables):
    return {str(side):[{'k':list(k),'n':v.n,'s':v.s,'s2':v.s2} for k,v in tb.items()] for side,tb in tables.items()}
def deserialize_tables(obj):
    return {int(side):{tuple(r['k']):Stat(int(r['n']),float(r['s']),float(r['s2'])) for r in rows} for side,rows in obj.items()}

def engine_run(inst,ticks,strat):
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks);eng.add_strategy(strat);eng.run();r=strat.summary();eng.dispose();return r

def train_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*0.40)
    tables,n_events=build_counterfactual_table(ticks,TF_SEC[a.tf],0,split,a.horizon_mult)
    Path(a.out).write_text(json.dumps({'raw_ticks_total':len(ticks),'train_ticks':split,'train_trigger_events':n_events,'tables':serialize_tables(tables)}),encoding='utf-8')

def eval_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*0.40); test=ticks[split:]
    if a.variant=='PURE':s=G75TsugiV2(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'))
    else:
        raw=json.loads(Path(a.table).read_text());tables=deserialize_tables(raw['tables'])
        s=ExpectedActionV4(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'),tables,a.min_samples,a.z,a.margin,a.variant=='EV_ACTION_SIZER')
    r=engine_run(inst,test,s);Path(a.out).write_text(json.dumps({**r,'test_ticks':len(test),'raw_ticks_total':len(ticks)}),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--tf',choices=TF_SEC,required=True);ap.add_argument('--variant',choices=['PURE','EV_ACTION','EV_ACTION_SIZER'],default='PURE');ap.add_argument('--raw-bidask-only',action='store_true');ap.add_argument('--worker',choices=['train','eval']);ap.add_argument('--out');ap.add_argument('--table');ap.add_argument('--min-samples',type=int,default=20);ap.add_argument('--z',type=float,default=0.5);ap.add_argument('--margin',type=float,default=0.0);ap.add_argument('--horizon-mult',type=float,default=1.0);a=ap.parse_args()
    if a.worker=='train':return train_worker(a)
    if a.worker=='eval':return eval_worker(a)
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    d=Path('results/g75-expected-action-v4')/a.experiment_id;d.mkdir(parents=True,exist_ok=True);tr=d/f'{a.tf}_train.json';ev=d/f'{a.tf}_{a.variant}_eval.json'
    base=[sys.executable,__file__,'--catalog',a.catalog,'--experiment-id',a.experiment_id,'--tf',a.tf,'--raw-bidask-only','--min-samples',str(a.min_samples),'--z',str(a.z),'--margin',str(a.margin),'--horizon-mult',str(a.horizon_mult)]
    subprocess.run(base+['--worker','train','--out',str(tr)],check=True,env=os.environ.copy())
    subprocess.run(base+['--worker','eval','--variant',a.variant,'--table',str(tr),'--out',str(ev)],check=True,env=os.environ.copy())
    train=json.loads(tr.read_text());out=json.loads(ev.read_text())
    final={**out,'tf':a.tf,'variant_eval':a.variant,'train_ticks':train['train_ticks'],'train_trigger_events':train['train_trigger_events'],'split':'40% chronological train / 60% causal OOS','counterfactual_label':'both LONG and SHORT executable Bid/Ask PnL at fixed forward horizon inside TRAIN only','horizon_mult':a.horizon_mult,'min_samples':a.min_samples,'lcb_z':a.z,'ev_margin':a.margin,'frozen_opportunity_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},'verification_level':'CAUSAL_RAW_BIDASK_COUNTERFACTUAL_EXPECTED_ACTION_V4'}
    (d/f'{a.tf}_{a.variant}.json').write_text(json.dumps(final,indent=2),encoding='utf-8');print(json.dumps(final,indent=2))
if __name__=='__main__':main()
