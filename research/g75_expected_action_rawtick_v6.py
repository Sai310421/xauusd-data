from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from collections import deque
import numpy as np

from nautilus_trader.model.data import QuoteTick
from g75_tsugi_causal_rawtick_v2 import G75TsugiV2, Cfg, TF_SEC
from g75_expected_action_rawtick_v5 import f, load_all, engine_run, basket_counterfactual

FEATURE_NAMES=[
    'trigger_side','vel_short','vel_long','mom_1','mom_2','overshoot','spread',
    'range_short','range_long','hour_sin','hour_cos'
]


def _std(v):
    if len(v)<2:return 0.0
    a=np.asarray(v,dtype=float); return float(a.std())


def collect_events_continuous(ticks,tf_sec,start_i,end_i):
    bucket=None; anchor=None; started=False; bar_close=None
    prev_closes=deque(maxlen=6); mids=deque(maxlen=32); events=[]
    for i in range(start_i,end_i):
        t=ticks[i]; bid=f(t.bid_price); ask=f(t.ask_price); mid=(bid+ask)/2; mids.append(mid)
        b=(int(t.ts_event)//1_000_000_000)//tf_sec
        if bucket is None:
            bucket=b; anchor=mid
        elif b!=bucket:
            if bar_close is not None:
                prev_closes.append(bar_close); anchor=bar_close
            bucket=b; started=False
        bar_close=mid
        if started or anchor is None: continue
        side=1 if mid>=anchor+0.12 else (-1 if mid<=anchor-0.12 else 0)
        if not side: continue
        started=True
        ms=list(mids); pc=list(prev_closes)
        vel_s=ms[-1]-ms[-4] if len(ms)>=4 else 0.0
        vel_l=ms[-1]-ms[0] if len(ms)>=8 else vel_s
        mom1=pc[-1]-pc[-2] if len(pc)>=2 else 0.0
        mom2=pc[-1]-pc[-3] if len(pc)>=3 else mom1
        overshoot=max(0.0,abs(mid-anchor)-0.12)
        spread=max(0.0,ask-bid)
        rng_s=_std(ms[-8:]) if len(ms)>=4 else 0.0
        rng_l=_std(ms)
        sec=(int(t.ts_event)//1_000_000_000)%86400; ang=2*np.pi*sec/86400.0
        x=np.array([side,vel_s,vel_l,mom1,mom2,overshoot,spread,rng_s,rng_l,np.sin(ang),np.cos(ang)],dtype=float)
        events.append((i,x,side))
    return events


def fit_ridge(X,y,alpha=10.0):
    X=np.asarray(X,float); y=np.asarray(y,float)
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1.0
    Z=(X-mu)/sd
    Z1=np.column_stack([np.ones(len(Z)),Z])
    I=np.eye(Z1.shape[1]); I[0,0]=0.0
    beta=np.linalg.solve(Z1.T@Z1 + alpha*I, Z1.T@y)
    resid=y-Z1@beta
    sigma=float(np.sqrt(np.mean(resid*resid)))
    return {'mu':mu.tolist(),'sd':sd.tolist(),'beta':beta.tolist(),'sigma':sigma,'n':int(len(y))}


def pred(model,x):
    mu=np.asarray(model['mu']); sd=np.asarray(model['sd']); beta=np.asarray(model['beta'])
    z=(np.asarray(x)-mu)/sd
    return float(beta[0]+z@beta[1:])


def train_models(ticks,tf_sec,start_i,end_i,horizon_mult=4.0,alpha=10.0):
    events=collect_events_continuous(ticks,tf_sec,start_i,end_i)
    X=[]; yl=[]; ys=[]
    for i,x,_ in events:
        pl,_=basket_counterfactual(ticks,i,end_i,1,tf_sec,horizon_mult)
        ps,_=basket_counterfactual(ticks,i,end_i,-1,tf_sec,horizon_mult)
        X.append(x); yl.append(pl); ys.append(ps)
    if len(X)<50: raise SystemExit('insufficient train events')
    return {'long':fit_ridge(X,yl,alpha),'short':fit_ridge(X,ys,alpha),'train_trigger_events':len(X)}


class ExpectedActionV6(G75TsugiV2):
    def __init__(self,cfg,models,margin=0.0,sigma_penalty=0.10,sizing=False):
        super().__init__(cfg)
        self.models=models; self.margin=margin; self.sigma_penalty=sigma_penalty; self.sizing=sizing
        self.ev_accepts=self.ev_skips=self.action_flips=0; self.current_score=0.0
    def _x(self,trigger_side,mid,bid,ask,t):
        ms=list(self.tick_mids); pc=list(self.prev_closes)
        vel_s=ms[-1]-ms[-4] if len(ms)>=4 else 0.0
        vel_l=ms[-1]-ms[0] if len(ms)>=8 else vel_s
        mom1=pc[-1]-pc[-2] if len(pc)>=2 else 0.0
        mom2=pc[-1]-pc[-3] if len(pc)>=3 else mom1
        overshoot=max(0.0,abs(mid-self.anchor)-self.config.trigger)
        spread=max(0.0,ask-bid)
        rng_s=_std(ms[-8:]) if len(ms)>=4 else 0.0; rng_l=_std(ms)
        sec=(int(t.ts_event)//1_000_000_000)%86400; ang=2*np.pi*sec/86400.0
        return np.array([trigger_side,vel_s,vel_l,mom1,mom2,overshoot,spread,rng_s,rng_l,np.sin(ang),np.cos(ang)],float)
    def cap(self):
        base=super().cap()
        if not self.sizing:return base
        if self.current_score<0.20:return max(1,round(base*0.3))
        if self.current_score<0.60:return max(1,round(base*0.6))
        return base
    def on_quote_tick(self,t:QuoteTick):
        bid=f(t.bid_price); ask=f(t.ask_price); mid=(bid+ask)/2
        self.last_bid=bid; self.last_ask=ask; self.tick_mids.append(mid)
        b=(int(t.ts_event)//1_000_000_000)//self.config.tf_sec
        if self.bucket is None:
            self.bucket=b; self.anchor=mid
        elif b!=self.bucket:
            self.finish_bar(); self.bucket=b; self.started_bucket=False
        self.bar_close_mid=mid
        if not self.active and not self.started_bucket and self.anchor is not None:
            trigger_side=1 if mid>=self.anchor+self.config.trigger else (-1 if mid<=self.anchor-self.config.trigger else 0)
            if trigger_side:
                self.started_bucket=True; self.trigger_candidates+=1
                x=self._x(trigger_side,mid,bid,ask,t)
                pl=pred(self.models['long'],x)-self.sigma_penalty*self.models['long']['sigma']
                ps=pred(self.models['short'],x)-self.sigma_penalty*self.models['short']['sigma']
                score,side=max((pl,1),(ps,-1),key=lambda z:z[0])
                if score<=self.margin:
                    self.ev_skips+=1; return
                self.ev_accepts+=1; self.current_score=score
                if side!=trigger_side:self.action_flips+=1
                entry=ask if side>0 else bid
                self.side=side; self.active=True; self.entries=[entry]; self.last_add=entry
                self.extreme=bid if side>0 else ask; self.maxlayers=max(self.maxlayers,1)
        if not self.active:return
        px=bid if self.side>0 else ask
        self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        while len(self.entries)<self.cap():
            target=self.last_add+self.side*self.config.add
            cross=px>=target if self.side>0 else px<=target
            if not cross:break
            fill=ask if self.side>0 else bid
            self.entries.append(fill); self.last_add=target; self.adds+=1; self.maxlayers=max(self.maxlayers,len(self.entries))
        self.dd(bid,ask)
    def summary(self):
        x=super().summary(); x.update({'ev_accepts':self.ev_accepts,'ev_skips':self.ev_skips,
            'action_flips':self.action_flips,'ev_sizing':self.sizing,
            'decision_parity_ok':self.trigger_candidates==(self.ev_accepts+self.ev_skips)})
        return x


def train_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*0.40)
    models=train_models(ticks,TF_SEC[a.tf],0,split,a.horizon_mult,a.alpha)
    Path(a.out).write_text(json.dumps({'raw_ticks_total':len(ticks),'train_ticks':split,'models':models}),encoding='utf-8')

def eval_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*0.40); test=ticks[split:]
    if a.variant=='PURE': s=G75TsugiV2(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'))
    else:
        raw=json.loads(Path(a.model).read_text()); models=raw['models']
        s=ExpectedActionV6(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'),models,a.margin,a.sigma_penalty,a.variant=='EV_ACTION_SIZER')
    r=engine_run(inst,test,s); Path(a.out).write_text(json.dumps({**r,'test_ticks':len(test),'raw_ticks_total':len(ticks)}),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--tf',choices=TF_SEC,required=True); ap.add_argument('--variant',choices=['PURE','EV_ACTION','EV_ACTION_SIZER'],default='PURE')
    ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--worker',choices=['train','eval']); ap.add_argument('--out'); ap.add_argument('--model')
    ap.add_argument('--margin',type=float,default=0.0); ap.add_argument('--sigma-penalty',type=float,default=0.10); ap.add_argument('--alpha',type=float,default=10.0); ap.add_argument('--horizon-mult',type=float,default=4.0)
    a=ap.parse_args()
    if a.worker=='train': return train_worker(a)
    if a.worker=='eval': return eval_worker(a)
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    d=Path('results/g75-expected-action-v6')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    tr=d/f'{a.tf}_train.json'; ev=d/f'{a.tf}_{a.variant}_eval.json'
    base=[sys.executable,__file__,'--catalog',a.catalog,'--experiment-id',a.experiment_id,'--tf',a.tf,'--raw-bidask-only','--margin',str(a.margin),'--sigma-penalty',str(a.sigma_penalty),'--alpha',str(a.alpha),'--horizon-mult',str(a.horizon_mult)]
    subprocess.run(base+['--worker','train','--out',str(tr)],check=True,env=os.environ.copy())
    subprocess.run(base+['--worker','eval','--variant',a.variant,'--model',str(tr),'--out',str(ev)],check=True,env=os.environ.copy())
    train=json.loads(tr.read_text()); out=json.loads(ev.read_text())
    final={**out,'tf':a.tf,'variant_eval':a.variant,'train_ticks':train['train_ticks'],'train_trigger_events':train['models']['train_trigger_events'],
           'split':'40% chronological train / 60% causal OOS','model':'continuous ridge EV on train-only basket PnL',
           'feature_names':FEATURE_NAMES,'alpha':a.alpha,'sigma_penalty':a.sigma_penalty,'ev_margin':a.margin,
           'frozen_opportunity_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},
           'verification_level':'CAUSAL_RAW_BIDASK_EXPECTED_ACTION_V6_CONTINUOUS_EV'}
    (d/f'{a.tf}_{a.variant}.json').write_text(json.dumps(final,indent=2),encoding='utf-8'); print(json.dumps(final,indent=2))

if __name__=='__main__': main()
