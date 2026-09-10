from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

from g75_tsugi_causal_rawtick_v2 import G75TsugiV2, Cfg, TF_SEC
from g75_expected_action_rawtick_v5 import f, load_all, engine_run
from g75_expected_action_rawtick_v6 import collect_events_continuous, fit_ridge, pred


def first_passage_label(ticks, i0, end_i, side, tf_sec, tp=0.20, sl=0.10, horizon_mult=4.0):
    t0=ticks[i0]
    bid0=f(t0.bid_price); ask0=f(t0.ask_price)
    entry=ask0 if side>0 else bid0
    start_ns=int(t0.ts_event)
    deadline=start_ns+int(tf_sec*horizon_mult*1_000_000_000)
    last=0.0
    for j in range(i0+1,end_i):
        t=ticks[j]; bid=f(t.bid_price); ask=f(t.ask_price)
        px=bid if side>0 else ask
        pnl=(px-entry)*side
        last=pnl
        if pnl>=tp: return tp, 1
        if pnl<=-sl: return -sl, 0
        if int(t.ts_event)>=deadline: break
    return last, int(last>0)


def train_models(ticks,tf_sec,start_i,end_i,tp,sl,horizon_mult,alpha):
    events=collect_events_continuous(ticks,tf_sec,start_i,end_i)
    X=[]; yl=[]; ys=[]; pl=[]; ps=[]
    for i,x,_ in events:
        pnl_l,win_l=first_passage_label(ticks,i,end_i,1,tf_sec,tp,sl,horizon_mult)
        pnl_s,win_s=first_passage_label(ticks,i,end_i,-1,tf_sec,tp,sl,horizon_mult)
        X.append(x); yl.append(float(win_l)); ys.append(float(win_s)); pl.append(float(pnl_l)); ps.append(float(pnl_s))
    if len(X)<50: raise SystemExit('insufficient train events')
    m_l=fit_ridge(X,yl,alpha); m_s=fit_ridge(X,ys,alpha)
    return {
        'long':m_l,'short':m_s,'train_trigger_events':len(X),
        'train_long_winrate':float(np.mean(yl)),'train_short_winrate':float(np.mean(ys)),
        'train_long_pnl_mean':float(np.mean(pl)),'train_short_pnl_mean':float(np.mean(ps)),
    }


class ExpectedEntryV7(G75TsugiV2):
    def __init__(self,cfg,models,tp=.20,sl=.10,margin=0.0,uncertainty=0.03,sizing=False):
        super().__init__(cfg)
        self.models=models; self.tp=tp; self.sl=sl; self.margin=margin; self.uncertainty=uncertainty; self.sizing=sizing
        self.ev_accepts=self.ev_skips=self.action_flips=0
        self.current_ev=0.0; self.pred_ev_sum=0.0; self.pred_ev_max=-1e9; self.pred_ev_min=1e9
    def _x(self,trigger_side,mid,bid,ask,t):
        # Match v6 feature construction exactly.
        from g75_expected_action_rawtick_v6 import _std
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
        # Scale by forecast edge above zero; preserve at least one layer.
        if self.current_ev<0.02:return max(1,round(base*.3))
        if self.current_ev<0.05:return max(1,round(base*.6))
        return base
    def on_quote_tick(self,t):
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
                spread=max(0.0,ask-bid)
                p_l=float(np.clip(pred(self.models['long'],x),0.0,1.0))
                p_s=float(np.clip(pred(self.models['short'],x),0.0,1.0))
                # Conservative expected value: subtract uncertainty and executable spread.
                ev_l=p_l*self.tp-(1-p_l)*self.sl-spread-self.uncertainty
                ev_s=p_s*self.tp-(1-p_s)*self.sl-spread-self.uncertainty
                ev,side=max((ev_l,1),(ev_s,-1),key=lambda z:z[0])
                self.pred_ev_sum+=ev; self.pred_ev_max=max(self.pred_ev_max,ev); self.pred_ev_min=min(self.pred_ev_min,ev)
                if ev<=self.margin:
                    self.ev_skips+=1; return
                self.ev_accepts+=1; self.current_ev=ev
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
        x=super().summary(); n=max(1,self.trigger_candidates)
        x.update({
            'ev_accepts':self.ev_accepts,'ev_skips':self.ev_skips,'action_flips':self.action_flips,
            'ev_sizing':self.sizing,'decision_parity_ok':self.trigger_candidates==(self.ev_accepts+self.ev_skips),
            'pred_ev_avg':self.pred_ev_sum/n,'pred_ev_max':self.pred_ev_max if self.trigger_candidates else None,
            'pred_ev_min':self.pred_ev_min if self.trigger_candidates else None,
            'accept_rate_pct':100*self.ev_accepts/n,
        })
        return x


def train_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*.40)
    models=train_models(ticks,TF_SEC[a.tf],0,split,a.tp,a.sl,a.horizon_mult,a.alpha)
    Path(a.out).write_text(json.dumps({'raw_ticks_total':len(ticks),'train_ticks':split,'models':models}),encoding='utf-8')


def eval_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*.40); test=ticks[split:]
    if a.variant=='PURE': s=G75TsugiV2(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'))
    else:
        raw=json.loads(Path(a.model).read_text()); models=raw['models']
        s=ExpectedEntryV7(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'),models,a.tp,a.sl,a.margin,a.uncertainty,a.variant=='EV_ENTRY_SIZER')
    r=engine_run(inst,test,s); Path(a.out).write_text(json.dumps({**r,'test_ticks':len(test),'raw_ticks_total':len(ticks)}),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--tf',choices=TF_SEC,required=True); ap.add_argument('--variant',choices=['PURE','EV_ENTRY','EV_ENTRY_SIZER'],default='PURE')
    ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--worker',choices=['train','eval']); ap.add_argument('--out'); ap.add_argument('--model')
    ap.add_argument('--tp',type=float,default=.20); ap.add_argument('--sl',type=float,default=.10); ap.add_argument('--margin',type=float,default=0.0)
    ap.add_argument('--uncertainty',type=float,default=.03); ap.add_argument('--alpha',type=float,default=10.0); ap.add_argument('--horizon-mult',type=float,default=4.0)
    a=ap.parse_args()
    if a.worker=='train': return train_worker(a)
    if a.worker=='eval': return eval_worker(a)
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    d=Path('results/g75-expected-entry-v7')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    tr=d/f'{a.tf}_train.json'; ev=d/f'{a.tf}_{a.variant}_eval.json'
    base=[sys.executable,__file__,'--catalog',a.catalog,'--experiment-id',a.experiment_id,'--tf',a.tf,'--raw-bidask-only',
          '--tp',str(a.tp),'--sl',str(a.sl),'--margin',str(a.margin),'--uncertainty',str(a.uncertainty),'--alpha',str(a.alpha),'--horizon-mult',str(a.horizon_mult)]
    subprocess.run(base+['--worker','train','--out',str(tr)],check=True,env=os.environ.copy())
    subprocess.run(base+['--worker','eval','--variant',a.variant,'--model',str(tr),'--out',str(ev)],check=True,env=os.environ.copy())
    train=json.loads(tr.read_text()); out=json.loads(ev.read_text())
    final={**out,'tf':a.tf,'variant_eval':a.variant,'train_ticks':train['train_ticks'],'train_trigger_events':train['models']['train_trigger_events'],
           'split':'40% chronological train / 60% causal OOS','ev_target':'first-passage TP/SL expectancy at G75 opportunity, train only',
           'tp':a.tp,'sl':a.sl,'rr':a.tp/a.sl,'alpha':a.alpha,'uncertainty':a.uncertainty,'ev_margin':a.margin,
           'train_long_winrate':train['models']['train_long_winrate'],'train_short_winrate':train['models']['train_short_winrate'],
           'frozen_opportunity_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},
           'verification_level':'CAUSAL_RAW_BIDASK_G75_FIRST_PASSAGE_EXPECTED_ENTRY_V7'}
    (d/f'{a.tf}_{a.variant}.json').write_text(json.dumps(final,indent=2),encoding='utf-8'); print(json.dumps(final,indent=2))

if __name__=='__main__': main()
