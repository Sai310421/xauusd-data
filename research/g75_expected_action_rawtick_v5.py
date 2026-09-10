from __future__ import annotations
import argparse, json, os, subprocess, sys
from collections import defaultdict
from pathlib import Path

from nautilus_trader.model.data import QuoteTick

from g75_tsugi_causal_rawtick_v2 import G75TsugiV2, Cfg, TF_SEC
from g75_expected_action_rawtick_v4 import (
    Stat, f, qbin, VEL_CUTS, MOM_CUTS, OVR_CUTS,
    load_all, collect_trigger_events, serialize_tables, deserialize_tables,
    engine_run,
)


def feature_from_state(trigger_side, anchor, mid, mids, prev_closes):
    vel=(mids[-1]-mids[0]) if len(mids)>=3 else 0.0
    mom=(prev_closes[-1]-prev_closes[-2]) if len(prev_closes)>=2 else 0.0
    overshoot=max(0.0,abs(mid-anchor)-0.12)
    return (int(trigger_side),qbin(vel,VEL_CUTS),qbin(mom,MOM_CUTS),qbin(overshoot,OVR_CUTS))


def basket_counterfactual(ticks, i0, end_i, side, tf_sec, max_horizon_mult=4.0):
    """Train-only counterfactual using the same G75 basket geometry.
    Entry/adds use executable Bid/Ask, add spacing=.025, max layers=10,
    and basket exits on .20 reversal from favorable extreme or max horizon.
    """
    t0=ticks[i0]
    bid0=f(t0.bid_price); ask0=f(t0.ask_price)
    entry=ask0 if side>0 else bid0
    entries=[entry]; last_add=entry
    extreme=bid0 if side>0 else ask0
    start_ns=int(t0.ts_event)
    deadline=start_ns+int(tf_sec*max_horizon_mult*1_000_000_000)
    exit_px=bid0 if side>0 else ask0
    j=i0+1
    while j<end_i:
        t=ticks[j]; bid=f(t.bid_price); ask=f(t.ask_price); px=bid if side>0 else ask
        exit_px=px
        extreme=max(extreme,px) if side>0 else min(extreme,px)
        while len(entries)<10:
            target=last_add+side*0.025
            cross=px>=target if side>0 else px<=target
            if not cross: break
            fill=ask if side>0 else bid
            entries.append(fill); last_add=target
        reversed_now=(px<=extreme-0.20) if side>0 else (px>=extreme+0.20)
        if reversed_now or int(t.ts_event)>=deadline:
            break
        j+=1
    pnl=sum((exit_px-e)*side for e in entries)
    return pnl, len(entries)


def build_basket_ev_table(ticks, tf_sec, start_i, end_i, horizon_mult=4.0):
    events=collect_trigger_events(ticks,tf_sec,start_i,end_i)
    tables={1:defaultdict(Stat),-1:defaultdict(Stat)}
    layer_stats={1:defaultdict(Stat),-1:defaultdict(Stat)}
    for i,key,_ in events:
        for side in (1,-1):
            pnl,layers=basket_counterfactual(ticks,i,end_i,side,tf_sec,horizon_mult)
            tables[side][key].add(pnl)
            layer_stats[side][key].add(float(layers))
    return tables,layer_stats,len(events)


class ExpectedActionV5(G75TsugiV2):
    """One opportunity decision per TF bucket; action chosen by train-only basket EV."""
    def __init__(self,cfg,tables,min_samples=12,z=0.25,margin=0.0,sizing=False):
        super().__init__(cfg)
        self.tables=tables; self.min_samples=min_samples; self.z=z; self.margin=margin; self.sizing=sizing
        self.ev_accepts=self.ev_skips=self.ev_unknown=self.action_flips=0
        self.current_lcb=0.0; self.current_mean=0.0
    def _feature(self,trigger_side,mid):
        return feature_from_state(trigger_side,self.anchor,mid,self.tick_mids,self.prev_closes)
    def score(self,side,key):
        st=self.tables.get(side,{}).get(key)
        if st is None or st.n<self.min_samples: return None
        return st.mean,st.lcb(self.z),st.n
    def cap(self):
        base=super().cap()
        if not self.sizing: return base
        if self.current_lcb<=0: return 1
        if self.current_lcb<0.10: return max(1,round(base*0.3))
        if self.current_lcb<0.30: return max(1,round(base*0.6))
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
                # CRITICAL parity fix: first trigger consumes the bucket whether accepted or skipped.
                self.started_bucket=True
                self.trigger_candidates+=1
                key=self._feature(trigger_side,mid)
                candidates=[]
                for action_side in (1,-1):
                    sc=self.score(action_side,key)
                    if sc is not None: candidates.append((sc[1],sc[0],action_side,sc[2]))
                if not candidates:
                    self.ev_unknown+=1; return
                lcb,mean,side,n=max(candidates,key=lambda x:x[0])
                if lcb<=self.margin:
                    self.ev_skips+=1; return
                self.ev_accepts+=1; self.current_lcb=lcb; self.current_mean=mean
                if side!=trigger_side: self.action_flips+=1
                entry=ask if side>0 else bid
                self.side=side; self.active=True; self.entries=[entry]; self.last_add=entry
                self.extreme=bid if side>0 else ask; self.maxlayers=max(self.maxlayers,1)

        if not self.active: return
        px=bid if self.side>0 else ask
        self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        while len(self.entries)<self.cap():
            target=self.last_add+self.side*self.config.add
            cross=px>=target if self.side>0 else px<=target
            if not cross: break
            fill=ask if self.side>0 else bid
            self.entries.append(fill); self.last_add=target; self.adds+=1; self.maxlayers=max(self.maxlayers,len(self.entries))
        self.dd(bid,ask)
    def summary(self):
        x=super().summary()
        x.update({'ev_accepts':self.ev_accepts,'ev_skips':self.ev_skips,'ev_unknown':self.ev_unknown,
                  'action_flips':self.action_flips,'ev_sizing':self.sizing,
                  'decision_parity_ok': self.trigger_candidates==(self.ev_accepts+self.ev_skips+self.ev_unknown)})
        return x


def train_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*0.40)
    tables,layer_stats,n_events=build_basket_ev_table(ticks,TF_SEC[a.tf],0,split,a.horizon_mult)
    obj={'raw_ticks_total':len(ticks),'train_ticks':split,'train_trigger_events':n_events,
         'tables':serialize_tables(tables),'layer_tables':serialize_tables(layer_stats)}
    Path(a.out).write_text(json.dumps(obj),encoding='utf-8')


def eval_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*0.40); test=ticks[split:]
    if a.variant=='PURE':
        s=G75TsugiV2(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'))
    else:
        raw=json.loads(Path(a.table).read_text()); tables=deserialize_tables(raw['tables'])
        s=ExpectedActionV5(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'),tables,
                           a.min_samples,a.z,a.margin,a.variant=='EV_ACTION_SIZER')
    r=engine_run(inst,test,s)
    Path(a.out).write_text(json.dumps({**r,'test_ticks':len(test),'raw_ticks_total':len(ticks)}),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--tf',choices=TF_SEC,required=True); ap.add_argument('--variant',choices=['PURE','EV_ACTION','EV_ACTION_SIZER'],default='PURE')
    ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--worker',choices=['train','eval'])
    ap.add_argument('--out'); ap.add_argument('--table'); ap.add_argument('--min-samples',type=int,default=12)
    ap.add_argument('--z',type=float,default=0.25); ap.add_argument('--margin',type=float,default=0.0); ap.add_argument('--horizon-mult',type=float,default=4.0)
    a=ap.parse_args()
    if a.worker=='train': return train_worker(a)
    if a.worker=='eval': return eval_worker(a)
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    d=Path('results/g75-expected-action-v5')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    tr=d/f'{a.tf}_train.json'; ev=d/f'{a.tf}_{a.variant}_eval.json'
    base=[sys.executable,__file__,'--catalog',a.catalog,'--experiment-id',a.experiment_id,'--tf',a.tf,
          '--raw-bidask-only','--min-samples',str(a.min_samples),'--z',str(a.z),'--margin',str(a.margin),'--horizon-mult',str(a.horizon_mult)]
    subprocess.run(base+['--worker','train','--out',str(tr)],check=True,env=os.environ.copy())
    subprocess.run(base+['--worker','eval','--variant',a.variant,'--table',str(tr),'--out',str(ev)],check=True,env=os.environ.copy())
    train=json.loads(tr.read_text()); out=json.loads(ev.read_text())
    final={**out,'tf':a.tf,'variant_eval':a.variant,'train_ticks':train['train_ticks'],'train_trigger_events':train['train_trigger_events'],
           'split':'40% chronological train / 60% causal OOS',
           'counterfactual_label':'LONG and SHORT basket PnL using add=.025/max10/reversal=.20; train only',
           'candidate_policy':'first G75 trigger per TF bucket consumes bucket even when EV skips',
           'horizon_mult':a.horizon_mult,'min_samples':a.min_samples,'lcb_z':a.z,'ev_margin':a.margin,
           'frozen_opportunity_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},
           'verification_level':'CAUSAL_RAW_BIDASK_EXPECTED_ACTION_V5_BASKET_LABEL_PARITY'}
    (d/f'{a.tf}_{a.variant}.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print(json.dumps(final,indent=2))

if __name__=='__main__': main()
