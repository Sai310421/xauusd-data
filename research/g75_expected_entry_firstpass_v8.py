from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

from g75_tsugi_causal_rawtick_v2 import G75TsugiV2, Cfg, TF_SEC
from g75_expected_action_rawtick_v5 import f, load_all, engine_run
from g75_expected_action_rawtick_v6 import collect_events_continuous, fit_ridge, pred, _std


def first_passage_net_pnl(ticks, i0, end_i, side, tf_sec, tp_move=0.20, sl_move=0.10, horizon_mult=8.0):
    """Spread-correct first passage.

    Barriers are defined on MID movement from the trigger, not executable PnL.
    PnL is then calculated from executable Ask/Bid, so spread is charged exactly once.
    This avoids the v7 pathology where a SL smaller than spread stopped almost every
    trade immediately and spread was then subtracted a second time in EV.
    """
    t0 = ticks[i0]
    bid0 = f(t0.bid_price); ask0 = f(t0.ask_price); mid0 = (bid0 + ask0) / 2.0
    entry = ask0 if side > 0 else bid0
    start_ns = int(t0.ts_event)
    deadline = start_ns + int(tf_sec * horizon_mult * 1_000_000_000)
    last_bid, last_ask = bid0, ask0
    outcome = 'TIMEOUT'
    for j in range(i0 + 1, end_i):
        t = ticks[j]
        bid = f(t.bid_price); ask = f(t.ask_price); mid = (bid + ask) / 2.0
        last_bid, last_ask = bid, ask
        move = (mid - mid0) * side
        if move >= tp_move:
            outcome = 'TP'; break
        if move <= -sl_move:
            outcome = 'SL'; break
        if int(t.ts_event) >= deadline:
            break
    exit_px = last_bid if side > 0 else last_ask
    net_pnl = (exit_px - entry) * side
    return float(net_pnl), outcome


def train_models(ticks, tf_sec, start_i, end_i, tp_move, sl_move, horizon_mult, alpha):
    events = collect_events_continuous(ticks, tf_sec, start_i, end_i)
    X, yl, ys = [], [], []
    out_l = {'TP':0,'SL':0,'TIMEOUT':0}; out_s = {'TP':0,'SL':0,'TIMEOUT':0}
    for i, x, _ in events:
        pl, ol = first_passage_net_pnl(ticks, i, end_i, 1, tf_sec, tp_move, sl_move, horizon_mult)
        ps, os_ = first_passage_net_pnl(ticks, i, end_i, -1, tf_sec, tp_move, sl_move, horizon_mult)
        X.append(x); yl.append(pl); ys.append(ps); out_l[ol]+=1; out_s[os_]+=1
    if len(X) < 50:
        raise SystemExit('insufficient train events')
    return {
        'long': fit_ridge(X, yl, alpha),
        'short': fit_ridge(X, ys, alpha),
        'train_trigger_events': len(X),
        'train_long_net_mean': float(np.mean(yl)),
        'train_short_net_mean': float(np.mean(ys)),
        'train_long_positive_rate': float(np.mean(np.asarray(yl) > 0)),
        'train_short_positive_rate': float(np.mean(np.asarray(ys) > 0)),
        'train_long_outcomes': out_l,
        'train_short_outcomes': out_s,
    }


class ExpectedEntryV8(G75TsugiV2):
    def __init__(self, cfg, models, margin=0.0, uncertainty_frac=0.05, sizing=False):
        super().__init__(cfg)
        self.models = models; self.margin = margin; self.uncertainty_frac = uncertainty_frac; self.sizing = sizing
        self.ev_accepts = self.ev_skips = self.action_flips = 0
        self.current_ev = 0.0; self.pred_ev_sum = 0.0; self.pred_ev_max = -1e9; self.pred_ev_min = 1e9
    def _x(self, trigger_side, mid, bid, ask, t):
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
        if not self.sizing: return base
        # EV is net price-PnL per 0.01-lot-equivalent basket seed.
        if self.current_ev < 0.02: return max(1, round(base*.3))
        if self.current_ev < 0.05: return max(1, round(base*.6))
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
                raw_l=pred(self.models['long'],x); raw_s=pred(self.models['short'],x)
                # Residual sigma penalty only; spread is already embedded exactly once in labels.
                ev_l=raw_l-self.uncertainty_frac*self.models['long']['sigma']
                ev_s=raw_s-self.uncertainty_frac*self.models['short']['sigma']
                ev,side=max((ev_l,1),(ev_s,-1),key=lambda z:z[0])
                self.pred_ev_sum+=ev; self.pred_ev_max=max(self.pred_ev_max,ev); self.pred_ev_min=min(self.pred_ev_min,ev)
                if ev<=self.margin:
                    self.ev_skips+=1; return
                self.ev_accepts+=1; self.current_ev=ev
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
        x=super().summary(); n=max(1,self.trigger_candidates)
        x.update({'ev_accepts':self.ev_accepts,'ev_skips':self.ev_skips,'action_flips':self.action_flips,
                  'ev_sizing':self.sizing,'decision_parity_ok':self.trigger_candidates==(self.ev_accepts+self.ev_skips),
                  'pred_ev_avg':self.pred_ev_sum/n,'pred_ev_max':self.pred_ev_max if self.trigger_candidates else None,
                  'pred_ev_min':self.pred_ev_min if self.trigger_candidates else None,
                  'accept_rate_pct':100*self.ev_accepts/n})
        return x


def train_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*.40)
    models=train_models(ticks,TF_SEC[a.tf],0,split,a.tp_move,a.sl_move,a.horizon_mult,a.alpha)
    Path(a.out).write_text(json.dumps({'raw_ticks_total':len(ticks),'train_ticks':split,'models':models}),encoding='utf-8')


def eval_worker(a):
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*.40); test=ticks[split:]
    if a.variant=='PURE':
        s=G75TsugiV2(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'))
    else:
        raw=json.loads(Path(a.model).read_text()); models=raw['models']
        s=ExpectedEntryV8(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'),models,a.margin,a.uncertainty_frac,a.variant=='EV_ENTRY_SIZER')
    r=engine_run(inst,test,s); Path(a.out).write_text(json.dumps({**r,'test_ticks':len(test),'raw_ticks_total':len(ticks)}),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--tf',choices=TF_SEC,required=True); ap.add_argument('--variant',choices=['PURE','EV_ENTRY','EV_ENTRY_SIZER'],default='PURE')
    ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--worker',choices=['train','eval']); ap.add_argument('--out'); ap.add_argument('--model')
    ap.add_argument('--tp-move',type=float,default=.20); ap.add_argument('--sl-move',type=float,default=.10); ap.add_argument('--margin',type=float,default=0.0)
    ap.add_argument('--uncertainty-frac',type=float,default=.05); ap.add_argument('--alpha',type=float,default=10.0); ap.add_argument('--horizon-mult',type=float,default=8.0)
    a=ap.parse_args()
    if a.worker=='train': return train_worker(a)
    if a.worker=='eval': return eval_worker(a)
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    d=Path('results/g75-expected-entry-v8')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    tr=d/f'{a.tf}_train.json'; ev=d/f'{a.tf}_{a.variant}_eval.json'
    base=[sys.executable,__file__,'--catalog',a.catalog,'--experiment-id',a.experiment_id,'--tf',a.tf,'--raw-bidask-only',
          '--tp-move',str(a.tp_move),'--sl-move',str(a.sl_move),'--margin',str(a.margin),'--uncertainty-frac',str(a.uncertainty_frac),'--alpha',str(a.alpha),'--horizon-mult',str(a.horizon_mult)]
    subprocess.run(base+['--worker','train','--out',str(tr)],check=True,env=os.environ.copy())
    subprocess.run(base+['--worker','eval','--variant',a.variant,'--model',str(tr),'--out',str(ev)],check=True,env=os.environ.copy())
    train=json.loads(tr.read_text()); out=json.loads(ev.read_text()); m=train['models']
    final={**out,'tf':a.tf,'variant_eval':a.variant,'train_ticks':train['train_ticks'],'train_trigger_events':m['train_trigger_events'],
           'split':'40% chronological train / 60% causal OOS','ev_target':'spread-correct MID first-passage, executable net PnL regression',
           'tp_move':a.tp_move,'sl_move':a.sl_move,'move_rr':a.tp_move/a.sl_move,'alpha':a.alpha,'uncertainty_frac':a.uncertainty_frac,'ev_margin':a.margin,
           'train_long_net_mean':m['train_long_net_mean'],'train_short_net_mean':m['train_short_net_mean'],
           'train_long_positive_rate':m['train_long_positive_rate'],'train_short_positive_rate':m['train_short_positive_rate'],
           'train_long_outcomes':m['train_long_outcomes'],'train_short_outcomes':m['train_short_outcomes'],
           'frozen_opportunity_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},
           'verification_level':'CAUSAL_RAW_BIDASK_G75_FIRST_PASSAGE_EV_V8_SPREAD_CORRECT'}
    (d/f'{a.tf}_{a.variant}.json').write_text(json.dumps(final,indent=2),encoding='utf-8'); print(json.dumps(final,indent=2))

if __name__=='__main__': main()
