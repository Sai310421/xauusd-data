from __future__ import annotations
import argparse, json, math
from collections import deque
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import f, load_all
from g75_expected_action_rawtick_v6 import fit_ridge, pred
from g75_tsugi_causal_rawtick_v2 import TF_SEC


def mid(t):
    return (f(t.bid_price) + f(t.ask_price)) / 2.0


def spread(t):
    return max(0.0, f(t.ask_price) - f(t.bid_price))


def causal_events(ticks, tf_sec, start_i, end_i, trigger_mult, spread_window, pb_frac, reclaim_frac, wait_mult):
    out=[]; spr=deque(maxlen=spread_window)
    bucket=None; anchor=None; consumed=False
    for i in range(start_i,end_i):
        t=ticks[i]; m=mid(t); sp=spread(t); spr.append(sp)
        b=(int(t.ts_event)//1_000_000_000)//tf_sec
        if bucket is None or b!=bucket:
            bucket=b; anchor=m; consumed=False
        if consumed or anchor is None or len(spr)<max(32,spread_window//4):
            continue
        med=float(np.median(np.asarray(spr,float)))
        trig=max(0.12, trigger_mult*med)
        side=1 if m>=anchor+trig else (-1 if m<=anchor-trig else 0)
        if not side: continue
        consumed=True
        pb=max(0.03,pb_frac*trig); rec=max(0.01,reclaim_frac*trig)
        deadline=int(t.ts_event)+int(tf_sec*wait_mult*1e9)
        base=m; saw=False; depth=0.0; j=None
        for k in range(i+1,end_i):
            mk=mid(ticks[k]); adv=(base-mk)*side; depth=max(depth,adv)
            if adv>=pb: saw=True
            if saw and (mk-base)*side>=rec:
                j=k; break
            if int(ticks[k].ts_event)>=deadline: break
        if j is None: continue
        tj=ticks[j]; mj=mid(tj); spj=spread(tj); wait_s=(int(tj.ts_event)-int(t.ts_event))/1e9
        # Causal features normalized by adaptive trigger/cost scale.
        sec=(int(tj.ts_event)//1_000_000_000)%86400; ang=2*np.pi*sec/86400.0
        x=np.array([side,trig,med,spj,depth/max(trig,1e-9),wait_s/max(tf_sec,1),
                    (mj-base)*side/max(trig,1e-9),np.sin(ang),np.cos(ang)],float)
        out.append((j,x,side,trig,med))
    return out


def net_first_passage(ticks,i0,end_i,side,tf_sec,trig,tp_mult,cost_mult,horizon_mult):
    t0=ticks[i0]; m0=mid(t0); sp0=spread(t0)
    tp=max(tp_mult*trig,cost_mult*sp0); sl=tp/2.0
    entry=f(t0.ask_price) if side>0 else f(t0.bid_price)
    deadline=int(t0.ts_event)+int(tf_sec*horizon_mult*1e9)
    last_exit=entry; outcome='TIMEOUT'
    for j in range(i0+1,end_i):
        t=ticks[j]; m=mid(t); move=(m-m0)*side
        last_exit=f(t.bid_price) if side>0 else f(t.ask_price)
        if move>=tp: outcome='TP'; break
        if move<=-sl: outcome='SL'; break
        if int(t.ts_event)>=deadline: break
    return (last_exit-entry)*side,outcome,tp,sl,sp0


def train(ticks,tf_sec,start_i,end_i,a):
    evs=causal_events(ticks,tf_sec,start_i,end_i,a.trigger_mult,a.spread_window,a.pb_frac,a.reclaim_frac,a.wait_mult)
    X=[]; yl=[]; ys=[]; stats={'L':{},'S':{}}
    for i,x,_,trig,_ in evs:
        pl,ol,_,_,_=net_first_passage(ticks,i,end_i,1,tf_sec,trig,a.tp_mult,a.cost_mult,a.horizon_mult)
        ps,os,_,_,_=net_first_passage(ticks,i,end_i,-1,tf_sec,trig,a.tp_mult,a.cost_mult,a.horizon_mult)
        X.append(x); yl.append(pl); ys.append(ps)
        stats['L'][ol]=stats['L'].get(ol,0)+1; stats['S'][os]=stats['S'].get(os,0)+1
    if len(X)<50: raise SystemExit('insufficient adaptive train events')
    return {'long':fit_ridge(X,yl,a.alpha),'short':fit_ridge(X,ys,a.alpha),'n':len(X),
            'long_mean':float(np.mean(yl)),'short_mean':float(np.mean(ys)),'outcomes':stats}


def evaluate(ticks,tf_sec,start_i,end_i,m,a):
    evs=causal_events(ticks,tf_sec,start_i,end_i,a.trigger_mult,a.spread_window,a.pb_frac,a.reclaim_frac,a.wait_mult)
    rows=[]; flips=0
    for i,x,trig_side,trig,_ in evs:
        el=pred(m['long'],x)-a.uncertainty_frac*m['long']['sigma']
        es=pred(m['short'],x)-a.uncertainty_frac*m['short']['sigma']
        ev,side=max((el,1),(es,-1),key=lambda z:z[0])
        if ev<=a.margin: continue
        flips+=int(side!=trig_side)
        pnl,out,tp,sl,sp=net_first_passage(ticks,i,end_i,side,tf_sec,trig,a.tp_mult,a.cost_mult,a.horizon_mult)
        rows.append((pnl,out,ev,tp,sl,sp,trig))
    pn=[r[0] for r in rows]; gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0)); pf=gp/gl if gl else (math.inf if gp else 0.0)
    eq=1000.; peak=eq; mdd=0.
    for p in pn:
        eq+=p; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100 if peak>0 else 0)
    return {'confirmed_candidates':len(evs),'ev_accepts':len(rows),'accept_rate_pct':100*len(rows)/max(1,len(evs)),
            'action_flips':flips,'N':len(pn),'WR_pct':100*sum(x>0 for x in pn)/max(1,len(pn)),
            'PF':pf,'expectancy':float(np.mean(pn)) if pn else 0.0,'net':sum(pn),'maxDD_pct':mdd,
            'tp_count':sum(r[1]=='TP' for r in rows),'sl_count':sum(r[1]=='SL' for r in rows),'timeout_count':sum(r[1]=='TIMEOUT' for r in rows),
            'avg_pred_ev':float(np.mean([r[2] for r in rows])) if rows else 0.0,
            'avg_spread':float(np.mean([r[5] for r in rows])) if rows else 0.0,
            'avg_trigger':float(np.mean([r[6] for r in rows])) if rows else 0.0}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--experiment-id',required=True); p.add_argument('--tf',choices=TF_SEC,required=True)
    p.add_argument('--raw-bidask-only',action='store_true'); p.add_argument('--trigger-mult',type=float,required=True); p.add_argument('--spread-window',type=int,default=256)
    p.add_argument('--pb-frac',type=float,default=.25); p.add_argument('--reclaim-frac',type=float,default=.10); p.add_argument('--wait-mult',type=float,default=2.0)
    p.add_argument('--tp-mult',type=float,default=2.0); p.add_argument('--cost-mult',type=float,default=2.5); p.add_argument('--horizon-mult',type=float,default=8.0)
    p.add_argument('--alpha',type=float,default=10.0); p.add_argument('--uncertainty-frac',type=float,default=.05); p.add_argument('--margin',type=float,default=0.0); a=p.parse_args()
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    _,ticks=load_all(a.catalog); split=int(len(ticks)*.40); tf=TF_SEC[a.tf]
    m=train(ticks,tf,0,split,a); r=evaluate(ticks,tf,split,len(ticks),m,a)
    out={**r,'tf':a.tf,'trigger_mult':a.trigger_mult,'spread_window':a.spread_window,
         'split':'40% chronological train / 60% causal OOS','train_events':m['n'],'train_long_net_mean':m['long_mean'],'train_short_net_mean':m['short_mean'],'train_outcomes':m['outcomes'],
         'opportunity_rule':'trigger=max(0.12, trigger_mult*causal_rolling_median_spread)',
         'confirmation':'adaptive trigger -> pullback=max(0.03,pb_frac*trigger) -> reclaim=max(0.01,reclaim_frac*trigger) -> EV action',
         'exit_rule':'TP=max(tp_mult*trigger,cost_mult*entry_spread); SL=TP/2; executable BidAsk once',
         'pb_frac':a.pb_frac,'reclaim_frac':a.reclaim_frac,'tp_mult':a.tp_mult,'cost_mult':a.cost_mult,
         'legacy_reference':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},
         'verification_level':'CAUSAL_RAW_BIDASK_G75_ADAPTIVE_SPREAD_EXPECTED_ENTRY_V10'}
    d=Path('results/g75-expected-entry-v10')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    tag=str(a.trigger_mult).replace('.','p'); (d/f'{a.tf}_K{tag}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
