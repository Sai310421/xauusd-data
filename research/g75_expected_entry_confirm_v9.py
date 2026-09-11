from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import f, load_all
from g75_expected_action_rawtick_v6 import collect_events_continuous, fit_ridge, pred
from g75_tsugi_causal_rawtick_v2 import TF_SEC


def mid(t):
    return (f(t.bid_price)+f(t.ask_price))/2.0

def spread(t):
    return max(0.0,f(t.ask_price)-f(t.bid_price))

def find_pullback_reclaim(ticks,i0,end_i,side,tf_sec,pb_min=0.03,reclaim=0.01,wait_mult=2.0):
    m0=mid(ticks[i0]); deadline=int(ticks[i0].ts_event)+int(tf_sec*wait_mult*1e9)
    saw_pb=False; pb_depth=0.0
    for j in range(i0+1,end_i):
        m=mid(ticks[j]); adverse=(m0-m)*side
        if adverse>pb_depth: pb_depth=adverse
        if adverse>=pb_min: saw_pb=True
        if saw_pb and (m-m0)*side>=reclaim:
            return j,pb_depth
        if int(ticks[j].ts_event)>=deadline: break
    return None,pb_depth

def net_first_passage(ticks,i0,end_i,side,tf_sec,tp_floor=.20,sl_floor=.10,tp_spread_mult=2.5,horizon_mult=8.0):
    t0=ticks[i0]; m0=mid(t0); sp0=spread(t0)
    tp=max(tp_floor,tp_spread_mult*sp0); sl=max(sl_floor,tp/2.0)
    entry=f(t0.ask_price) if side>0 else f(t0.bid_price)
    deadline=int(t0.ts_event)+int(tf_sec*horizon_mult*1e9)
    last_exit=entry; outcome='TIMEOUT'
    for j in range(i0+1,end_i):
        t=ticks[j]; m=mid(t); move=(m-m0)*side
        ex=f(t.bid_price) if side>0 else f(t.ask_price)
        last_exit=ex
        if move>=tp: outcome='TP'; break
        if move<=-sl: outcome='SL'; break
        if int(t.ts_event)>=deadline: break
    pnl=(last_exit-entry)*side
    return pnl,outcome,tp,sl,sp0

def make_confirm_events(ticks,tf_sec,start_i,end_i,pb_min,reclaim,wait_mult):
    base=collect_events_continuous(ticks,tf_sec,start_i,end_i)
    out=[]
    for i,x,trig_side in base:
        j,pbd=find_pullback_reclaim(ticks,i,end_i,trig_side,tf_sec,pb_min,reclaim,wait_mult)
        if j is None: continue
        wait_s=(int(ticks[j].ts_event)-int(ticks[i].ts_event))/1e9
        x2=np.concatenate([np.asarray(x,float),np.array([pbd,wait_s,spread(ticks[j])],float)])
        out.append((j,x2,trig_side))
    return out

def train(ticks,tf_sec,start_i,end_i,a):
    evs=make_confirm_events(ticks,tf_sec,start_i,end_i,a.pb_min,a.reclaim,a.wait_mult)
    X=[]; yl=[]; ys=[]; stats={'L':{},'S':{}}
    for i,x,_ in evs:
        pl,ol,_,_,_=net_first_passage(ticks,i,end_i,1,tf_sec,a.tp_floor,a.sl_floor,a.tp_spread_mult,a.horizon_mult)
        ps,os,_,_,_=net_first_passage(ticks,i,end_i,-1,tf_sec,a.tp_floor,a.sl_floor,a.tp_spread_mult,a.horizon_mult)
        X.append(x); yl.append(pl); ys.append(ps)
        stats['L'][ol]=stats['L'].get(ol,0)+1; stats['S'][os]=stats['S'].get(os,0)+1
    if len(X)<50: raise SystemExit('insufficient confirmed train events')
    return {'long':fit_ridge(X,yl,a.alpha),'short':fit_ridge(X,ys,a.alpha),'n':len(X),
            'long_mean':float(np.mean(yl)),'short_mean':float(np.mean(ys)),'outcomes':stats}

def evaluate(ticks,tf_sec,start_i,end_i,models,a):
    evs=make_confirm_events(ticks,tf_sec,start_i,end_i,a.pb_min,a.reclaim,a.wait_mult)
    rows=[]; accepts=flips=0
    for i,x,trig_side in evs:
        el=pred(models['long'],x)-a.uncertainty_frac*models['long']['sigma']
        es=pred(models['short'],x)-a.uncertainty_frac*models['short']['sigma']
        ev,side=max((el,1),(es,-1),key=lambda z:z[0])
        if ev<=a.margin: continue
        accepts+=1; flips+=int(side!=trig_side)
        pnl,out,tp,sl,sp=net_first_passage(ticks,i,end_i,side,tf_sec,a.tp_floor,a.sl_floor,a.tp_spread_mult,a.horizon_mult)
        rows.append((pnl,out,ev,tp,sl,sp))
    pn=[r[0] for r in rows]; gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0))
    pf=gp/gl if gl else (math.inf if gp else 0.0)
    eq=1000.0; peak=eq; mdd=0.0
    for p in pn:
        eq+=p; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100 if peak>0 else 0)
    return {'confirmed_candidates':len(evs),'ev_accepts':accepts,'accept_rate_pct':100*accepts/max(1,len(evs)),
            'action_flips':flips,'N':len(pn),'WR_pct':100*sum(x>0 for x in pn)/max(1,len(pn)),
            'PF':pf,'expectancy':float(np.mean(pn)) if pn else 0.0,'net':sum(pn),'maxDD_pct':mdd,
            'tp_count':sum(r[1]=='TP' for r in rows),'sl_count':sum(r[1]=='SL' for r in rows),'timeout_count':sum(r[1]=='TIMEOUT' for r in rows),
            'avg_pred_ev':float(np.mean([r[2] for r in rows])) if rows else 0.0,
            'avg_spread':float(np.mean([r[5] for r in rows])) if rows else 0.0}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--experiment-id',required=True); p.add_argument('--tf',choices=TF_SEC,required=True)
    p.add_argument('--raw-bidask-only',action='store_true'); p.add_argument('--pb-min',type=float,default=.03); p.add_argument('--reclaim',type=float,default=.01); p.add_argument('--wait-mult',type=float,default=2.0)
    p.add_argument('--tp-floor',type=float,default=.20); p.add_argument('--sl-floor',type=float,default=.10); p.add_argument('--tp-spread-mult',type=float,default=2.5); p.add_argument('--horizon-mult',type=float,default=8.0)
    p.add_argument('--alpha',type=float,default=10.0); p.add_argument('--uncertainty-frac',type=float,default=.05); p.add_argument('--margin',type=float,default=0.0); a=p.parse_args()
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    inst,ticks=load_all(a.catalog); split=int(len(ticks)*.40); tf=TF_SEC[a.tf]
    m=train(ticks,tf,0,split,a); r=evaluate(ticks,tf,split,len(ticks),m,a)
    spreads=np.array([spread(t) for t in ticks[split:]],float)
    out={**r,'tf':a.tf,'split':'40% chronological train / 60% causal OOS','train_confirmed_events':m['n'],
         'train_long_net_mean':m['long_mean'],'train_short_net_mean':m['short_mean'],'train_outcomes':m['outcomes'],
         'confirmation':'G75 trigger -> pullback >= pb_min -> reclaim >= reclaim -> EV decision','pb_min':a.pb_min,'reclaim':a.reclaim,'wait_mult':a.wait_mult,
         'cost_rule':'TP=max(tp_floor,tp_spread_mult*entry_spread); SL=max(sl_floor,TP/2); executable BidAsk PnL once',
         'tp_floor':a.tp_floor,'sl_floor':a.sl_floor,'tp_spread_mult':a.tp_spread_mult,
         'spread_p50':float(np.quantile(spreads,.5)),'spread_p75':float(np.quantile(spreads,.75)),'spread_p90':float(np.quantile(spreads,.9)),
         'frozen_opportunity_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},
         'verification_level':'CAUSAL_RAW_BIDASK_G75_TWO_STAGE_EXPECTED_ENTRY_V9'}
    d=Path('results/g75-expected-entry-v9')/a.experiment_id; d.mkdir(parents=True,exist_ok=True); (d/f'{a.tf}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
