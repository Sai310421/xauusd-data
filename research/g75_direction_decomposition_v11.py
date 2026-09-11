from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import load_all
from g75_expected_action_rawtick_v6 import pred
from g75_expected_entry_adaptive_v10 import causal_events, net_first_passage, train
from g75_tsugi_causal_rawtick_v2 import TF_SEC


def summarize(rows):
    pn=[r['pnl'] for r in rows]
    gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0))
    pf=gp/gl if gl else (math.inf if gp else 0.0)
    eq=1000.0; peak=eq; mdd=0.0
    for p in pn:
        eq+=p; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100 if peak>0 else 0.0)
    return {
        'N':len(pn), 'WR_pct':100*sum(x>0 for x in pn)/max(1,len(pn)), 'PF':pf,
        'expectancy':float(np.mean(pn)) if pn else 0.0, 'net':float(sum(pn)), 'maxDD_pct':mdd,
        'tp_count':sum(r['outcome']=='TP' for r in rows),
        'sl_count':sum(r['outcome']=='SL' for r in rows),
        'timeout_count':sum(r['outcome']=='TIMEOUT' for r in rows),
        'avg_spread':float(np.mean([r['spread'] for r in rows])) if rows else 0.0,
        'avg_trigger':float(np.mean([r['trigger'] for r in rows])) if rows else 0.0,
    }


def eval_policy(ticks,tf_sec,start_i,end_i,m,a,policy):
    evs=causal_events(ticks,tf_sec,start_i,end_i,a.trigger_mult,a.spread_window,a.pb_frac,a.reclaim_frac,a.wait_mult)
    rows=[]; positive_scores=0; flips=0
    for i,x,trig_side,trig,_ in evs:
        el=pred(m['long'],x)-a.uncertainty_frac*m['long']['sigma']
        es=pred(m['short'],x)-a.uncertainty_frac*m['short']['sigma']
        if policy=='FOLLOW':
            side=trig_side; score=el if side>0 else es
        elif policy=='FADE':
            side=-trig_side; score=el if side>0 else es
        elif policy=='EV_ALL':
            score,side=max((el,1),(es,-1),key=lambda z:z[0])
        elif policy=='EV_POS':
            score,side=max((el,1),(es,-1),key=lambda z:z[0])
            if score<=a.margin: continue
        else:
            raise ValueError(policy)
        positive_scores += int(score>0)
        flips += int(side!=trig_side)
        pnl,out,tp,sl,sp=net_first_passage(ticks,i,end_i,side,tf_sec,trig,a.tp_mult,a.cost_mult,a.horizon_mult)
        rows.append({'pnl':pnl,'outcome':out,'score':score,'spread':sp,'trigger':trig,'side':side,'trig_side':trig_side})
    s=summarize(rows)
    s.update({
        'policy':policy,'confirmed_candidates':len(evs),'positive_score_count':positive_scores,
        'positive_score_rate_pct':100*positive_scores/max(1,len(rows)),
        'action_flips':flips,
        'avg_pred_score':float(np.mean([r['score'] for r in rows])) if rows else 0.0,
    })
    return s


def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--experiment-id',required=True)
    p.add_argument('--tf',choices=TF_SEC,default='M1'); p.add_argument('--raw-bidask-only',action='store_true')
    p.add_argument('--trigger-mults',default='1,1.5,2,2.5'); p.add_argument('--spread-window',type=int,default=256)
    p.add_argument('--pb-frac',type=float,default=.25); p.add_argument('--reclaim-frac',type=float,default=.10); p.add_argument('--wait-mult',type=float,default=2.0)
    p.add_argument('--tp-mult',type=float,default=2.0); p.add_argument('--cost-mult',type=float,default=2.5); p.add_argument('--horizon-mult',type=float,default=8.0)
    p.add_argument('--alpha',type=float,default=10.0); p.add_argument('--uncertainty-frac',type=float,default=.05); p.add_argument('--margin',type=float,default=0.0); a=p.parse_args()
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    _,ticks=load_all(a.catalog); split=int(len(ticks)*.40); tf=TF_SEC[a.tf]
    results=[]
    for km in [float(x) for x in a.trigger_mults.split(',')]:
        a.trigger_mult=km
        m=train(ticks,tf,0,split,a)
        for policy in ('FOLLOW','FADE','EV_ALL','EV_POS'):
            r=eval_policy(ticks,tf,split,len(ticks),m,a,policy)
            r.update({'tf':a.tf,'trigger_mult':km,'train_events':m['n'],'train_long_net_mean':m['long_mean'],'train_short_net_mean':m['short_mean']})
            results.append(r)
    out={
        'experiment_id':a.experiment_id,
        'purpose':'Separate opportunity/confirmation edge from direction-model edge before further optimization',
        'split':'40% chronological train / 60% causal OOS',
        'policies':['FOLLOW trigger direction','FADE trigger direction','EV_ALL choose higher model score without gating','EV_POS choose higher model score only if score > 0'],
        'results':results,
        'verification_level':'CAUSAL_RAW_BIDASK_G75_DIRECTION_DECOMPOSITION_V11'
    }
    d=Path('results/g75-direction-v11')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    (d/f'{a.tf}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
