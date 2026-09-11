from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import load_all
from g75_expected_entry_adaptive_v10 import causal_events, net_first_passage
from g75_tsugi_causal_rawtick_v2 import TF_SEC


def summarize(rows):
    pn=[r['pnl'] for r in rows]
    gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0))
    pf=gp/gl if gl else (math.inf if gp else 0.0)
    eq=1000.0; peak=eq; mdd=0.0
    for p in pn:
        eq+=p; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100 if peak>0 else 0.0)
    return {'N':len(pn),'WR_pct':100*sum(x>0 for x in pn)/max(1,len(pn)),'PF':pf,
            'expectancy':float(np.mean(pn)) if pn else 0.0,'net':float(sum(pn)),'maxDD_pct':mdd}


def path_features(ticks,i0,end_i,side,tf_sec,horizon_mult=2.0):
    m0=(float(ticks[i0].bid_price)+float(ticks[i0].ask_price))/2.0
    deadline=int(ticks[i0].ts_event)+int(tf_sec*horizon_mult*1e9)
    fav=adv=0.0; travel=0.0; prev=m0; last_i=i0
    for j in range(i0+1,end_i):
        m=(float(ticks[j].bid_price)+float(ticks[j].ask_price))/2.0
        d=(m-m0)*side; fav=max(fav,d); adv=max(adv,-d); travel+=abs(m-prev); prev=m; last_i=j
        if int(ticks[j].ts_event)>=deadline: break
    elapsed=max(1e-9,(int(ticks[last_i].ts_event)-int(ticks[i0].ts_event))/1e9)
    efficiency=(fav-adv)/max(travel,1e-9)
    return fav,adv,travel,elapsed,efficiency


def bucket(v,edges):
    return int(np.digitize([v],edges)[0])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--experiment-id',required=True)
    p.add_argument('--tf',choices=TF_SEC,default='M1'); p.add_argument('--raw-bidask-only',action='store_true')
    p.add_argument('--trigger-mults',default='1,1.5,2,2.5'); p.add_argument('--spread-window',type=int,default=256)
    p.add_argument('--pb-fracs',default='.15,.25,.35'); p.add_argument('--reclaim-fracs',default='.05,.10,.20')
    p.add_argument('--wait-mults',default='1,2,4'); p.add_argument('--tp-mults',default='1.5,2,3'); p.add_argument('--cost-mult',type=float,default=2.5)
    p.add_argument('--horizon-mults',default='4,8,12'); p.add_argument('--risk-fracs',default='.25,.5,1.0'); a=p.parse_args()
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    _,ticks=load_all(a.catalog); split=int(len(ticks)*.40); tf=TF_SEC[a.tf]
    configs=[]
    for km in map(float,a.trigger_mults.split(',')):
      for pb in map(float,a.pb_fracs.split(',')):
       for rc in map(float,a.reclaim_fracs.split(',')):
        for wt in map(float,a.wait_mults.split(',')):
         evs=causal_events(ticks,tf,0,split,km,a.spread_window,pb,rc,wt)
         if len(evs)<80: continue
         train_rows=[]
         for i,x,trig_side,trig,_ in evs:
            for side in (trig_side,-trig_side):
                fav,adv,travel,elapsed,eff=path_features(ticks,i,split,side,tf,2.0)
                sp=float(ticks[i].ask_price)-float(ticks[i].bid_price)
                train_rows.append((i,side,trig,trig_side,sp,fav,adv,travel,elapsed,eff))
         # Discrete positive-region map over direction x distance x time x path x cost.
         regions={}
         for i,side,trig,trig_side,sp,fav,adv,travel,elapsed,eff in train_rows:
            for tp_mult in map(float,a.tp_mults.split(',')):
             for hz in map(float,a.horizon_mults.split(',')):
                pnl,out,_,_,_=net_first_passage(ticks,i,split,side,tf,trig,tp_mult,a.cost_mult,hz)
                key=(int(side==trig_side),bucket(trig,[.6,1.0,1.5]),bucket(elapsed/tf,[.5,1,2]),bucket(eff,[-.2,0,.2]),bucket(sp/max(trig,1e-9),[.25,.5,.75]),tp_mult,hz)
                z=regions.setdefault(key,[]); z.append(pnl)
         positive={k:(float(np.mean(v)),len(v),float(np.std(v))) for k,v in regions.items() if len(v)>=20 and np.mean(v)>0}
         # OOS: choose best matched positive region; otherwise skip.
         oos=causal_events(ticks,tf,split,len(ticks),km,a.spread_window,pb,rc,wt)
         rows=[]
         for i,x,trig_side,trig,_ in oos:
            candidates=[]
            for side in (trig_side,-trig_side):
                fav,adv,travel,elapsed,eff=path_features(ticks,i,len(ticks),side,tf,2.0)
                sp=float(ticks[i].ask_price)-float(ticks[i].bid_price)
                for tp_mult in map(float,a.tp_mults.split(',')):
                 for hz in map(float,a.horizon_mults.split(',')):
                    key=(int(side==trig_side),bucket(trig,[.6,1.0,1.5]),bucket(elapsed/tf,[.5,1,2]),bucket(eff,[-.2,0,.2]),bucket(sp/max(trig,1e-9),[.25,.5,.75]),tp_mult,hz)
                    if key in positive:
                        mu,n,sd=positive[key]; score=mu-0.25*sd/math.sqrt(n)
                        candidates.append((score,side,tp_mult,hz))
            if not candidates: continue
            score,side,tp_mult,hz=max(candidates,key=lambda z:z[0])
            pnl,out,_,_,_=net_first_passage(ticks,i,len(ticks),side,tf,trig,tp_mult,a.cost_mult,hz)
            rows.append({'pnl':pnl,'score':score,'side':side,'tp_mult':tp_mult,'hz':hz})
         base=summarize(rows)
         # Capital dimension: scale only after positive-region selection.
         for rf in map(float,a.risk_fracs.split(',')):
            scaled=[{'pnl':r['pnl']*rf} for r in rows]
            s=summarize(scaled); s.update({'trigger_mult':km,'pb_frac':pb,'reclaim_frac':rc,'wait_mult':wt,'risk_frac':rf,
                'positive_regions':len(positive),'accepted':len(rows),'tf':a.tf})
            configs.append(s)
    configs.sort(key=lambda r:(r['expectancy']>0,r['PF'],r['net'],-r['maxDD_pct']),reverse=True)
    out={'experiment_id':a.experiment_id,
         'objective':'Search positive subregions in direction x distance x time x path x cost, then apply capital sizing only after edge is positive',
         'dimensions':['direction','distance','time','path','cost','capital'],
         'split':'40% chronological train / 60% causal OOS','top20':configs[:20],'all_count':len(configs),
         'verification_level':'CAUSAL_RAW_BIDASK_G75_SIXDIM_POSITIVE_REGION_V12'}
    d=Path('results/g75-sixdim-v12')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    (d/f'{a.tf}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
