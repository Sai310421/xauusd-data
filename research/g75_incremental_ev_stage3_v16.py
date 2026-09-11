from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import f, load_all
from g75_expected_entry_adaptive_v10 import causal_events, mid, spread
from g75_tsugi_causal_rawtick_v2 import TF_SEC
from g75_three_stage_atr_v13 import build_atr_map, atr_at, exec_entry, exec_exit

ACTIONS=('EXIT','HOLD','ADD','REVERSE')

def bindex(x,cuts):
    for i,c in enumerate(cuts):
        if x<c:return i
    return len(cuts)

def state_key(mfe_r,mae_r,elapsed_tf,cost_ratio,km,dmode,level=0):
    core=(bindex(mfe_r,[.25,.5,1,1.5]),bindex(abs(mae_r),[.5,1,1.5,2]),bindex(elapsed_tf,[.5,1,2,4]),bindex(cost_ratio,[.25,.5,1]),round(km,2),dmode)
    if level==0:return core
    if level==1:return core[:2]+core[4:]
    return core[4:]

def basket_pnl(t,entries):
    pl=0.; w=0.
    for e,wt,sd in entries:
        px=f(t.bid_price) if sd>0 else f(t.ask_price)
        pl+=(px-e)*sd*wt; w+=wt
    return pl,w

def continue_from(ticks,j0,end_i,entries,side,tf,trig,amap,a,deadline,runner_mult):
    peak=-1e99; be=lock=runner=False; last=j0
    hard=max(a.hard_sl_mult*trig,a.cost_mult*spread(ticks[j0])/2.0)
    for j in range(j0+1,end_i):
        t=ticks[j]; last=j; pnl,w=basket_pnl(t,entries); risk=max(hard*max(w,1e-9),1e-9); r=pnl/risk; peak=max(peak,pnl)
        be |= r>=a.be_arm_r; lock |= r>=a.lock1_r; runner |= r>=a.runner_arm_r
        if pnl<=-risk: break
        if be and r<=a.be_floor_r: break
        if lock and r<=a.lock1_floor_r: break
        if runner:
            atr=atr_at(t,tf,amap,max(trig,spread(t)))
            floor=max(a.runner_floor_r*risk,peak-runner_mult*atr*max(w,1e-9))
            if pnl<=floor: break
        if int(t.ts_event)>=deadline: break
    return basket_pnl(ticks[last],entries)[0]

def event_counterfactuals(ticks,i0,end_i,side,tf,trig,amap,a,km,dmode,runner_mult):
    t0=ticks[i0]; m0=mid(t0); deadline=int(t0.ts_event)+int(tf*a.horizon_mult*1e9)
    entries=[(exec_entry(t0,side),a.w1,side)]; stage=1; adv=0.; anchor=None; mfe=0.; mae=0.; hard=max(a.hard_sl_mult*trig,a.cost_mult*spread(t0)/2.0)
    for j in range(i0+1,end_i):
        t=ticks[j]; mv=(mid(t)-m0)*side; mfe=max(mfe,mv); mae=min(mae,mv); adv=max(adv,-mv)
        if stage==1:
            if anchor is None and adv>=a.add2_mult*trig: anchor=mid(t)
            if anchor is not None and (mid(t)-anchor)*side>=a.reclaim_add_mult*trig:
                entries.append((exec_entry(t,side),a.w2,side)); stage=2; anchor=None
        elif stage==2:
            if anchor is None and adv>=a.add3_mult*trig: anchor=mid(t)
            if anchor is not None and (mid(t)-anchor)*side>=a.reclaim_add_mult*trig:
                pnl_now,w=basket_pnl(t,entries); risk=max(hard*w,1e-9)
                elapsed=(int(t.ts_event)-int(t0.ts_event))/1e9/max(tf,1)
                keyvals=(mfe/risk,mae/risk,elapsed,spread(t)/max(atr_at(t,tf,amap,trig),1e-9))
                vals={'EXIT':pnl_now}
                vals['HOLD']=continue_from(ticks,j,end_i,list(entries),side,tf,trig,amap,a,deadline,runner_mult)
                add=list(entries)+[(exec_entry(t,side),a.w3,side)]
                vals['ADD']=continue_from(ticks,j,end_i,add,side,tf,trig,amap,a,deadline,runner_mult)
                # True reverse: crystallize current basket now, then open a fresh opposite leg; no alias with hedge.
                rev_entry=exec_entry(t,-side); rev=[(rev_entry,w,-side)]
                vals['REVERSE']=pnl_now+continue_from(ticks,j,end_i,rev,-side,tf,trig,amap,a,deadline,runner_mult)
                return keyvals,vals
        pnl,w=basket_pnl(t,entries); risk=max(hard*w,1e-9)
        if pnl<=-risk or int(t.ts_event)>=deadline: break
    return None,None

def stats(xs,z):
    n=len(xs)
    if not n:return (0,-1e99,0.)
    m=float(np.mean(xs)); se=float(np.std(xs,ddof=1)/math.sqrt(n)) if n>1 else 1e9
    return n,m-z*se,m

def learn(ticks,events,end_i,tf,amap,a,km,dmode,runner):
    db=defaultdict(lambda:defaultdict(list))
    for i,_,trig_side,trig,_ in events:
        side=trig_side if dmode=='FOLLOW' else -trig_side
        st,vals=event_counterfactuals(ticks,i,end_i,side,tf,trig,amap,a,km,dmode,runner)
        if st is None: continue
        base=vals['EXIT']
        for lvl in range(3):
            k=state_key(*st,km,dmode,level=lvl)
            for ac in ACTIONS: db[(lvl,k)][ac].append(vals[ac]-base)
    return db

def choose(db,st,km,dmode,a):
    best=('EXIT',0.)
    for lvl in range(3):
        k=state_key(*st,km,dmode,level=lvl); bucket=db.get((lvl,k),{})
        candidates=[]
        for ac in ACTIONS:
            n,lcb,m=stats(bucket.get(ac,[]),a.lcb_z)
            if n>=a.min_samples: candidates.append((lcb,ac,n,m))
        if candidates:
            lcb,ac,n,m=max(candidates)
            return (ac,lcb,n,m) if lcb>0 else ('EXIT',lcb,n,m)
    return ('EXIT',0.,0,0.)

def summarize(pn):
    n=len(pn); gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0)); pf=gp/gl if gl else (math.inf if gp else 0.)
    eq=1000.; peak=eq; dd=0.
    for p in pn: eq+=p; peak=max(peak,eq); dd=max(dd,(peak-eq)/peak*100 if peak>0 else 0.)
    return {'N':n,'WR_pct':100*sum(x>0 for x in pn)/max(1,n),'PF':pf,'expectancy':float(np.mean(pn)) if pn else 0.,'net':float(sum(pn)),'maxDD_pct':dd}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True);p.add_argument('--experiment-id',required=True);p.add_argument('--tf',choices=TF_SEC,default='M1');p.add_argument('--raw-bidask-only',action='store_true')
    p.add_argument('--trigger-mults',default='1.5,2');p.add_argument('--directions',default='FOLLOW,FADE');p.add_argument('--runner-mult',type=float,default=3.0);p.add_argument('--min-samples',type=int,default=20);p.add_argument('--lcb-z',type=float,default=.5)
    p.add_argument('--spread-window',type=int,default=256);p.add_argument('--pb-frac',type=float,default=.25);p.add_argument('--reclaim-frac',type=float,default=.10);p.add_argument('--wait-mult',type=float,default=2.)
    p.add_argument('--w1',type=float,default=.30);p.add_argument('--w2',type=float,default=.30);p.add_argument('--w3',type=float,default=.40);p.add_argument('--add2-mult',type=float,default=.50);p.add_argument('--add3-mult',type=float,default=1.0);p.add_argument('--reclaim-add-mult',type=float,default=.15)
    p.add_argument('--cost-mult',type=float,default=2.5);p.add_argument('--hard-sl-mult',type=float,default=2.0);p.add_argument('--horizon-mult',type=float,default=12.)
    p.add_argument('--be-arm-r',type=float,default=.5);p.add_argument('--be-floor-r',type=float,default=.05);p.add_argument('--lock1-r',type=float,default=1.0);p.add_argument('--lock1-floor-r',type=float,default=.35);p.add_argument('--runner-arm-r',type=float,default=1.5);p.add_argument('--runner-floor-r',type=float,default=.75);a=p.parse_args()
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    _,ticks=load_all(a.catalog);tf=TF_SEC[a.tf];split=int(len(ticks)*.40);amap=build_atr_map(ticks,tf,14);out=[]
    for km in map(float,a.trigger_mults.split(',')):
      train=causal_events(ticks,tf,0,split,km,a.spread_window,a.pb_frac,a.reclaim_frac,a.wait_mult);test=causal_events(ticks,tf,split,len(ticks),km,a.spread_window,a.pb_frac,a.reclaim_frac,a.wait_mult)
      for dm in map(str.strip,a.directions.split(',')):
        dm=dm.upper();db=learn(ticks,train,split,tf,amap,a,km,dm,a.runner_mult); policy=[]; bases={ac:[] for ac in ACTIONS}; counts=defaultdict(int)
        for i,_,ts,trig,_ in test:
          side=ts if dm=='FOLLOW' else -ts; st,vals=event_counterfactuals(ticks,i,len(ticks),side,tf,trig,amap,a,km,dm,a.runner_mult)
          if st is None: continue
          for ac in ACTIONS:bases[ac].append(vals[ac])
          ch=choose(db,st,km,dm,a);counts[ch[0]]+=1;policy.append(vals[ch[0]])
        s=summarize(policy);s.update({'tf':a.tf,'trigger_mult':km,'direction_mode':dm,'mode':'POLICY','action_counts':dict(counts),'stage3_decisions':len(policy),'train_stage3':sum(len(v.get('EXIT',[])) for (lvl,k),v in db.items() if lvl==0)});out.append(s)
        for ac,pn in bases.items(): q=summarize(pn);q.update({'tf':a.tf,'trigger_mult':km,'direction_mode':dm,'mode':ac});out.append(q)
    out.sort(key=lambda r:(r['expectancy']>0,r['PF'],r['expectancy'],-r['maxDD_pct']),reverse=True)
    res={'experiment_id':a.experiment_id,'split':'40% chronological train / 60% OOS','objective':'Choose Stage3 action by conservative incremental EV relative to EXIT NOW','actions':list(ACTIONS),'hedge_note':'HEDGE removed: v15 implemented HEDGE identically to REVERSE; true dual-leg hedge requires separate semantics.','policy':'LCB(mean incremental PnL vs EXIT)>0, coarse causal state with hierarchical backoff','anti_lookahead':'Training counterfactuals end at split; OOS action uses only learned aggregates and path-so-far state.','top20':out[:20],'all':out,'verification_level':'CAUSAL_RAW_BIDASK_INCREMENTAL_EV_STAGE3_V16'}
    d=Path('results/g75-incremental-ev-stage3-v16')/a.experiment_id;d.mkdir(parents=True,exist_ok=True);(d/f'{a.tf}.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
