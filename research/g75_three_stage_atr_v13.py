from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import f, load_all
from g75_expected_entry_adaptive_v10 import causal_events, mid, spread
from g75_tsugi_causal_rawtick_v2 import TF_SEC


def build_atr_map(ticks, tf_sec:int, period:int=14):
    bars=[]; cur=None; o=h=l=c=None
    for t in ticks:
        m=mid(t); b=(int(t.ts_event)//1_000_000_000)//tf_sec
        if cur is None or b!=cur:
            if cur is not None: bars.append((cur,o,h,l,c))
            cur=b; o=h=l=c=m
        else:
            h=max(h,m); l=min(l,m); c=m
    if cur is not None: bars.append((cur,o,h,l,c))
    atr_by_bucket={}; trs=[]; prev_close=None
    for b,o,h,l,c in bars:
        tr=(h-l) if prev_close is None else max(h-l,abs(h-prev_close),abs(l-prev_close))
        # Store ATR for this bucket using only completed prior buckets.
        if trs:
            atr_by_bucket[b]=float(np.mean(trs[-period:]))
        else:
            atr_by_bucket[b]=tr
        trs.append(tr); prev_close=c
    return atr_by_bucket


def atr_at(t, tf_sec, amap, fallback):
    b=(int(t.ts_event)//1_000_000_000)//tf_sec
    return max(float(amap.get(b,fallback)),1e-9)


def exec_entry(t,side): return f(t.ask_price) if side>0 else f(t.bid_price)
def exec_exit(t,side): return f(t.bid_price) if side>0 else f(t.ask_price)


def three_stage_trade(ticks,i0,end_i,side,tf_sec,trig,amap,a,trail_mult,mode='ATR'):
    t0=ticks[i0]; m0=mid(t0); deadline=int(t0.ts_event)+int(tf_sec*a.horizon_mult*1e9)
    # Stage-1 probe; stages 2/3 are conditional recovery confirmations, not blind averaging.
    weights=[a.w1,a.w2,a.w3]
    entries=[(exec_entry(t0,side),weights[0])]
    stage=1; adverse_peak=0.0; reclaim_anchor=None; best_mid=m0
    basket_floor=None; max_fav=0.0; exit_reason='TIMEOUT'; last_j=i0
    base_atr=atr_at(t0,tf_sec,amap,max(trig,spread(t0)))
    fixed_tp=max(a.tp_mult*trig,a.cost_mult*spread(t0))
    hard_sl=max(a.hard_sl_mult*trig, fixed_tp/2.0)
    trail_armed=False
    for j in range(i0+1,end_i):
        t=ticks[j]; m=mid(t); last_j=j
        move=(m-m0)*side; max_fav=max(max_fav,move); adverse_peak=max(adverse_peak,-move)
        # Conditional stage-2 / stage-3: adverse excursion first, then reclaim toward original direction.
        next_stage=stage+1
        if next_stage<=3:
            threshold=(a.add2_mult if next_stage==2 else a.add3_mult)*trig
            if reclaim_anchor is None and adverse_peak>=threshold:
                reclaim_anchor=m
            if reclaim_anchor is not None and (m-reclaim_anchor)*side>=a.reclaim_add_mult*trig:
                entries.append((exec_entry(t,side),weights[next_stage-1])); stage=next_stage
                reclaim_anchor=None
        total_w=sum(w for _,w in entries)
        avg_entry=sum(px*w for px,w in entries)/total_w
        px=exec_exit(t,side)
        basket_pnl=(px-avg_entry)*side*total_w
        # Hard risk cap applies to executable basket PnL converted to price*weight units.
        if basket_pnl<=-hard_sl*total_w:
            exit_reason='HARD_SL'; break
        if mode=='FIXED':
            if basket_pnl>=fixed_tp*total_w:
                exit_reason='FIXED_TP'; break
        else:
            cur_atr=atr_at(t,tf_sec,amap,base_atr)
            arm=max(a.arm_atr*cur_atr,a.arm_trig*trig)*total_w
            if basket_pnl>=arm:
                trail_armed=True
            if trail_armed:
                # Trail the executable basket profit peak; trail distance expands/contracts with causal ATR.
                if basket_floor is None: basket_floor=basket_pnl
                basket_floor=max(basket_floor,basket_pnl)
                trail_dist=trail_mult*cur_atr*total_w
                protect=max(a.min_lock*trig*total_w,basket_floor-trail_dist)
                if basket_pnl<=protect:
                    exit_reason='ATR_TRAIL'; break
        if int(t.ts_event)>=deadline: break
    tx=ticks[last_j]; px=exec_exit(tx,side); total_w=sum(w for _,w in entries)
    pnl=sum((px-e)*side*w for e,w in entries)
    return {'pnl':pnl,'stages':stage,'reason':exit_reason,'weight':total_w,'max_fav':max_fav,'atr0':base_atr}


def summarize(rows):
    pn=[r['pnl'] for r in rows]; gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0))
    pf=gp/gl if gl else (math.inf if gp else 0.0)
    eq=1000.; peak=eq; mdd=0.
    for p in pn:
        eq+=p; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100 if peak>0 else 0.)
    n=len(pn)
    return {'N':n,'WR_pct':100*sum(x>0 for x in pn)/max(1,n),'PF':pf,'expectancy':float(np.mean(pn)) if pn else 0.,'net':float(sum(pn)),'maxDD_pct':mdd,
            'stage1_pct':100*sum(r['stages']==1 for r in rows)/max(1,n),'stage2_pct':100*sum(r['stages']==2 for r in rows)/max(1,n),'stage3_pct':100*sum(r['stages']==3 for r in rows)/max(1,n),
            'positive_3stage_pct':100*sum((r['stages']==3 and r['pnl']>0) for r in rows)/max(1,sum(r['stages']==3 for r in rows))}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--experiment-id',required=True); p.add_argument('--tf',choices=TF_SEC,default='M1'); p.add_argument('--raw-bidask-only',action='store_true')
    p.add_argument('--trigger-mults',default='1,1.5,2'); p.add_argument('--directions',default='FOLLOW,FADE'); p.add_argument('--trail-mults',default='1,1.5,2,2.5,3');
    p.add_argument('--spread-window',type=int,default=256); p.add_argument('--pb-frac',type=float,default=.25); p.add_argument('--reclaim-frac',type=float,default=.10); p.add_argument('--wait-mult',type=float,default=2.)
    p.add_argument('--w1',type=float,default=.30); p.add_argument('--w2',type=float,default=.30); p.add_argument('--w3',type=float,default=.40); p.add_argument('--add2-mult',type=float,default=.50); p.add_argument('--add3-mult',type=float,default=1.00); p.add_argument('--reclaim-add-mult',type=float,default=.15)
    p.add_argument('--tp-mult',type=float,default=2.0); p.add_argument('--cost-mult',type=float,default=2.5); p.add_argument('--hard-sl-mult',type=float,default=2.0); p.add_argument('--horizon-mult',type=float,default=12.)
    p.add_argument('--arm-atr',type=float,default=.75); p.add_argument('--arm-trig',type=float,default=.50); p.add_argument('--min-lock',type=float,default=.10); a=p.parse_args()
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    _,ticks=load_all(a.catalog); tf=TF_SEC[a.tf]; split=int(len(ticks)*.40); amap=build_atr_map(ticks,tf,14)
    # OOS-only execution; no outcome-based region selection and no post-entry feature is used for entry selection.
    rows_out=[]
    for km in map(float,a.trigger_mults.split(',')):
        evs=causal_events(ticks,tf,split,len(ticks),km,a.spread_window,a.pb_frac,a.reclaim_frac,a.wait_mult)
        for dmode in a.directions.split(','):
            dmode=dmode.strip().upper()
            for exit_mode,tm in [('FIXED',0.0)]+[('ATR',x) for x in map(float,a.trail_mults.split(','))]:
                rows=[]
                for i,x,trig_side,trig,_ in evs:
                    side=trig_side if dmode=='FOLLOW' else -trig_side
                    rows.append(three_stage_trade(ticks,i,len(ticks),side,tf,trig,amap,a,tm,exit_mode))
                s=summarize(rows); s.update({'tf':a.tf,'trigger_mult':km,'direction_mode':dmode,'exit_mode':exit_mode,'trail_mult':tm,'confirmed_candidates':len(evs)})
                rows_out.append(s)
    rows_out.sort(key=lambda r:(r['expectancy']>0,r['PF'],r['net'],-r['maxDD_pct']),reverse=True)
    out={'experiment_id':a.experiment_id,'split':'first 40% excluded; execution on final 60% chronological OOS','objective':'Test whether conditional 3-stage recovery plus dynamic ATR trailing improves basket expectancy versus fixed TP without future leakage','entry_rule':'G75 adaptive trigger -> pullback -> reclaim; direction FOLLOW or FADE only','sequence':'E1 probe; E2 after adverse 0.5*trigger then reclaim; E3 after adverse 1.0*trigger then reclaim; weights 0.30/0.30/0.40','atr_rule':'14 completed TF bars only; ATR trail arms after causal profit threshold; executable Bid/Ask PnL','anti_lookahead':'No post-entry MFE/MAE/path feature is used to decide entry, direction, stage, or trail parameter','top20':rows_out[:20],'all':rows_out,'verification_level':'CAUSAL_RAW_BIDASK_G75_3STAGE_DYNAMIC_ATR_V13'}
    d=Path('results/g75-three-stage-atr-v13')/a.experiment_id; d.mkdir(parents=True,exist_ok=True); (d/f'{a.tf}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
