from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import f, load_all
from g75_expected_entry_adaptive_v10 import causal_events, mid, spread
from g75_tsugi_causal_rawtick_v2 import TF_SEC
from g75_three_stage_atr_v13 import build_atr_map, atr_at, exec_entry, exec_exit


def q(v,p): return float(np.quantile(v,p)) if v else 0.0


def trade(ticks,i0,end_i,side,tf_sec,trig,amap,a,stage3_mode,runner_mult):
    t0=ticks[i0]; ts0=int(t0.ts_event); m0=mid(t0)
    deadline=ts0+int(tf_sec*a.horizon_mult*1e9)
    base_atr=atr_at(t0,tf_sec,amap,max(trig,spread(t0)))
    hard_sl=max(a.hard_sl_mult*trig,a.cost_mult*spread(t0)/2.0)
    entries=[(exec_entry(t0,side),a.w1,side)]
    stage=1; adverse_peak=0.0; reclaim_anchor=None; last_j=i0
    mfe=0.0; mae=0.0; peak_basket=-1e99; reason='TIMEOUT'
    # Profit state machine flags
    be_armed=False; lock1_armed=False; runner_armed=False
    s3_action='NONE'

    def basket_pnl(t):
        px_long=f(t.bid_price); px_short=f(t.ask_price)
        p=0.0; w=0.0
        for e,wt,sd in entries:
            px=px_long if sd>0 else px_short
            p+=(px-e)*sd*wt; w+=wt
        return p,w

    for j in range(i0+1,end_i):
        t=ticks[j]; ts=int(t.ts_event); m=mid(t); last_j=j
        move=(m-m0)*side; mfe=max(mfe,move); mae=min(mae,move)
        adverse_peak=max(adverse_peak,-move)

        # Stage 2 remains same-direction after adverse excursion + reclaim.
        if stage==1:
            th=a.add2_mult*trig
            if reclaim_anchor is None and adverse_peak>=th: reclaim_anchor=m
            if reclaim_anchor is not None and (m-reclaim_anchor)*side>=a.reclaim_add_mult*trig:
                entries.append((exec_entry(t,side),a.w2,side)); stage=2; reclaim_anchor=None

        # Stage 3 action is causal and configurable; no post-entry labels are used.
        elif stage==2:
            th=a.add3_mult*trig
            if reclaim_anchor is None and adverse_peak>=th: reclaim_anchor=m
            if reclaim_anchor is not None and (m-reclaim_anchor)*side>=a.reclaim_add_mult*trig:
                if stage3_mode=='CONTINUE':
                    entries.append((exec_entry(t,side),a.w3,side)); s3_action='CONTINUE'
                elif stage3_mode=='REVERSE':
                    entries.append((exec_entry(t,-side),a.w3,-side)); s3_action='REVERSE'
                elif stage3_mode=='HEDGE':
                    entries.append((exec_entry(t,-side),a.w3,-side)); s3_action='HEDGE'
                elif stage3_mode=='ABORT':
                    s3_action='ABORT'
                stage=3; reclaim_anchor=None

        pnl,total_w=basket_pnl(t); peak_basket=max(peak_basket,pnl)
        risk_unit=max(hard_sl*max(total_w,1e-9),1e-9)
        r=pnl/risk_unit

        # Excursion-aware profit state machine.
        if r>=a.be_arm_r: be_armed=True
        if r>=a.lock1_r: lock1_armed=True
        if r>=a.runner_arm_r: runner_armed=True

        # Hard loss cap always dominates.
        if pnl<=-risk_unit:
            reason='HARD_SL'; break
        # Once the trade had useful MFE, do not allow it to round-trip all the way back.
        if be_armed and r<=a.be_floor_r:
            reason='BE_PROTECT'; break
        if lock1_armed and r<=a.lock1_floor_r:
            reason='LOCK1'; break

        if runner_armed:
            cur_atr=atr_at(t,tf_sec,amap,base_atr)
            trail_dist=runner_mult*cur_atr*max(total_w,1e-9)
            floor=max(a.runner_floor_r*risk_unit,peak_basket-trail_dist)
            if pnl<=floor:
                reason='ATR_RUNNER'; break

        if ts>=deadline: break

    tx=ticks[last_j]; pnl,total_w=basket_pnl(tx)
    return {'pnl':pnl,'stages':stage,'s3_action':s3_action,'reason':reason,'MFE':mfe,'MAE':mae,'weight':total_w}


def summarize(rows):
    pn=[r['pnl'] for r in rows]; n=len(rows); wins=[x for x in pn if x>0]
    gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0)); pf=gp/gl if gl else (math.inf if gp else 0.0)
    eq=1000.; peak=eq; mdd=0.
    for p in pn:
        eq+=p; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100 if peak>0 else 0.)
    s3=[r for r in rows if r['stages']==3]
    return {
      'N':n,'WR_pct':100*len(wins)/max(1,n),'PF':pf,'expectancy':float(np.mean(pn)) if pn else 0.0,'net':float(sum(pn)),'maxDD_pct':mdd,
      'MFE_mean':float(np.mean([r['MFE'] for r in rows])) if rows else 0.0,'MAE_mean':float(np.mean([r['MAE'] for r in rows])) if rows else 0.0,
      'stage3_pct':100*len(s3)/max(1,n),'positive_3stage_pct':100*sum(r['pnl']>0 for r in s3)/max(1,len(s3)),
      'be_protect_pct':100*sum(r['reason']=='BE_PROTECT' for r in rows)/max(1,n),'lock1_pct':100*sum(r['reason']=='LOCK1' for r in rows)/max(1,n),'runner_exit_pct':100*sum(r['reason']=='ATR_RUNNER' for r in rows)/max(1,n),
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--experiment-id',required=True); p.add_argument('--tf',choices=TF_SEC,default='M1'); p.add_argument('--raw-bidask-only',action='store_true')
    p.add_argument('--trigger-mults',default='1.5,2'); p.add_argument('--directions',default='FOLLOW,FADE'); p.add_argument('--stage3-modes',default='CONTINUE,ABORT,REVERSE,HEDGE'); p.add_argument('--runner-mults',default='1.5,2,2.5,3')
    p.add_argument('--spread-window',type=int,default=256); p.add_argument('--pb-frac',type=float,default=.25); p.add_argument('--reclaim-frac',type=float,default=.10); p.add_argument('--wait-mult',type=float,default=2.)
    p.add_argument('--w1',type=float,default=.30); p.add_argument('--w2',type=float,default=.30); p.add_argument('--w3',type=float,default=.40); p.add_argument('--add2-mult',type=float,default=.50); p.add_argument('--add3-mult',type=float,default=1.00); p.add_argument('--reclaim-add-mult',type=float,default=.15)
    p.add_argument('--cost-mult',type=float,default=2.5); p.add_argument('--hard-sl-mult',type=float,default=2.0); p.add_argument('--horizon-mult',type=float,default=12.)
    p.add_argument('--be-arm-r',type=float,default=.50); p.add_argument('--be-floor-r',type=float,default=.05); p.add_argument('--lock1-r',type=float,default=1.0); p.add_argument('--lock1-floor-r',type=float,default=.35); p.add_argument('--runner-arm-r',type=float,default=1.5); p.add_argument('--runner-floor-r',type=float,default=.75)
    a=p.parse_args();
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    _,ticks=load_all(a.catalog); tf=TF_SEC[a.tf]; split=int(len(ticks)*.40); amap=build_atr_map(ticks,tf,14)
    outrows=[]
    for km in map(float,a.trigger_mults.split(',')):
        evs=causal_events(ticks,tf,split,len(ticks),km,a.spread_window,a.pb_frac,a.reclaim_frac,a.wait_mult)
        for dmode in a.directions.split(','):
            dmode=dmode.strip().upper()
            for s3 in a.stage3_modes.split(','):
                s3=s3.strip().upper()
                for rm in map(float,a.runner_mults.split(',')):
                    rows=[]
                    for i,x,trig_side,trig,_ in evs:
                        side=trig_side if dmode=='FOLLOW' else -trig_side
                        rows.append(trade(ticks,i,len(ticks),side,tf,trig,amap,a,s3,rm))
                    s=summarize(rows); s.update({'tf':a.tf,'trigger_mult':km,'direction_mode':dmode,'stage3_mode':s3,'runner_mult':rm,'confirmed_candidates':len(evs)})
                    outrows.append(s)
    outrows.sort(key=lambda r:(r['expectancy']>0,r['PF'],r['expectancy'],-r['maxDD_pct']),reverse=True)
    out={'experiment_id':a.experiment_id,'split':'first 40% excluded; final 60% chronological OOS','objective':'Excursion-aware profit protection plus causal Stage3 action decomposition','profit_state_machine':'0.5R=>BE protect, 1R=>profit lock, 1.5R=>ATR runner','stage3_actions':['CONTINUE','ABORT','REVERSE','HEDGE'],'anti_lookahead':'Only contemporaneous price, completed-bar ATR, realized path-so-far and configured thresholds are used. Future MFE/MAE are not used for decisions.','top20':outrows[:20],'all':outrows,'verification_level':'CAUSAL_RAW_BIDASK_G75_EXCURSION_AWARE_STAGE3_V15'}
    d=Path('results/g75-excursion-aware-v15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True); (d/f'{a.tf}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
