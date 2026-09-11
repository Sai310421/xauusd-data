from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import f, load_all
from g75_expected_entry_adaptive_v10 import causal_events, mid, spread
from g75_tsugi_causal_rawtick_v2 import TF_SEC
from g75_three_stage_atr_v13 import build_atr_map, atr_at, exec_entry, exec_exit


def q(v,p):
    return float(np.quantile(v,p)) if v else 0.0


def excursion_trade(ticks,i0,end_i,side,tf_sec,trig,amap,a,trail_mult,mode='ATR'):
    t0=ticks[i0]; ts0=int(t0.ts_event); m0=mid(t0)
    deadline=ts0+int(tf_sec*a.horizon_mult*1e9)
    weights=[a.w1,a.w2,a.w3]; entries=[(exec_entry(t0,side),weights[0])]
    stage=1; adverse_peak=0.0; reclaim_anchor=None
    base_atr=atr_at(t0,tf_sec,amap,max(trig,spread(t0)))
    fixed_tp=max(a.tp_mult*trig,a.cost_mult*spread(t0)); hard_sl=max(a.hard_sl_mult*trig,fixed_tp/2.0)
    trail_armed=False; basket_peak=-1e99; exit_reason='TIMEOUT'; last_j=i0

    mfe=-1e99; mae=1e99; mfe_ts=ts0; mae_ts=ts0
    pre_tp_mae=0.0; pre_sl_mfe=0.0; tp_seen=False; sl_seen=False
    # Diagnostics only: post-entry continuation across fixed horizons; never used by decisions.
    horizon_mfe={'M1':-1e99,'M5':-1e99,'M15':-1e99}
    horizon_ns={'M1':60,'M5':300,'M15':900}

    for j in range(i0+1,end_i):
        t=ticks[j]; ts=int(t.ts_event); m=mid(t); last_j=j
        move=(m-m0)*side
        if move>mfe: mfe=move; mfe_ts=ts
        if move<mae: mae=move; mae_ts=ts
        for k,sec in horizon_ns.items():
            if ts-ts0<=sec*1_000_000_000: horizon_mfe[k]=max(horizon_mfe[k],move)

        adverse_peak=max(adverse_peak,-move)
        next_stage=stage+1
        if next_stage<=3:
            threshold=(a.add2_mult if next_stage==2 else a.add3_mult)*trig
            if reclaim_anchor is None and adverse_peak>=threshold: reclaim_anchor=m
            if reclaim_anchor is not None and (m-reclaim_anchor)*side>=a.reclaim_add_mult*trig:
                entries.append((exec_entry(t,side),weights[next_stage-1])); stage=next_stage; reclaim_anchor=None

        total_w=sum(w for _,w in entries); avg_entry=sum(px*w for px,w in entries)/total_w
        px=exec_exit(t,side); basket_pnl=(px-avg_entry)*side*total_w
        basket_peak=max(basket_peak,basket_pnl)

        # Counterfactual path diagnostics for the actual active basket. These are labels only.
        if not tp_seen:
            pre_tp_mae=min(pre_tp_mae,basket_pnl/total_w)
            if basket_pnl>=fixed_tp*total_w: tp_seen=True
        if not sl_seen:
            pre_sl_mfe=max(pre_sl_mfe,basket_pnl/total_w)
            if basket_pnl<=-hard_sl*total_w: sl_seen=True

        if basket_pnl<=-hard_sl*total_w:
            exit_reason='HARD_SL'; break
        if mode=='FIXED':
            if basket_pnl>=fixed_tp*total_w: exit_reason='FIXED_TP'; break
        else:
            cur_atr=atr_at(t,tf_sec,amap,base_atr)
            arm=max(a.arm_atr*cur_atr,a.arm_trig*trig)*total_w
            if basket_pnl>=arm: trail_armed=True
            if trail_armed:
                trail_dist=trail_mult*cur_atr*total_w
                protect=max(a.min_lock*trig*total_w,basket_peak-trail_dist)
                if basket_pnl<=protect: exit_reason='ATR_TRAIL'; break
        if ts>=deadline: break

    tx=ticks[last_j]; px=exec_exit(tx,side); total_w=sum(w for _,w in entries)
    pnl=sum((px-e)*side*w for e,w in entries)
    mfe=max(0.0,mfe if mfe>-1e90 else 0.0); mae=min(0.0,mae if mae<1e90 else 0.0)
    # Opportunity/capture metrics are diagnostic labels, not entry features.
    available_mfe=max(0.0,mfe)*total_w
    capture=(pnl/available_mfe) if available_mfe>1e-12 and pnl>0 else 0.0
    cont_m1=max(0.0,horizon_mfe['M1']); cont_m5=max(0.0,horizon_mfe['M5']); cont_m15=max(0.0,horizon_mfe['M15'])
    return {
        'pnl':pnl,'stages':stage,'reason':exit_reason,'weight':total_w,'atr0':base_atr,
        'MFE':mfe,'MAE':mae,'MFE_R':mfe/max(hard_sl,1e-9),'MAE_R':mae/max(hard_sl,1e-9),
        'tp_reached':tp_seen,'sl_reached':sl_seen,'TP_before_MAE':pre_tp_mae,'SL_before_MFE':pre_sl_mfe,
        'time_to_MFE_sec':max(0.0,(mfe_ts-ts0)/1e9),'time_to_MAE_sec':max(0.0,(mae_ts-ts0)/1e9),
        'capture_ratio':capture,'MFE_M1':cont_m1,'MFE_M5':cont_m5,'MFE_M15':cont_m15,
        'cont_M1_M5':bool(cont_m5>cont_m1+1e-12),'cont_M5_M15':bool(cont_m15>cont_m5+1e-12),
        'fixed_tp':fixed_tp,'hard_sl':hard_sl,
    }


def summarize(rows):
    pn=[r['pnl'] for r in rows]; gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0)); n=len(rows)
    pf=gp/gl if gl else (math.inf if gp else 0.0)
    eq=1000.; peak=eq; mdd=0.
    for p in pn:
        eq+=p; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100 if peak>0 else 0.)
    mfe=[r['MFE'] for r in rows]; mae=[r['MAE'] for r in rows]; ttm=[r['time_to_MFE_sec'] for r in rows]
    losses=[r for r in rows if r['pnl']<=0]; wins=[r for r in rows if r['pnl']>0]; s3=[r for r in rows if r['stages']==3]
    caps=[r['capture_ratio'] for r in wins if r['capture_ratio']>=0]
    return {
        'N':n,'WR_pct':100*len(wins)/max(1,n),'PF':pf,'expectancy':float(np.mean(pn)) if pn else 0.0,'net':float(sum(pn)),'maxDD_pct':mdd,
        'MFE_mean':float(np.mean(mfe)) if mfe else 0.0,'MFE_p50':q(mfe,.5),'MFE_p75':q(mfe,.75),'MFE_p90':q(mfe,.9),
        'MAE_mean':float(np.mean(mae)) if mae else 0.0,'MAE_p10':q(mae,.1),'MAE_p25':q(mae,.25),'MAE_p50':q(mae,.5),
        'time_to_MFE_p50_sec':q(ttm,.5),'time_to_MFE_p75_sec':q(ttm,.75),'time_to_MFE_p90_sec':q(ttm,.9),
        'TP_before_MAE_mean':float(np.mean([r['TP_before_MAE'] for r in rows if r['tp_reached']])) if any(r['tp_reached'] for r in rows) else 0.0,
        'SL_before_MFE_mean':float(np.mean([r['SL_before_MFE'] for r in losses if r['sl_reached']])) if any(r['sl_reached'] for r in losses) else 0.0,
        'losses_with_MFE_ge_0_5R_pct':100*sum(r['MFE_R']>=.5 for r in losses)/max(1,len(losses)),
        'losses_with_MFE_ge_1R_pct':100*sum(r['MFE_R']>=1.0 for r in losses)/max(1,len(losses)),
        'capture_ratio_win_mean':float(np.mean(caps)) if caps else 0.0,'capture_ratio_win_p50':q(caps,.5),
        'M1_to_M5_cont_pct':100*sum(r['cont_M1_M5'] for r in rows)/max(1,n),'M5_to_M15_cont_pct':100*sum(r['cont_M5_M15'] for r in rows)/max(1,n),
        'stage3_pct':100*len(s3)/max(1,n),'positive_3stage_pct':100*sum(r['pnl']>0 for r in s3)/max(1,len(s3)),
        'stage3_MFE_mean':float(np.mean([r['MFE'] for r in s3])) if s3 else 0.0,'stage3_MAE_mean':float(np.mean([r['MAE'] for r in s3])) if s3 else 0.0,
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--experiment-id',required=True); p.add_argument('--tf',choices=TF_SEC,default='M1'); p.add_argument('--raw-bidask-only',action='store_true')
    p.add_argument('--trigger-mults',default='1,1.5,2'); p.add_argument('--directions',default='FOLLOW,FADE'); p.add_argument('--trail-mults',default='2,3')
    p.add_argument('--spread-window',type=int,default=256); p.add_argument('--pb-frac',type=float,default=.25); p.add_argument('--reclaim-frac',type=float,default=.10); p.add_argument('--wait-mult',type=float,default=2.)
    p.add_argument('--w1',type=float,default=.30); p.add_argument('--w2',type=float,default=.30); p.add_argument('--w3',type=float,default=.40); p.add_argument('--add2-mult',type=float,default=.50); p.add_argument('--add3-mult',type=float,default=1.00); p.add_argument('--reclaim-add-mult',type=float,default=.15)
    p.add_argument('--tp-mult',type=float,default=2.0); p.add_argument('--cost-mult',type=float,default=2.5); p.add_argument('--hard-sl-mult',type=float,default=2.0); p.add_argument('--horizon-mult',type=float,default=12.)
    p.add_argument('--arm-atr',type=float,default=.75); p.add_argument('--arm-trig',type=float,default=.50); p.add_argument('--min-lock',type=float,default=.10); a=p.parse_args()
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    _,ticks=load_all(a.catalog); tf=TF_SEC[a.tf]; split=int(len(ticks)*.40); amap=build_atr_map(ticks,tf,14)
    outrows=[]
    for km in map(float,a.trigger_mults.split(',')):
        evs=causal_events(ticks,tf,split,len(ticks),km,a.spread_window,a.pb_frac,a.reclaim_frac,a.wait_mult)
        for dmode in a.directions.split(','):
            dmode=dmode.strip().upper()
            for exit_mode,tm in [('FIXED',0.0)]+[('ATR',x) for x in map(float,a.trail_mults.split(','))]:
                rows=[]
                for i,x,trig_side,trig,_ in evs:
                    side=trig_side if dmode=='FOLLOW' else -trig_side
                    rows.append(excursion_trade(ticks,i,len(ticks),side,tf,trig,amap,a,tm,exit_mode))
                s=summarize(rows); s.update({'tf':a.tf,'trigger_mult':km,'direction_mode':dmode,'exit_mode':exit_mode,'trail_mult':tm,'confirmed_candidates':len(evs)})
                outrows.append(s)
    outrows.sort(key=lambda r:(r['expectancy']>0,r['PF'],r['expectancy'],-r['maxDD_pct']),reverse=True)
    out={
      'experiment_id':a.experiment_id,'split':'first 40% excluded; final 60% chronological OOS',
      'objective':'Diagnose Direction x Timing x Excursion x Exit using causal entries and post-entry path labels only',
      'diagnostics':['MFE','MAE','TP-before-MAE','SL-before-MFE','time-to-MFE','M1->M5->M15 continuation','capture ratio','stage3 excursion'],
      'anti_lookahead':'All excursion/path values are labels after entry and are never used to choose entry, direction, stage, or trail in this run.',
      'top20':outrows[:20],'all':outrows,'verification_level':'CAUSAL_RAW_BIDASK_G75_EXCURSION_DIAGNOSTIC_V14'}
    d=Path('results/g75-excursion-v14')/a.experiment_id; d.mkdir(parents=True,exist_ok=True); (d/f'{a.tf}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
