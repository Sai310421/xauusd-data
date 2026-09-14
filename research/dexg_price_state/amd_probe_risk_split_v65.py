from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_liquidity_cycle_v6 as v6

START_EQUITY = 1000.0
BASE_LOT = 0.01
CONTRACT_OZ = 100.0
QTY_OZ = BASE_LOT * CONTRACT_OZ
RISK_MULTS = [1.0, 1.25, 1.5, 2.0]
TPS = [0.30, 0.50, 0.75, 1.00]
SLS = [0.10, 0.15, 0.20, 0.30]
HOLDS = [3, 5, 10, 15]
SCENARIOS = [
    {'name':'RAW_NO_CB','commission_side':3.5,'spread':0.10,'slip_side':0.02,'rebate_rt':0.0},
    {'name':'ZERO_NO_CB','commission_side':6.0,'spread':0.02,'slip_side':0.02,'rebate_rt':0.0},
    {'name':'ZERO_TARITALI_1_35','commission_side':6.0,'spread':0.02,'slip_side':0.02,'rebate_rt':1.35},
    {'name':'ZERO_PAYBACKFX_2_4375_SENS','commission_side':6.0,'spread':0.02,'slip_side':0.02,'rebate_rt':2.4375},
]

def cost_usd(sc, weight):
    lot = BASE_LOT * weight
    spread = sc['spread'] * QTY_OZ * weight
    commission = 2.0 * sc['commission_side'] * lot
    slippage = 2.0 * sc['slip_side'] * QTY_OZ * weight
    rebate = sc['rebate_rt'] * lot
    return spread + commission + slippage - rebate

def arm_events(ticks, b15):
    ev=[]
    for i in range(15,len(b15)-1):
        if not v6.accumulation_ok(b15,i): continue
        A,S,H,D=b15[i-3],b15[i-2],b15[i-1],b15[i]
        anchor=v6.initial_sweep_side(A,S)
        if not anchor: continue
        hm=v6.opposite_harvest_meta(A,S,H,anchor)
        if not hm['broad']: continue
        arm_idx,_=v6.find_arm(ticks,D,anchor,hm['return_level'])
        if arm_idx is None or arm_idx>=len(ticks)-2: continue
        ev.append((arm_idx,-anchor))
    return ev

def future_indices(ticks, arm_idx, side, hold_s):
    end=int(ticks[arm_idx].ts_event)+int(hold_s*1e9)
    out=[]
    for j in range(arm_idx+1,len(ticks)):
        if int(ticks[j].ts_event)>end: break
        out.append(j)
    return out

def tranche_plan(ticks, arm_idx, side, variant, hold_s):
    entry=v6.mid(ticks[arm_idx]); idxs=future_indices(ticks,arm_idx,side,hold_s)
    if variant=='SINGLE': return [(arm_idx,1.0)]
    plan=[(arm_idx,0.50)]
    if not idxs: return plan
    if variant=='SPLIT_A':
        l2=None; peak=-1e9
        for j in idxs:
            m=(v6.mid(ticks[j])-entry)*side; peak=max(peak,m)
            if l2 is None and m>=0.10: l2=j; plan.append((j,0.30)); continue
            if l2 is not None and peak>=0.10 and m<=peak-0.05 and m>-0.20:
                plan.append((j,0.20)); break
    elif variant=='SPLIT_B':
        got2=False
        for j in idxs:
            m=(v6.mid(ticks[j])-entry)*side
            if not got2 and m>=0.10:
                plan.append((j,0.30)); got2=True
            if got2 and m>=0.20:
                plan.append((j,0.20)); break
    elif variant=='SPLIT_C':
        got2=False; prev=[]; peak=-1e9
        for j in idxs:
            m=(v6.mid(ticks[j])-entry)*side; peak=max(peak,m); prev.append(m)
            if len(prev)>4: prev.pop(0)
            accel=len(prev)>=4 and prev[-1]>prev[-2]>prev[-3]>prev[-4] and (prev[-1]-prev[-4])>=0.05
            if not got2 and accel:
                plan.append((j,0.30)); got2=True; continue
            # PA proxy: failed adverse break / wick rejection = pullback then reclaim above recent path midpoint.
            if got2 and peak>=0.08 and m<=peak-0.04 and m>-0.18:
                # find causal reclaim after this pullback
                pull=m
                for k in idxs:
                    if k<=j: continue
                    mk=(v6.mid(ticks[k])-entry)*side
                    if mk>=pull+0.04:
                        plan.append((k,0.20)); return plan
                break
    return plan

def simulate_signal(ticks, arm_idx, side, variant, tp, sl, hold_s, sc):
    plan=tranche_plan(ticks,arm_idx,side,variant,hold_s)
    last=max(x[0] for x in plan); end=int(ticks[arm_idx].ts_event)+int(hold_s*1e9)
    active=[]; pidx=0; pnl=0.0; filled=0; exit_idx=last
    j=arm_idx
    while j < len(ticks) and int(ticks[j].ts_event)<=end:
        while pidx<len(plan) and plan[pidx][0]<=j:
            ei,w=plan[pidx]; active.append((v6.mid(ticks[ei]),w)); pnl-=cost_usd(sc,w); filled+=1; pidx+=1
        if active:
            weighted=sum((v6.mid(ticks[j])-e)*side*w for e,w in active)
            active_w=sum(w for _,w in active)
            if active_w>0:
                avg_move=weighted/active_w
                if avg_move>=tp or avg_move<=-sl:
                    pnl += weighted*QTY_OZ; exit_idx=j; return pnl,filled,active_w,avg_move,exit_idx
        j+=1
    if active:
        j=max(arm_idx,min(j-1,len(ticks)-1)); weighted=sum((v6.mid(ticks[j])-e)*side*w for e,w in active)
        pnl += weighted*QTY_OZ; exit_idx=j
        return pnl,filled,sum(w for _,w in active),weighted/max(sum(w for _,w in active),1e-12),exit_idx
    return 0.0,0,0.0,0.0,exit_idx

def metrics(pnls, filleds, risk_mult, n_days):
    scaled=[x*risk_mult for x in pnls]; eq=START_EQUITY; peak=eq; maxdd=0.0; maxdd_usd=0.0
    gross_win=gross_loss=0.0; wins=0
    for x in scaled:
        eq+=x; peak=max(peak,eq); dd=peak-eq
        if dd>maxdd_usd: maxdd_usd=dd
        maxdd=max(maxdd,dd/peak if peak else 0.0)
        if x>0: wins+=1; gross_win+=x
        elif x<0: gross_loss+=-x
    net=sum(scaled); pf=gross_win/gross_loss if gross_loss>0 else None
    ret=net/START_EQUITY; monthly=ret*(21.0/max(n_days,1))
    return {'signals':len(pnls),'filled_tranches':sum(filleds),'avg_tranches_per_signal':sum(filleds)/len(filleds) if filleds else 0.0,
            'wr':wins/len(scaled) if scaled else 0.0,'pf':pf,'expectancy_usd':net/len(scaled) if scaled else 0.0,
            'net_pnl_usd':net,'return_pct':ret*100.0,'max_dd_pct':maxdd*100.0,'max_dd_usd':maxdd_usd,
            'rf':net/maxdd_usd if maxdd_usd>0 else None,'monthly21_pct':monthly*100.0,'ending_equity':eq,'risk_mult':risk_mult}

def run(catalog,max_ticks=3000000):
    ticks=v6.load_ticks(catalog,max_ticks); b15=v6.bars_n(ticks,15); events=arm_events(ticks,b15)
    dates={datetime.fromtimestamp(int(t.ts_event)/1e9,tz=timezone.utc).date() for t in ticks}
    n_days=max(1,sum(1 for d in dates if d.weekday()<5))
    variants=['SINGLE','SPLIT_A','SPLIT_B','SPLIT_C']; rows=[]
    for sc in SCENARIOS:
        for var in variants:
            for tp in TPS:
                for sl in SLS:
                    for hold in HOLDS:
                        pnls=[]; fills=[]
                        for arm_idx,side in events:
                            p,f,_,_,_=simulate_signal(ticks,arm_idx,side,var,tp,sl,hold,sc); pnls.append(p); fills.append(f)
                        base=metrics(pnls,fills,1.0,n_days)
                        # Only scale if unscaled edge is genuinely positive and PF > 1.
                        for rm in RISK_MULTS:
                            m=metrics(pnls,fills,rm,n_days)
                            allowed=(base['expectancy_usd']>0 and (base['pf'] or 0)>1.0)
                            m.update({'scenario':sc['name'],'variant':var,'tp':tp,'sl':sl,'hold_s':hold,'scale_allowed':allowed})
                            if rm>1.0 and not allowed: m['selection_eligible']=False
                            else: m['selection_eligible']=True
                            rows.append(m)
    eligible=[r for r in rows if r['selection_eligible'] and r['expectancy_usd']>0 and (r['pf'] or 0)>1.0]
    eligible.sort(key=lambda r:(r['return_pct']/max(r['max_dd_pct'],1e-9)),reverse=True)
    return {'status':'CAUSAL_V65_PROBE_3SPLIT_EXNESS_COST_PNL_SENSITIVITY','ticks':len(ticks),'bars15':len(b15),'signals':len(events),
            'business_days_observed':n_days,'start_equity':START_EQUITY,'base_lot':BASE_LOT,
            'note':'Raw Dukascopy drives causal state/price path. Exness account costs/rebates are modeled sensitivity, not empirical Exness tick replay. SPLIT_C PA confirmation uses causal tick-path acceleration + pullback/reclaim proxy.',
            'top20':eligible[:20],'all_results':rows}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--max-ticks',type=int,default=3000000); a=ap.parse_args()
    res=run(a.catalog,a.max_ticks); d=Path('results/dexg-amd15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/'AMD_PROBE_RISK_SPLIT_V65.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps({'status':res['status'],'signals':res['signals'],'business_days_observed':res['business_days_observed'],'top20':res['top20']},indent=2))
if __name__=='__main__': main()
