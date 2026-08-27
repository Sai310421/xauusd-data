from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import nautilus_trader

from research.minimumspike_reality_noise_bt_v1 import (
    DATA, PARAMS, PROFILES, atr14, build_signals, draw_cost
)

OUT = Path('results/minimumspike-tp075-sweep-v1')
OUT.mkdir(parents=True, exist_ok=True)

# Test profit-taking around the proposed 0.75 level while preserving the recovered core.
TP_MULTS = [0.60, 0.75, 0.90, 1.00]
TRAIL_ACTS = [0.30, 0.40, 0.50]
TRAIL_DIST = 0.20
RISK_MULT = 2.80
TARGET_DD_PCT = 3.0


def run_once(df, signals, p, seed: int, tp_mult: float, trail_act: float):
    rng=np.random.default_rng(seed)
    o,h,l,c=[df[x].to_numpy() for x in ['open','high','low','close']]
    equity=10000.0; peak=equity; maxdd=0.0
    trades=[]; next_free=0
    for entry_i, spike_i, atr in signals:
        if entry_i < next_free: continue
        if rng.random() < p.reject_prob: continue
        ec,_,_,_,_=draw_cost(rng,p,atr)
        entry=o[entry_i]+ec
        stop=l[spike_i]-PARAMS['guard_ext']*atr
        tp=entry+tp_mult*atr
        trail=None; exit_px=None; exit_i=None
        end=min(len(df)-1, entry_i+PARAMS['max_hold'])
        for k in range(entry_i,end+1):
            held=k-entry_i
            active_stop=max(stop, trail) if trail is not None else stop
            if l[k] <= active_stop and held >= PARAMS['min_hold']:
                exit_px=active_stop; exit_i=k; break
            if h[k] >= tp and held >= PARAMS['min_hold']:
                exit_px=tp; exit_i=k; break
            if h[k] >= entry+trail_act*atr:
                newtrail=h[k]-TRAIL_DIST*atr
                trail=newtrail if trail is None else max(trail,newtrail)
        if exit_px is None:
            exit_i=end; exit_px=c[end]
        xc,_,_,_,_=draw_cost(rng,p,atr)
        exit_fill=exit_px-xc
        gross=(exit_fill-entry)*RISK_MULT
        commission=p.commission_rt*RISK_MULT
        net=gross-commission
        equity+=net; peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/peak*100)
        trades.append(net)
        next_free=exit_i+1
    a=np.asarray(trades,float)
    wins=a[a>0]; losses=a[a<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    wr=float((a>0).mean()*100) if len(a) else 0.0
    ret=(equity/10000-1)*100
    monthly=(equity/10000)**(21/90)-1 if equity>0 else -1
    avgwin=float(wins.mean()) if len(wins) else 0.0
    avgloss=float(abs(losses.mean())) if len(losses) else 0.0
    return dict(tp_mult=tp_mult,trail_act=trail_act,N=len(a),WR_pct=wr,PF=pf,MaxDD_pct=maxdd,
                Return90d_pct=ret,Monthly21_pct=monthly*100,EndingBalance=equity,
                NetProfit=equity-10000,AvgWin=avgwin,AvgLoss=avgloss,
                AvgWinLossRatio=(avgwin/avgloss if avgloss>0 else None))


def stats(rows,key):
    vals=np.array([r[key] for r in rows if r[key] is not None],float)
    vals=vals[np.isfinite(vals)]
    return {'median':float(np.median(vals)),'p10':float(np.percentile(vals,10)),'p90':float(np.percentile(vals,90))}


def main():
    df=pd.read_csv(DATA)
    df.columns=[x.lower() for x in df.columns]
    df['datetime']=pd.to_datetime(df['datetime'])
    df=df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    df['atr']=atr14(df)
    signals=build_signals(df)

    keys=['N','WR_pct','PF','MaxDD_pct','Return90d_pct','Monthly21_pct','EndingBalance','NetProfit','AvgWin','AvgLoss','AvgWinLossRatio']
    all_rows=[]; summary={}
    for tp in TP_MULTS:
        for ta in TRAIL_ACTS:
            ck=f'tp{tp:.2f}_trail{ta:.2f}'
            summary[ck]={}
            for p in PROFILES:
                seeds=[260827] if p.name=='IDEAL' else list(range(260827,260927))
                rows=[run_once(df,signals,p,s,tp,ta) for s in seeds]
                all_rows.extend([{**r,'profile':p.name,'seed':s} for r,s in zip(rows,seeds)])
                summary[ck][p.name]={k:stats(rows,k) for k in keys}

    # Rank first by NORMAL PF, then NORMAL monthly return, subject to TAIL p90 DD <= 3%.
    candidates=[]
    for ck,v in summary.items():
        tail_dd=v['TAIL']['MaxDD_pct']['p90']
        if tail_dd <= TARGET_DD_PCT:
            candidates.append((v['NORMAL']['PF']['median'],v['NORMAL']['Monthly21_pct']['median'],ck,tail_dd))
    recommended=max(candidates) if candidates else None

    report={
        'engine':'NautilusTrader runtime + deterministic clean-room execution harness',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'status':'CLEAN_ROOM_TP075_SWEEP_PROXY',
        'data':str(DATA),
        'risk_mult':RISK_MULT,
        'tp_mults':TP_MULTS,
        'trail_acts':TRAIL_ACTS,
        'trail_dist':TRAIL_DIST,
        'target_dd_pct':TARGET_DD_PCT,
        'recommended_by_normal_pf_then_return_under_tail_dd3':recommended,
        'profiles':[asdict(x) for x in PROFILES],
        'summary':summary,
        'limitations':['M1 OHLC not raw Bid/Ask ticks','risk scaling is lot-linear proxy','intrabar order modeled conservatively','broker noise distributions are proxies']
    }
    pd.DataFrame(all_rows).to_csv(OUT/'all_runs.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
