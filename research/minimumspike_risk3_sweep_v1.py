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

OUT = Path('results/minimumspike-risk3-sweep-v1')
OUT.mkdir(parents=True, exist_ok=True)

# Risk is expressed as a lot/PnL multiplier vs the 0.01-lot clean-room baseline.
# Sweep around the level expected to bring TAIL p90 DD close to, but not above, 3%.
RISK_MULTIPLIERS = [1.0, 2.0, 2.5, 2.75, 2.80, 3.0]
TARGET_DD_PCT = 3.0


def run_once(df, signals, p, seed: int, risk_mult: float):
    rng=np.random.default_rng(seed)
    o,h,l,c=[df[x].to_numpy() for x in ['open','high','low','close']]
    equity=10000.0; peak=equity; maxdd=0.0
    trades=[]; costs={'spread':0.,'slippage':0.,'latency':0.,'tail':0.,'commission':0.}
    next_free=0
    for entry_i, spike_i, atr in signals:
        if entry_i < next_free: continue
        if rng.random() < p.reject_prob: continue
        ec,sp,sl,la,ta=draw_cost(rng,p,atr)
        entry=o[entry_i]+ec
        costs['spread']+=(sp/2)*risk_mult
        costs['slippage']+=sl*risk_mult
        costs['latency']+=la*risk_mult
        costs['tail']+=ta*risk_mult
        stop=l[spike_i]-PARAMS['guard_ext']*atr
        tp=entry+PARAMS['tp_atr']*atr
        trail=None; exit_px=None; exit_i=None
        end=min(len(df)-1, entry_i+PARAMS['max_hold'])
        for k in range(entry_i,end+1):
            held=k-entry_i
            active_stop=max(stop, trail) if trail is not None else stop
            if l[k] <= active_stop and held >= PARAMS['min_hold']:
                exit_px=active_stop; exit_i=k; break
            if h[k] >= tp and held >= PARAMS['min_hold']:
                exit_px=tp; exit_i=k; break
            if h[k] >= entry+PARAMS['trail_act']*atr:
                newtrail=h[k]-PARAMS['trail_dist']*atr
                trail=newtrail if trail is None else max(trail,newtrail)
        if exit_px is None:
            exit_i=end; exit_px=c[end]
        xc,sp2,sl2,la2,ta2=draw_cost(rng,p,atr)
        exit_fill=exit_px-xc
        costs['spread']+=(sp2/2)*risk_mult
        costs['slippage']+=sl2*risk_mult
        costs['latency']+=la2*risk_mult
        costs['tail']+=ta2*risk_mult
        gross=(exit_fill-entry)*risk_mult
        commission=p.commission_rt*risk_mult
        net=gross-commission
        costs['commission']+=commission
        equity+=net; peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/peak*100)
        trades.append(net)
        next_free=exit_i+1
    a=np.asarray(trades,float)
    wins=a[a>0]; losses=a[a<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    wr=float((a>0).mean()*100) if len(a) else 0.0
    ret=(equity/10000-1)*100
    monthly=(equity/10000)**(21/90)-1 if equity>0 else -1
    return dict(risk_mult=risk_mult,N=len(a),WR_pct=wr,PF=pf,MaxDD_pct=maxdd,
                Return90d_pct=ret,Monthly21_pct=monthly*100,EndingBalance=equity,
                NetProfit=equity-10000,noise_cost_usd=sum(costs.values()),
                **{f'cost_{k}_usd':v for k,v in costs.items()})


def stats(rows, key):
    vals=np.array([r[key] for r in rows],float)
    vals=vals[np.isfinite(vals)]
    return {'median':float(np.median(vals)),'p10':float(np.percentile(vals,10)),'p90':float(np.percentile(vals,90))}


def main():
    df=pd.read_csv(DATA)
    df.columns=[x.lower() for x in df.columns]
    df['datetime']=pd.to_datetime(df['datetime'])
    df=df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    df['atr']=atr14(df)
    signals=build_signals(df)

    keys=['N','WR_pct','PF','MaxDD_pct','Return90d_pct','Monthly21_pct','EndingBalance','NetProfit','noise_cost_usd']
    all_rows=[]; summary={}
    for m in RISK_MULTIPLIERS:
        summary[str(m)]={}
        for p in PROFILES:
            seeds=[260827] if p.name=='IDEAL' else list(range(260827,260927))
            rows=[run_once(df,signals,p,s,m) for s in seeds]
            for r in rows:
                r['profile']=p.name; r['seed']=s if False else None
            all_rows.extend([{**r,'profile':p.name} for r in rows])
            summary[str(m)][p.name]={k:stats(rows,k) for k in keys}

    # Conservative recommendation: highest multiplier whose TAIL p90 MaxDD <= target.
    eligible=[]
    for m in RISK_MULTIPLIERS:
        dd=summary[str(m)]['TAIL']['MaxDD_pct']['p90']
        if dd <= TARGET_DD_PCT:
            eligible.append((m,dd))
    recommended=max(eligible,key=lambda x:x[0]) if eligible else None

    report={
        'engine':'NautilusTrader runtime + deterministic clean-room execution harness',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'status':'CLEAN_ROOM_RISK_SWEEP_PROXY',
        'data':str(DATA),
        'target_dd_pct':TARGET_DD_PCT,
        'risk_multipliers':RISK_MULTIPLIERS,
        'recommended_by_tail_p90_dd':recommended,
        'profiles':[asdict(x) for x in PROFILES],
        'summary':summary,
        'limitations':['M1 OHLC not raw Bid/Ask ticks','risk scaling is lot-linear proxy','intrabar order modeled conservatively','broker noise distributions are proxies']
    }
    pd.DataFrame(all_rows).to_csv(OUT/'all_runs.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
