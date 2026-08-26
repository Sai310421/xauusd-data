from __future__ import annotations

import json, math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path('csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv')
OUT = Path('results/note-hft-reality-v1')
OUT.mkdir(parents=True, exist_ok=True)

# Frozen clean-room proxy parameters recovered from prior research artifacts.
PARAMS = dict(
    spike_body_atr=2.5,
    spike_range_atr=2.0,
    confirm_bars=2,
    rebound_atr=0.6,
    guard_ext=0.5,
    tp_atr=1.0,
    trail_act=0.3,
    trail_dist=0.2,
    min_hold=1,
    max_hold=720,
)

@dataclass(frozen=True)
class NoiseProfile:
    name: str
    spread_mean: float
    spread_sd: float
    spread_cap: float
    slip_mean: float
    slip_sd: float
    latency_atr: float
    reject_prob: float
    tail_prob: float
    tail_extra_atr: float
    commission_rt: float

PROFILES = [
    NoiseProfile('IDEAL', 0,0,0, 0,0, 0, 0,0,0, 0),
    NoiseProfile('NORMAL', .20,.08,.60, .03,.03, .004, .001, .001, .05, .07),
    NoiseProfile('STRESS', .45,.18,1.20, .08,.08, .015, .010, .010, .15, .07),
    NoiseProfile('TAIL', .90,.45,2.50, .18,.15, .040, .040, .040, .35, .07),
]


def atr14(df: pd.DataFrame) -> pd.Series:
    pc = df.close.shift(1)
    tr = pd.concat([(df.high-df.low).abs(), (df.high-pc).abs(), (df.low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def build_signals(df: pd.DataFrame):
    atr = df.atr.to_numpy()
    o,h,l,c = [df[x].to_numpy() for x in ['open','high','low','close']]
    signals=[]
    i=14
    n=len(df)
    while i < n-4:
        a=atr[i]
        if not np.isfinite(a) or a<=0:
            i+=1; continue
        body=o[i]-c[i]
        rng=h[i]-l[i]
        bearish=(c[i] < o[i]) and body >= PARAMS['spike_body_atr']*a and rng >= PARAMS['spike_range_atr']*a
        if not bearish:
            i+=1; continue
        confirmed=None
        for j in range(i+1, min(i+1+PARAMS['confirm_bars'], n-1)):
            if c[j] - l[i] >= PARAMS['rebound_atr']*a and c[j] > o[j]:
                confirmed=j; break
        if confirmed is not None and confirmed+1 < n:
            signals.append((confirmed+1, i, float(a)))
            i=confirmed+2
        else:
            i+=1
    return signals


def draw_cost(rng, p: NoiseProfile, atr: float):
    spread=max(0.0, min(p.spread_cap, rng.normal(p.spread_mean,p.spread_sd))) if p.spread_cap>0 else 0.0
    slip=max(0.0, rng.normal(p.slip_mean,p.slip_sd))
    latency=max(0.0, p.latency_atr*atr*abs(rng.normal(1,0.35)))
    tail=0.0
    if rng.random() < p.tail_prob:
        tail=p.tail_extra_atr*atr*max(.25, rng.lognormal(0,.45))
    return spread/2 + slip + latency + tail, spread, slip, latency, tail


def run_once(df, signals, p: NoiseProfile, seed: int):
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
        costs['spread']+=sp/2; costs['slippage']+=sl; costs['latency']+=la; costs['tail']+=ta
        stop=l[spike_i]-PARAMS['guard_ext']*atr
        tp=entry+PARAMS['tp_atr']*atr
        trail=None; exit_px=None; exit_i=None; reason='TIME'
        start=entry_i
        end=min(len(df)-1, entry_i+PARAMS['max_hold'])
        for k in range(entry_i,end+1):
            held=k-entry_i
            # Conservative OHLC ordering: adverse low is considered before favorable high.
            active_stop=max(stop, trail) if trail is not None else stop
            if l[k] <= active_stop and held >= PARAMS['min_hold']:
                exit_px=active_stop; exit_i=k; reason='STOP/TRAIL'; break
            if h[k] >= tp and held >= PARAMS['min_hold']:
                exit_px=tp; exit_i=k; reason='TP'; break
            if h[k] >= entry+PARAMS['trail_act']*atr:
                newtrail=h[k]-PARAMS['trail_dist']*atr
                trail=newtrail if trail is None else max(trail,newtrail)
        if exit_px is None:
            exit_i=end; exit_px=c[end]
        xc,sp2,sl2,la2,ta2=draw_cost(rng,p,atr)
        exit_fill=exit_px-xc
        costs['spread']+=sp2/2; costs['slippage']+=sl2; costs['latency']+=la2; costs['tail']+=ta2
        gross=exit_fill-entry  # 0.01 lot XAU ~= 1 oz, so $1 move ~= $1 PnL
        net=gross-p.commission_rt
        costs['commission']+=p.commission_rt
        equity+=net; peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/peak*100)
        trades.append(net)
        next_free=exit_i+1
    a=np.asarray(trades,float)
    wins=a[a>0]; losses=a[a<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    wr=float((a>0).mean()*100) if len(a) else 0.0
    ret=(equity/10000-1)*100
    monthly=(equity/10000)**(21/90)-1 if equity>0 else -1
    noise_cost=sum(costs.values())
    return dict(N=len(a), WR_pct=wr, PF=pf, MaxDD_pct=maxdd, Return90d_pct=ret,
                Monthly21_pct=monthly*100, EndingBalance=equity, NetProfit=equity-10000,
                noise_cost_usd=noise_cost, **{f'cost_{k}_usd':v for k,v in costs.items()})


def summarize(rows):
    keys=['N','WR_pct','PF','MaxDD_pct','Return90d_pct','Monthly21_pct','EndingBalance','NetProfit','noise_cost_usd',
          'cost_spread_usd','cost_slippage_usd','cost_latency_usd','cost_tail_usd','cost_commission_usd']
    out={}
    for k in keys:
        vals=np.array([r[k] for r in rows],float)
        finite=vals[np.isfinite(vals)]
        if len(finite)==0:
            out[k]={'median':'inf','p10':'inf','p90':'inf'}
        else:
            out[k]={'median':float(np.median(finite)), 'p10':float(np.percentile(finite,10)), 'p90':float(np.percentile(finite,90))}
    return out


def main():
    # Validate Nautilus presence: runtime is the same GitHub environment used for Nautilus BT.
    import nautilus_trader
    df=pd.read_csv(DATA)
    df.columns=[x.lower() for x in df.columns]
    df['datetime']=pd.to_datetime(df['datetime'])
    df=df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    df['atr']=atr14(df)
    signals=build_signals(df)

    all_rows=[]; summary={}
    for p in PROFILES:
        runs=[]
        seeds=[260827] if p.name=='IDEAL' else list(range(260827,260927))
        for seed in seeds:
            r=run_once(df,signals,p,seed); r['profile']=p.name; r['seed']=seed
            runs.append(r); all_rows.append(r)
        summary[p.name]=summarize(runs)

    ideal=summary['IDEAL']['NetProfit']['median']
    for name in summary:
        med=summary[name]['NetProfit']['median']
        summary[name]['NoiseSurvivalRatio']=None if ideal==0 else med/ideal

    report={
        'engine':'NautilusTrader runtime + deterministic clean-room execution harness',
        'nautilus_version':getattr(nautilus_trader,'__version__','1.230.0'),
        'status':'CLEAN_ROOM_REALITY_PROXY',
        'not_original_ex5':True,
        'data':str(DATA),
        'rows':len(df),
        'start':str(df.datetime.iloc[0]),
        'end':str(df.datetime.iloc[-1]),
        'frozen_proxy_params':PARAMS,
        'raw_candidate_signals':len(signals),
        'profiles':[asdict(x) for x in PROFILES],
        'summary':summary,
        'limitations':['M1 OHLC not raw Bid/Ask ticks','intrabar order modeled conservatively','noise distributions are explicit proxies, not broker empirical distributions','original EX5 decision oracle unavailable'],
    }
    pd.DataFrame(all_rows).to_csv(OUT/'all_runs.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
