from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import nautilus_trader

from research.minimumspike_reality_noise_bt_v1 import PARAMS, PROFILES, atr14, draw_cost

DATA = Path('csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv')
OUT = Path('results/minimumspike-mtf-boost-v1')
OUT.mkdir(parents=True, exist_ok=True)

TFS = {
    'M1': '1min',
    'M5': '5min',
    'M15': '15min',
    'M30': '30min',
    'H1': '1h',
    'H4': '4h',
    'D1': '1D',
}

TP_MULT = 1.00
TRAIL_ACT = 0.40
TRAIL_DIST = 0.20
TARGET_PORTFOLIO_DD_PCT = 3.0

# Base risk is intentionally reduced from the single-TF 2.8x setting because
# multiple TFs may fire simultaneously. Confluence adds size only when signals overlap.
BASE_RISK = 0.80
BOOST_BY_COUNT = {
    1: 1.00,
    2: 1.25,
    3: 1.50,
    4: 1.65,
    5: 1.75,
    6: 1.85,
    7: 2.00,
}
MAX_EVENT_RISK = 2.80


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.set_index('datetime')
    y = x.resample(rule, label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    y['atr'] = atr14(y)
    return y


def build_signals(df: pd.DataFrame):
    atr = df.atr.to_numpy()
    o,h,l,c = [df[x].to_numpy() for x in ['open','high','low','close']]
    signals=[]
    i=14; n=len(df)
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
            entry_i=confirmed+1
            signals.append({
                'entry_i':entry_i,
                'spike_i':i,
                'atr':float(a),
                'entry_time':df.datetime.iloc[entry_i],
            })
            i=confirmed+2
        else:
            i+=1
    return signals


def confluence_counts(tf_signals: dict[str, list[dict]]) -> dict[tuple[str,int], int]:
    by_time=defaultdict(list)
    for tf, sigs in tf_signals.items():
        for idx,s in enumerate(sigs):
            by_time[pd.Timestamp(s['entry_time'])].append((tf,idx))
    out={}
    for _,items in by_time.items():
        count=len(items)
        for key in items:
            out[key]=count
    return out


def run_tf(df, signals, p, seed: int, tf: str, conf: dict[tuple[str,int],int]):
    rng=np.random.default_rng(seed + (abs(hash(tf)) % 100000))
    o,h,l,c=[df[x].to_numpy() for x in ['open','high','low','close']]
    rows=[]; next_free=0
    for si,s in enumerate(signals):
        entry_i=s['entry_i']; spike_i=s['spike_i']; atr=s['atr']
        if entry_i < next_free: continue
        if rng.random() < p.reject_prob: continue
        cc=conf.get((tf,si),1)
        boost=BOOST_BY_COUNT.get(cc, BOOST_BY_COUNT[max(BOOST_BY_COUNT)])
        risk=min(MAX_EVENT_RISK, BASE_RISK*boost)
        ec,_,_,_,_=draw_cost(rng,p,atr)
        entry=o[entry_i]+ec
        stop=l[spike_i]-PARAMS['guard_ext']*atr
        tp=entry+TP_MULT*atr
        trail=None; exit_px=None; exit_i=None
        end=min(len(df)-1, entry_i+PARAMS['max_hold'])
        for k in range(entry_i,end+1):
            held=k-entry_i
            active_stop=max(stop,trail) if trail is not None else stop
            if l[k] <= active_stop and held >= PARAMS['min_hold']:
                exit_px=active_stop; exit_i=k; break
            if h[k] >= tp and held >= PARAMS['min_hold']:
                exit_px=tp; exit_i=k; break
            if h[k] >= entry+TRAIL_ACT*atr:
                nt=h[k]-TRAIL_DIST*atr
                trail=nt if trail is None else max(trail,nt)
        if exit_px is None:
            exit_i=end; exit_px=c[end]
        xc,_,_,_,_=draw_cost(rng,p,atr)
        net=((exit_px-xc)-entry)*risk - p.commission_rt*risk
        rows.append({
            'tf':tf,'entry_time':str(s['entry_time']),'net':float(net),
            'confluence':cc,'risk_mult':risk,
        })
        next_free=exit_i+1
    return rows


def summarize(rows):
    if not rows:
        return dict(N=0,WR_pct=0.0,PF=0.0,NetProfit=0.0,MaxDD_pct=0.0)
    a=np.array([r['net'] for r in rows],float)
    wins=a[a>0]; losses=a[a<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    wr=float((a>0).mean()*100)
    eq=10000.0; peak=eq; mdd=0.0
    for x in a:
        eq+=x; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100)
    return dict(N=len(a),WR_pct=wr,PF=pf,NetProfit=float(a.sum()),MaxDD_pct=mdd,
                Return90d_pct=(eq/10000-1)*100,
                Monthly21_pct=((eq/10000)**(21/90)-1)*100 if eq>0 else -100.0,
                AvgRiskMult=float(np.mean([r['risk_mult'] for r in rows])),
                BoostedTrades=int(sum(r['confluence']>=2 for r in rows)))


def main():
    base=pd.read_csv(DATA)
    base.columns=[x.lower() for x in base.columns]
    base['datetime']=pd.to_datetime(base['datetime'])
    base=base.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)

    tf_data={tf:resample_ohlc(base,rule) for tf,rule in TFS.items()}
    tf_signals={tf:build_signals(df) for tf,df in tf_data.items()}
    conf=confluence_counts(tf_signals)

    report={
        'engine':'NautilusTrader runtime + deterministic clean-room MTF portfolio harness',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'status':'CLEAN_ROOM_MTF_BOOST_PROXY',
        'data':str(DATA),
        'timeframes':list(TFS),
        'tp_mult':TP_MULT,'trail_act':TRAIL_ACT,'trail_dist':TRAIL_DIST,
        'base_risk':BASE_RISK,'boost_by_count':BOOST_BY_COUNT,'max_event_risk':MAX_EVENT_RISK,
        'target_portfolio_dd_pct':TARGET_PORTFOLIO_DD_PCT,
        'raw_signal_count_by_tf':{tf:len(v) for tf,v in tf_signals.items()},
        'profiles':{},
        'limitations':['M1 OHLC resampled to higher TFs, not raw Bid/Ask ticks','portfolio concurrency is approximated by event aggregation','risk scaling is lot-linear proxy','broker noise distributions are proxies']
    }

    all_out=[]
    for p in PROFILES:
        seeds=[260828] if p.name=='IDEAL' else list(range(260828,260878))
        runs=[]
        tf_accum=defaultdict(list)
        for seed in seeds:
            rows=[]
            for tf in TFS:
                rr=run_tf(tf_data[tf],tf_signals[tf],p,seed,tf,conf)
                rows.extend(rr)
                tf_accum[tf].extend(rr)
            # deterministic portfolio ordering by entry time, then TF
            rows=sorted(rows,key=lambda r:(r['entry_time'],r['tf']))
            runs.append(summarize(rows))
            all_out.extend([{**r,'profile':p.name,'seed':seed} for r in rows])
        keys=['N','WR_pct','PF','NetProfit','MaxDD_pct','Return90d_pct','Monthly21_pct','AvgRiskMult','BoostedTrades']
        def stat(k):
            vals=np.array([r[k] for r in runs],float)
            vals=vals[np.isfinite(vals)]
            return {'median':float(np.median(vals)),'p10':float(np.percentile(vals,10)),'p90':float(np.percentile(vals,90))}
        report['profiles'][p.name]={k:stat(k) for k in keys}
        report['profiles'][p.name]['by_tf']={tf:summarize(tf_accum[tf]) for tf in TFS}

    pd.DataFrame(all_out).to_csv(OUT/'trades.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
