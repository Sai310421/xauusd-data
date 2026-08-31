from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path('csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv')
OUT = Path('results/chochomaru-cleanroom-discovery-v1')
OUT.mkdir(parents=True, exist_ok=True)

# Clean-room reconstruction only. Not original CHOCHOMARU code.
PARAMS = dict(
    fast_ema=9,
    mid_ema=20,
    slow_ema=50,
    trend_ema=200,
    rsi_period=14,
    atr_period=14,
    min_atr=0.40,
    cosmic_rsi_low=28.0,
    cosmic_rsi_recover=36.0,
    pullback_atr_max=0.90,
    resume_body_ratio=0.55,
    range_lookback=24,
    max_range_atr=5.0,
    break_atr_min=0.10,
    break_body_ratio=0.55,
    retest_atr=0.20,
    sl_atr=1.80,
    tp_atr=0.85,
    be_trigger_atr=0.55,
    be_offset_atr=0.05,
    trail_start_atr=0.75,
    trail_atr=0.45,
    max_hold_bars=180,
    delayed_stop_confirm_bars=2,
    delayed_stop_hardcap_atr=2.70,
    delayed_stop_reclaim_atr=0.10,
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
    NoiseProfile('NORMAL', .20,.08,.60, .03,.03, .004, .001,.001,.05,.07),
    NoiseProfile('STRESS', .45,.18,1.20, .08,.08, .015, .010,.010,.15,.07),
    NoiseProfile('TAIL', .90,.45,2.50, .18,.15, .040, .040,.040,.35,.07),
]

MODES = ['COSMIC', 'FLORA', 'ECLIPSE', 'FUSION']


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df.close.shift(1)
    tr = pd.concat([(df.high-df.low).abs(), (df.high-pc).abs(), (df.low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x['atr'] = atr(x, PARAMS['atr_period'])
    x['rsi'] = rsi(x.close, PARAMS['rsi_period'])
    for k,n in [('ef',PARAMS['fast_ema']),('em',PARAMS['mid_ema']),('es',PARAMS['slow_ema']),('et',PARAMS['trend_ema'])]:
        x[k] = x.close.ewm(span=n, adjust=False).mean()
    rng = (x.high-x.low).replace(0, np.nan)
    x['body_ratio'] = ((x.close-x.open).abs()/rng).fillna(0.0)
    x['range_high'] = x.high.shift(3).rolling(PARAMS['range_lookback']).max()
    x['range_low'] = x.low.shift(3).rolling(PARAMS['range_lookback']).min()
    return x


def build_signals(df: pd.DataFrame, mode: str):
    sig=[]
    for i in range(max(205, PARAMS['range_lookback']+5), len(df)-2):
        r1=df.iloc[i-1]; r2=df.iloc[i-2]
        if not np.isfinite(r1.atr) or r1.atr < PARAMS['min_atr']:
            continue
        bull = r1.ef>r1.em>r1.es and r1.close>r1.et
        bear = r1.ef<r1.em<r1.es and r1.close<r1.et
        cosmic_a_buy = bull and (r1.low <= r1.ef + PARAMS['pullback_atr_max']*r1.atr) and (r1.low >= r1.es - .25*r1.atr) and (r1.close>r1.open) and (r1.close>r1.ef) and (r1.body_ratio>=PARAMS['resume_body_ratio'])
        cosmic_b_buy = (r1.close>r1.et and r2.rsi<=PARAMS['cosmic_rsi_low'] and r1.rsi>=PARAMS['cosmic_rsi_recover'] and r1.close>r1.ef and r1.close>r2.close)
        cosmic_a_sell = bear and (r1.high >= r1.ef - PARAMS['pullback_atr_max']*r1.atr) and (r1.high <= r1.es + .25*r1.atr) and (r1.close<r1.open) and (r1.close<r1.ef) and (r1.body_ratio>=PARAMS['resume_body_ratio'])
        cosmic_b_sell = (r1.close<r1.et and r2.rsi>=(100-PARAMS['cosmic_rsi_low']) and r1.rsi<=(100-PARAMS['cosmic_rsi_recover']) and r1.close<r1.ef and r1.close<r2.close)

        rh,rl=r1.range_high,r1.range_low
        flora_buy=flora_sell=False
        if np.isfinite(rh) and np.isfinite(rl):
            compressed=(rh-rl) <= PARAMS['max_range_atr']*r1.atr
            break_up=(r2.close > rh + PARAMS['break_atr_min']*r2.atr and r2.body_ratio>=PARAMS['break_body_ratio'])
            break_dn=(r2.close < rl - PARAMS['break_atr_min']*r2.atr and r2.body_ratio>=PARAMS['break_body_ratio'])
            retest_up=(r1.low<=rh+PARAMS['retest_atr']*r1.atr and r1.close>rh and r1.close>r1.open)
            retest_dn=(r1.high>=rl-PARAMS['retest_atr']*r1.atr and r1.close<rl and r1.close<r1.open)
            flora_buy=compressed and break_up and retest_up
            flora_sell=compressed and break_dn and retest_dn

        ca,cb = cosmic_a_buy, cosmic_b_buy
        sa,sb = cosmic_a_sell, cosmic_b_sell
        buy=sell=False; reason=''
        if mode=='COSMIC':
            buy=ca or cb
            # COSMIC observed as long-only candidate.
            sell=False
            reason='COSMIC_A' if ca else ('COSMIC_B' if cb else '')
        elif mode in ('FLORA','ECLIPSE'):
            buy,sell=flora_buy,flora_sell
            reason='FLORA_RETEST'
        else:
            score_b=(1.0 if flora_buy else 0)+(0.6 if ca else 0)+(0.6 if cb else 0)
            score_s=(1.0 if flora_sell else 0)+(0.6 if sa else 0)+(0.6 if sb else 0)
            buy = score_b>=1.0 and not (score_s>score_b)
            sell = score_s>=1.0 and not (score_b>score_s)
            reason='FUSION'
        if buy ^ sell:
            sig.append(dict(entry_i=i, side=1 if buy else -1, atr=float(r1.atr), reason=reason))
    return sig


def draw_cost(rng, p: NoiseProfile, a: float):
    spread=max(0.0,min(p.spread_cap,rng.normal(p.spread_mean,p.spread_sd))) if p.spread_cap>0 else 0.0
    slip=max(0.0,rng.normal(p.slip_mean,p.slip_sd))
    latency=max(0.0,p.latency_atr*a*abs(rng.normal(1,0.35)))
    tail=0.0
    if rng.random()<p.tail_prob:
        tail=p.tail_extra_atr*a*max(.25,rng.lognormal(0,.45))
    return spread/2+slip+latency+tail, dict(spread=spread/2,slippage=slip,latency=latency,tail=tail)


def run_once(df, signals, mode, profile, seed):
    rng=np.random.default_rng(seed)
    equity=10000.0; peak=equity; maxdd=0.0; next_free=0
    trades=[]; costs={'spread':0.,'slippage':0.,'latency':0.,'tail':0.,'commission':0.}
    for s in signals:
        i=s['entry_i']; side=s['side']; a=s['atr']
        if i<next_free: continue
        if rng.random()<profile.reject_prob: continue
        ec,cc=draw_cost(rng,profile,a)
        for k,v in cc.items(): costs[k]+=v
        raw_entry=float(df.open.iloc[i])
        entry=raw_entry + ec*side
        vsl=entry - side*PARAMS['sl_atr']*a
        hard=entry - side*PARAMS['delayed_stop_hardcap_atr']*a
        tp=entry + side*PARAMS['tp_atr']*a
        broker_stop = hard if mode in ('ECLIPSE','FUSION') else vsl
        trail=None; armed=0; exit_px=None; exit_i=None; reason='TIME'
        end=min(len(df)-1,i+PARAMS['max_hold_bars'])
        for k in range(i,end+1):
            hi=float(df.high.iloc[k]); lo=float(df.low.iloc[k]); close=float(df.close.iloc[k]); ak=float(df.atr.iloc[k]) if np.isfinite(df.atr.iloc[k]) else a
            favorable=(hi-entry) if side==1 else (entry-lo)
            if favorable>=PARAMS['be_trigger_atr']*a:
                be=entry+side*PARAMS['be_offset_atr']*a
                trail=be if trail is None else (max(trail,be) if side==1 else min(trail,be))
            if favorable>=PARAMS['trail_start_atr']*a:
                tr=(hi-PARAMS['trail_atr']*ak) if side==1 else (lo+PARAMS['trail_atr']*ak)
                trail=tr if trail is None else (max(trail,tr) if side==1 else min(trail,tr))
            active_stop=broker_stop
            if trail is not None:
                active_stop=max(active_stop,trail) if side==1 else min(active_stop,trail)
            # Conservative OHLC ordering: adverse path first.
            stop_hit=(lo<=active_stop) if side==1 else (hi>=active_stop)
            if stop_hit:
                exit_px=active_stop; exit_i=k; reason='STOP/TRAIL'; break
            tp_hit=(hi>=tp) if side==1 else (lo<=tp)
            if tp_hit:
                exit_px=tp; exit_i=k; reason='TP'; break
            if mode in ('ECLIPSE','FUSION'):
                crossed=(lo<=vsl) if side==1 else (hi>=vsl)
                reclaimed=(close>=vsl+PARAMS['delayed_stop_reclaim_atr']*ak) if side==1 else (close<=vsl-PARAMS['delayed_stop_reclaim_atr']*ak)
                if crossed:
                    armed+=1
                    if reclaimed: armed=0
                    elif armed>=PARAMS['delayed_stop_confirm_bars']:
                        exit_px=close; exit_i=k; reason='DELAYED_SL'; break
        if exit_px is None:
            exit_i=end; exit_px=float(df.close.iloc[end])
        xc,cc2=draw_cost(rng,profile,a)
        for k,v in cc2.items(): costs[k]+=v
        exit_fill=exit_px - xc*side
        gross=(exit_fill-entry)*side
        net=gross-profile.commission_rt
        costs['commission']+=profile.commission_rt
        equity+=net; peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/peak*100)
        trades.append(dict(mode=mode,profile=profile.name,seed=seed,entry_i=i,exit_i=exit_i,side=side,net=net,reason=reason,signal=s['reason']))
        next_free=exit_i+1
    a=np.array([x['net'] for x in trades],float)
    wins=a[a>0]; losses=a[a<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    wr=float((a>0).mean()*100) if len(a) else 0.0
    ret=(equity/10000-1)*100
    days=max(1,(df.datetime.iloc[-1]-df.datetime.iloc[0]).days)
    monthly=((equity/10000)**(21/days)-1)*100 if equity>0 else -100.0
    return dict(mode=mode,profile=profile.name,seed=seed,N=len(a),WR_pct=wr,PF=pf,MaxDD_pct=maxdd,Return_pct=ret,Monthly21_pct=monthly,EndingBalance=equity,NetProfit=equity-10000,**{f'cost_{k}_usd':v for k,v in costs.items()}), trades


def summarize(rows):
    metrics=['N','WR_pct','PF','MaxDD_pct','Return_pct','Monthly21_pct','EndingBalance','NetProfit']
    out={}
    for k in metrics:
        x=np.array([r[k] for r in rows],float)
        f=x[np.isfinite(x)]
        out[k]=dict(median=float(np.median(f)) if len(f) else None,p10=float(np.percentile(f,10)) if len(f) else None,p90=float(np.percentile(f,90)) if len(f) else None)
    return out


def main():
    import nautilus_trader
    df=pd.read_csv(DATA)
    df.columns=[c.lower() for c in df.columns]
    df['datetime']=pd.to_datetime(df['datetime'],utc=True)
    df=df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    df=add_features(df)
    all_runs=[]; all_trades=[]; report={}
    for mode in MODES:
        signals=build_signals(df,mode)
        report[mode]={'raw_candidate_signals':len(signals),'profiles':{}}
        for p in PROFILES:
            seeds=[260831] if p.name=='IDEAL' else list(range(260831,260881))
            rows=[]
            for seed in seeds:
                r,t=run_once(df,signals,mode,p,seed)
                rows.append(r); all_runs.append(r); all_trades.extend(t)
            report[mode]['profiles'][p.name]=summarize(rows)
    payload={
        'engine':'NautilusTrader runtime + clean-room deterministic discovery harness',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
        'status':'DISCOVERY_ONLY_NOT_RAW_TICK_VALIDATION',
        'not_original_ex4':True,
        'data':str(DATA),
        'rows':len(df),
        'start':str(df.datetime.iloc[0]),
        'end':str(df.datetime.iloc[-1]),
        'params':PARAMS,
        'noise_profiles':[asdict(x) for x in PROFILES],
        'results':report,
        'limitations':['M1 mid-quote OHLC, not raw Bid/Ask ticks','intrabar ordering modeled conservatively','noise distributions are proxies','original IB-locked EA decision oracle unavailable'],
    }
    pd.DataFrame(all_runs).to_csv(OUT/'all_runs.csv',index=False)
    pd.DataFrame(all_trades).to_csv(OUT/'trades.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(payload,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
