from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import chochomaru_cleanroom_discovery_bt_v1 as base

DATA = base.DATA
OUT = Path('results/chochomaru-dynamic-atr-exit-v2')
OUT.mkdir(parents=True, exist_ok=True)

EXIT_VARIANTS = ['FIXED_TP', 'DYNAMIC_ATR', 'ATR_MFE_LOCK']
MODES = ['COSMIC', 'FLORA', 'ECLIPSE', 'FUSION']

# v2 focuses on RR improvement while preserving entry logic.
V2 = dict(
    initial_sl_atr=1.80,
    fixed_tp_atr=0.85,
    be_trigger_atr=0.55,
    be_offset_atr=0.05,
    dyn_start_atr=0.55,
    dyn_k_early=1.20,
    dyn_k_mid=0.85,
    dyn_k_late=0.55,
    mfe_lock_1=0.35,
    mfe_lock_2=0.50,
    mfe_lock_3=0.65,
    max_hold_bars=240,
)


def trail_k(mfe_atr: float) -> float:
    if mfe_atr < 1.0:
        return V2['dyn_k_early']
    if mfe_atr < 2.0:
        return V2['dyn_k_mid']
    return V2['dyn_k_late']


def lock_fraction(mfe_atr: float) -> float:
    if mfe_atr < 1.0:
        return V2['mfe_lock_1']
    if mfe_atr < 2.0:
        return V2['mfe_lock_2']
    return V2['mfe_lock_3']


def better_stop(side: int, current: float | None, candidate: float) -> float:
    if current is None:
        return candidate
    return max(current, candidate) if side == 1 else min(current, candidate)


def run_once(df, signals, mode, profile, seed, exit_variant):
    rng = np.random.default_rng(seed)
    equity = 10000.0
    peak = equity
    maxdd = 0.0
    next_free = 0
    trades = []
    costs = {'spread':0., 'slippage':0., 'latency':0., 'tail':0., 'commission':0.}

    for s in signals:
        i, side, a = s['entry_i'], s['side'], s['atr']
        if i < next_free or rng.random() < profile.reject_prob:
            continue

        ec, cc = base.draw_cost(rng, profile, a)
        for k,v in cc.items(): costs[k] += v
        entry = float(df.open.iloc[i]) + ec * side
        initial_sl = entry - side * V2['initial_sl_atr'] * a
        fixed_tp = entry + side * V2['fixed_tp_atr'] * a
        stop = initial_sl
        mfe = 0.0
        exit_px = None
        exit_i = None
        reason = 'TIME'
        end = min(len(df)-1, i + V2['max_hold_bars'])

        for k in range(i, end+1):
            hi = float(df.high.iloc[k]); lo = float(df.low.iloc[k])
            close = float(df.close.iloc[k])
            ak = float(df.atr.iloc[k]) if np.isfinite(df.atr.iloc[k]) else a
            favorable = (hi-entry) if side == 1 else (entry-lo)
            mfe = max(mfe, favorable)
            mfe_atr = mfe / a if a > 0 else 0.0

            # Break-even protection is common to all adaptive exits.
            if exit_variant != 'FIXED_TP' and mfe_atr >= V2['be_trigger_atr']:
                be = entry + side * V2['be_offset_atr'] * a
                stop = better_stop(side, stop, be)

            if exit_variant in ('DYNAMIC_ATR', 'ATR_MFE_LOCK') and mfe_atr >= V2['dyn_start_atr']:
                ktrail = trail_k(mfe_atr)
                dyn = (hi - ktrail*ak) if side == 1 else (lo + ktrail*ak)
                stop = better_stop(side, stop, dyn)

            if exit_variant == 'ATR_MFE_LOCK' and mfe_atr >= 0.75:
                locked = mfe * lock_fraction(mfe_atr)
                mfe_stop = entry + side * locked
                stop = better_stop(side, stop, mfe_stop)

            # Conservative OHLC ordering: adverse excursion first.
            stop_hit = (lo <= stop) if side == 1 else (hi >= stop)
            if stop_hit:
                exit_px, exit_i, reason = stop, k, 'STOP/TRAIL'
                break

            if exit_variant == 'FIXED_TP':
                tp_hit = (hi >= fixed_tp) if side == 1 else (lo <= fixed_tp)
                if tp_hit:
                    exit_px, exit_i, reason = fixed_tp, k, 'TP'
                    break

            # ECLIPSE/FUSION keep delayed-stop confirmation only before adaptive stop reaches BE.
            if mode in ('ECLIPSE','FUSION') and stop == initial_sl:
                crossed = (lo <= initial_sl) if side == 1 else (hi >= initial_sl)
                if crossed:
                    # One-bar close confirmation approximation for discovery.
                    confirm_bad = (close < initial_sl) if side == 1 else (close > initial_sl)
                    if confirm_bad:
                        exit_px, exit_i, reason = close, k, 'DELAYED_SL'
                        break

        if exit_px is None:
            exit_i = end
            exit_px = float(df.close.iloc[end])

        xc, cc2 = base.draw_cost(rng, profile, a)
        for k,v in cc2.items(): costs[k] += v
        exit_fill = exit_px - xc * side
        gross = (exit_fill-entry) * side
        net = gross - profile.commission_rt
        costs['commission'] += profile.commission_rt
        equity += net
        peak = max(peak, equity)
        maxdd = max(maxdd, (peak-equity)/peak*100)
        trades.append(dict(mode=mode, exit_variant=exit_variant, profile=profile.name, seed=seed,
                           entry_i=i, exit_i=exit_i, side=side, net=net, mfe=mfe,
                           mfe_atr=mfe/a if a>0 else 0.0, reason=reason, signal=s['reason']))
        next_free = exit_i + 1

    arr = np.array([x['net'] for x in trades], float)
    wins = arr[arr>0]; losses = arr[arr<0]
    pf = float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
    wr = float((arr>0).mean()*100) if len(arr) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    aw_al = avg_win/avg_loss if avg_loss>0 else None
    ret = (equity/10000-1)*100
    days = max(1, (df.datetime.iloc[-1]-df.datetime.iloc[0]).days)
    monthly = ((equity/10000)**(21/days)-1)*100 if equity>0 else -100.0
    return dict(mode=mode, exit_variant=exit_variant, profile=profile.name, seed=seed,
                N=len(arr), WR_pct=wr, PF=pf, AvgWin=avg_win, AvgLoss=avg_loss, AW_AL=aw_al,
                MaxDD_pct=maxdd, Return_pct=ret, Monthly21_pct=monthly,
                EndingBalance=equity, NetProfit=equity-10000), trades


def summarize(rows):
    metrics=['N','WR_pct','PF','AvgWin','AvgLoss','AW_AL','MaxDD_pct','Return_pct','Monthly21_pct','EndingBalance','NetProfit']
    out={}
    for key in metrics:
        vals=np.array([r[key] for r in rows if r[key] is not None], float)
        vals=vals[np.isfinite(vals)]
        out[key]=dict(median=float(np.median(vals)) if len(vals) else None,
                      p10=float(np.percentile(vals,10)) if len(vals) else None,
                      p90=float(np.percentile(vals,90)) if len(vals) else None)
    return out


def main():
    import nautilus_trader
    df=pd.read_csv(DATA)
    df.columns=[c.lower() for c in df.columns]
    df['datetime']=pd.to_datetime(df['datetime'], utc=True)
    df=df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    df=base.add_features(df)

    all_runs=[]; all_trades=[]; results={}
    for mode in MODES:
        signals=base.build_signals(df, mode)
        results[mode]={'raw_candidate_signals':len(signals),'exit_variants':{}}
        for ev in EXIT_VARIANTS:
            results[mode]['exit_variants'][ev]={}
            for p in base.PROFILES:
                seeds=[260831] if p.name=='IDEAL' else list(range(260831,260881))
                rows=[]
                for seed in seeds:
                    r,t=run_once(df, signals, mode, p, seed, ev)
                    rows.append(r); all_runs.append(r); all_trades.extend(t)
                results[mode]['exit_variants'][ev][p.name]=summarize(rows)

    runs=pd.DataFrame(all_runs)
    ideal=runs[runs.profile=='IDEAL'].copy()
    gate=[]
    for _,r in ideal.iterrows():
        gate.append(dict(mode=r['mode'],exit_variant=r['exit_variant'],PF=r['PF'],WR_pct=r['WR_pct'],
                         AW_AL=r['AW_AL'],MaxDD_pct=r['MaxDD_pct'],Monthly21_pct=r['Monthly21_pct'],
                         IDEAL_PF_1_5_PASS=bool(r['PF']>=1.5)))
    pd.DataFrame(gate).to_csv(OUT/'ideal_gate.csv',index=False)
    runs.to_csv(OUT/'all_runs.csv',index=False)
    pd.DataFrame(all_trades).to_csv(OUT/'trades.csv',index=False)

    payload=dict(engine='NautilusTrader runtime + CHOCHOMARU clean-room exit discovery v2',
                 nautilus_version=getattr(nautilus_trader,'__version__','unknown'),
                 status='DISCOVERY_ONLY_NOT_RAW_TICK_VALIDATION',
                 hypothesis='RR shortage -> dynamic ATR trail / MFE profit lock',
                 data=str(DATA), rows=len(df), start=str(df.datetime.iloc[0]), end=str(df.datetime.iloc[-1]),
                 exit_params=V2, results=results,
                 promotion_gate={'IDEAL_PF_min':1.5,'NORMAL_PF_min':1.2},
                 limitations=['M1 mid-quote OHLC, not raw Bid/Ask ticks','conservative adverse-first intrabar ordering','noise profiles are proxies','entry logic remains clean-room hypothesis'])
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(payload,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
