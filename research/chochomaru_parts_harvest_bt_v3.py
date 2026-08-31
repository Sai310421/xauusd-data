from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from chochomaru_cleanroom_discovery_bt_v1 import (
    DATA, PARAMS, PROFILES, MODES, add_features, build_signals, draw_cost,
)

OUT = Path('results/chochomaru-parts-harvest-v3')
OUT.mkdir(parents=True, exist_ok=True)

# Clean-room concept harvesting only: no copied strategy source from external bots.
# Parts under test are generic concepts observed across the supplied sources:
# regime classification, session gating, liquidity-sweep confirmation,
# momentum/pullback confirmation, and delayed ATR/MFE exit protection.

PARTS = [
    'BASE_DELAYED_EXIT',
    'REGIME',
    'SESSION',
    'LIQUIDITY',
    'REGIME_SESSION',
    'REGIME_LIQUIDITY',
    'FULL_STACK',
]

EXIT_VARIANTS = [
    'DELAYED_ATR_10',
    'DELAYED_ATR_15',
    'DELAYED_ATR_20',
    'MFE_ADAPTIVE',
]


def add_harvest_features(df: pd.DataFrame) -> pd.DataFrame:
    x = add_features(df)
    # ADX-style trend strength.
    up = x.high.diff()
    dn = -x.low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=x.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=x.index)
    pc = x.close.shift(1)
    tr = pd.concat([(x.high-x.low).abs(), (x.high-pc).abs(), (x.low-pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    x['adx'] = dx.rolling(14).mean().fillna(0.0)
    x['atr_pct'] = atr14.rolling(100).rank(pct=True).fillna(0.5)

    # Session tags in UTC: London/NY and overlap get highest priority.
    h = x.datetime.dt.hour
    x['session_good'] = ((h >= 7) & (h < 21)).astype(int)
    x['session_overlap'] = ((h >= 13) & (h < 16)).astype(int)

    # Generic liquidity sweep: take prior swing extreme and close back through it.
    prior_hi = x.high.shift(1).rolling(12).max()
    prior_lo = x.low.shift(1).rolling(12).min()
    x['sweep_buy'] = ((x.low < prior_lo) & (x.close > prior_lo) & (x.close > x.open)).astype(int)
    x['sweep_sell'] = ((x.high > prior_hi) & (x.close < prior_hi) & (x.close < x.open)).astype(int)

    # Momentum quality / pullback geometry.
    x['ema_gap_atr'] = ((x.ef-x.em).abs() / x.atr.replace(0, np.nan)).fillna(0.0)
    x['ema_slope_atr'] = ((x.ef-x.ef.shift(3)) / (3*x.atr.replace(0, np.nan))).fillna(0.0)
    return x


def keep_signal(df: pd.DataFrame, s: dict, part: str) -> bool:
    i = s['entry_i']
    r = df.iloc[i-1]
    side = s['side']

    # Avoid dead/noisy extremes: keep either trend or range-breakout-compatible regimes.
    regime_ok = (18.0 <= r.adx <= 55.0) and (0.15 <= r.atr_pct <= 0.90)
    session_ok = bool(r.session_good)
    sweep_ok = bool(r.sweep_buy if side == 1 else r.sweep_sell)
    momentum_ok = (r.ema_gap_atr >= 0.05) and ((r.ema_slope_atr > 0) if side == 1 else (r.ema_slope_atr < 0))

    if part == 'BASE_DELAYED_EXIT': return True
    if part == 'REGIME': return regime_ok
    if part == 'SESSION': return session_ok
    if part == 'LIQUIDITY': return sweep_ok
    if part == 'REGIME_SESSION': return regime_ok and session_ok
    if part == 'REGIME_LIQUIDITY': return regime_ok and sweep_ok
    if part == 'FULL_STACK': return regime_ok and session_ok and momentum_ok and (sweep_ok or r.session_overlap == 1)
    return True


def trail_distance(exit_variant: str, mfe_atr: float, atr_now: float) -> float:
    if exit_variant == 'DELAYED_ATR_10':
        return 0.95 * atr_now
    if exit_variant == 'DELAYED_ATR_15':
        return 1.00 * atr_now
    if exit_variant == 'DELAYED_ATR_20':
        return 1.10 * atr_now
    # Adaptive: loose while trend is young, progressively protect only after extension.
    if mfe_atr < 1.5: return 1.20 * atr_now
    if mfe_atr < 2.5: return 0.95 * atr_now
    if mfe_atr < 4.0: return 0.75 * atr_now
    return 0.60 * atr_now


def trail_start(exit_variant: str) -> float:
    return {
        'DELAYED_ATR_10': 1.0,
        'DELAYED_ATR_15': 1.5,
        'DELAYED_ATR_20': 2.0,
        'MFE_ADAPTIVE': 1.25,
    }[exit_variant]


def run_once(df, signals, mode, profile, seed, part, exit_variant):
    rng = np.random.default_rng(seed)
    equity = 10000.0
    peak = equity
    maxdd = 0.0
    next_free = 0
    trades = []

    for s in signals:
        if not keep_signal(df, s, part):
            continue
        i, side, a = s['entry_i'], s['side'], s['atr']
        if i < next_free or rng.random() < profile.reject_prob:
            continue

        ec, _ = draw_cost(rng, profile, a)
        entry = float(df.open.iloc[i]) + ec * side
        vsl = entry - side * PARAMS['sl_atr'] * a
        hard = entry - side * PARAMS['delayed_stop_hardcap_atr'] * a
        broker_stop = hard if mode in ('ECLIPSE','FUSION') else vsl

        best = entry
        trail = None
        armed = 0
        exit_px = None
        exit_i = None
        reason = 'TIME'
        start_mfe = trail_start(exit_variant)
        end = min(len(df)-1, i + PARAMS['max_hold_bars'])

        for k in range(i, end+1):
            hi = float(df.high.iloc[k]); lo = float(df.low.iloc[k]); close = float(df.close.iloc[k])
            ak = float(df.atr.iloc[k]) if np.isfinite(df.atr.iloc[k]) else a
            if side == 1:
                best = max(best, hi)
                mfe = best-entry
            else:
                best = min(best, lo)
                mfe = entry-best
            mfe_atr = mfe / max(a, 1e-9)

            # Do not choke early winners. No fixed TP in these variants.
            if mfe_atr >= start_mfe:
                dist = trail_distance(exit_variant, mfe_atr, ak)
                candidate = best - side * dist
                # MFE floor: lock progressively only after extension.
                if exit_variant == 'MFE_ADAPTIVE':
                    lock_frac = 0.0 if mfe_atr < 1.5 else (0.25 if mfe_atr < 2.5 else (0.45 if mfe_atr < 4.0 else 0.60))
                    floor = entry + side * (mfe * lock_frac)
                    candidate = max(candidate, floor) if side == 1 else min(candidate, floor)
                trail = candidate if trail is None else (max(trail, candidate) if side == 1 else min(trail, candidate))

            active_stop = broker_stop
            if trail is not None:
                active_stop = max(active_stop, trail) if side == 1 else min(active_stop, trail)

            stop_hit = (lo <= active_stop) if side == 1 else (hi >= active_stop)
            if stop_hit:
                exit_px = active_stop; exit_i = k; reason = 'STOP/TRAIL'; break

            if mode in ('ECLIPSE','FUSION'):
                crossed = (lo <= vsl) if side == 1 else (hi >= vsl)
                reclaimed = (close >= vsl + PARAMS['delayed_stop_reclaim_atr']*ak) if side == 1 else (close <= vsl - PARAMS['delayed_stop_reclaim_atr']*ak)
                if crossed:
                    armed += 1
                    if reclaimed:
                        armed = 0
                    elif armed >= PARAMS['delayed_stop_confirm_bars']:
                        exit_px = close; exit_i = k; reason = 'DELAYED_SL'; break

        if exit_px is None:
            exit_i = end
            exit_px = float(df.close.iloc[end])

        xc, _ = draw_cost(rng, profile, a)
        exit_fill = exit_px - xc * side
        net = (exit_fill-entry) * side - profile.commission_rt
        equity += net
        peak = max(peak, equity)
        maxdd = max(maxdd, (peak-equity)/peak*100)
        trades.append(dict(part=part, exit_variant=exit_variant, mode=mode, profile=profile.name,
                           seed=seed, net=net, entry_i=i, exit_i=exit_i, side=side,
                           reason=reason, signal=s['reason']))
        next_free = exit_i + 1

    a = np.array([t['net'] for t in trades], float)
    wins = a[a > 0]; losses = a[a < 0]
    pf = float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum() != 0 else (float('inf') if len(wins) else 0.0)
    wr = float((a > 0).mean()*100) if len(a) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    aw_al = avg_win/avg_loss if avg_loss else 0.0
    ret = (equity/10000 - 1) * 100
    days = max(1, (df.datetime.iloc[-1]-df.datetime.iloc[0]).days)
    monthly = ((equity/10000)**(21/days)-1)*100 if equity > 0 else -100.0
    return dict(part=part,exit_variant=exit_variant,mode=mode,profile=profile.name,seed=seed,
                N=len(a),WR_pct=wr,PF=pf,AvgWin=avg_win,AvgLoss=avg_loss,AW_AL=aw_al,
                MaxDD_pct=maxdd,Return_pct=ret,Monthly21_pct=monthly,
                EndingBalance=equity,NetProfit=equity-10000), trades


def summarize(rows):
    keys = ['N','WR_pct','PF','AvgWin','AvgLoss','AW_AL','MaxDD_pct','Return_pct','Monthly21_pct','EndingBalance','NetProfit']
    out = {}
    for k in keys:
        x = np.array([r[k] for r in rows], float)
        x = x[np.isfinite(x)]
        out[k] = dict(median=float(np.median(x)) if len(x) else None,
                      p10=float(np.percentile(x,10)) if len(x) else None,
                      p90=float(np.percentile(x,90)) if len(x) else None)
    return out


def main():
    df = pd.read_csv(DATA)
    df.columns = [c.lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    df = add_harvest_features(df)

    all_runs = []; all_trades = []; report = {}
    for mode in MODES:
        sig = build_signals(df, mode)
        for part in PARTS:
            for ev in EXIT_VARIANTS:
                key = f'{mode}|{part}|{ev}'
                report[key] = {}
                for p in PROFILES:
                    seeds = [260831] if p.name == 'IDEAL' else list(range(260831,260851))
                    rows = []
                    for seed in seeds:
                        r, t = run_once(df, sig, mode, p, seed, part, ev)
                        rows.append(r); all_runs.append(r); all_trades.extend(t)
                    report[key][p.name] = summarize(rows)

    runs = pd.DataFrame(all_runs)
    runs.to_csv(OUT/'all_runs.csv', index=False)
    pd.DataFrame(all_trades).to_csv(OUT/'trades.csv', index=False)

    # Ranking emphasizes NORMAL profitability and penalizes DD / tiny samples.
    med = runs.groupby(['mode','part','exit_variant','profile'], as_index=False).median(numeric_only=True)
    normal = med[med.profile=='NORMAL'].copy()
    normal['score'] = normal.PF + 0.02*normal.Monthly21_pct - 0.03*normal.MaxDD_pct
    normal.loc[normal.N < 80, 'score'] -= 1.0
    normal = normal.sort_values('score', ascending=False)
    normal.to_csv(OUT/'normal_ranking.csv', index=False)

    payload = {
        'status': 'DISCOVERY_ONLY_NOT_RAW_TICK_VALIDATION',
        'purpose': 'ablation test of generic harvested components for profitability improvement',
        'parts': PARTS,
        'exit_variants': EXIT_VARIANTS,
        'results': report,
        'promotion_gate': {'IDEAL_PF': 1.5, 'NORMAL_PF': 1.2, 'NORMAL_N_min': 80},
        'limitations': ['M1 mid-quote OHLC discovery only', 'not Raw Bid/Ask QuoteTick', 'generic concept reimplementation; no copied external strategy source'],
    }
    (OUT/'summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(normal.head(30).to_string(index=False))

if __name__ == '__main__':
    main()
