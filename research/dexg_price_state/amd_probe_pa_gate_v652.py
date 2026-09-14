from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import amd_probe_risk_split_v65 as v65

# v6.5.2: causal micro Price-Action gate on the proven ARM -> inverse Probe lane.
# All PA features are computed ONLY from Raw Bid/Ask ticks at or before ARM.
# They are diagnostic filters, not future-completed candles and not OHLC fallback data.

PA_GATES = [
    'ALL',
    'PA_CLOSE',
    'PA_RECLAIM',
    'PA_ACCEL',
    'PA_SCORE2',
    'PA_SCORE3',
]
TPS = [0.30, 0.50, 0.75, 1.00, 1.25, 1.50]
SLS = [0.10, 0.15, 0.20, 0.30]
HOLDS = [3, 5, 10, 15, 30]
VARIANTS = ['SINGLE', 'SPLIT_A', 'SPLIT_B', 'SPLIT_C']
RISK_MULTS = [1.0, 1.25, 1.5, 2.0]


def _window_indices(ticks, idx, seconds):
    start = int(ticks[idx].ts_event) - int(seconds * 1e9)
    j = idx
    while j > 0 and int(ticks[j - 1].ts_event) >= start:
        j -= 1
    return list(range(j, idx + 1))


def _subrange(ticks, idx, start_s, end_s):
    t0 = int(ticks[idx].ts_event)
    lo = t0 - int(start_s * 1e9)
    hi = t0 - int(end_s * 1e9)
    out = []
    j = idx
    while j >= 0 and int(ticks[j].ts_event) >= lo:
        ts = int(ticks[j].ts_event)
        if ts <= hi:
            out.append(j)
        j -= 1
    out.reverse()
    return out


def _range_mid(ticks, idxs):
    if not idxs:
        return 0.0
    vals = [v65.v6.mid(ticks[i]) for i in idxs]
    return max(vals) - min(vals)


def _move(ticks, idxs, side):
    if len(idxs) < 2:
        return 0.0
    return (v65.v6.mid(ticks[idxs[-1]]) - v65.v6.mid(ticks[idxs[0]])) * side


def pa_features(ticks, arm_idx, side):
    idxs = _window_indices(ticks, arm_idx, 3.0)
    vals = [v65.v6.mid(ticks[i]) for i in idxs]
    if len(vals) < 4:
        return {
            'close_ok': False, 'reclaim_ok': False, 'accel_ok': False,
            'expand_ok': False, 'body_ok': False, 'score': 0,
            'body_ratio': 0.0, 'close_location': 0.5,
        }

    first, last = vals[0], vals[-1]
    lo, hi = min(vals), max(vals)
    rng = max(hi - lo, 1e-12)
    body_side = (last - first) * side
    body_ratio = abs(last - first) / rng
    close_location = (last - lo) / rng if side > 0 else (hi - last) / rng

    # Failed adverse probe -> reclaim: price first explores against Probe side,
    # then recovers at least $0.04 from the adverse extreme by ARM.
    adverse_exc = (first - lo) if side > 0 else (hi - first)
    reclaim_amt = (last - lo) if side > 0 else (hi - last)
    reclaim_ok = adverse_exc >= 0.03 and reclaim_amt >= 0.04 and close_location >= 0.55

    # Raw-tick compression -> expansion: last 1s range exceeds preceding 2s normalized range.
    last1 = _subrange(ticks, arm_idx, 1.0, 0.0)
    prev2 = _subrange(ticks, arm_idx, 3.0, 1.0)
    r1 = _range_mid(ticks, last1)
    r2 = _range_mid(ticks, prev2)
    expand_ok = r1 >= 0.04 and r1 >= 0.65 * max(r2, 1e-12)

    # Tick acceleration in Probe direction: most recent 0.5s move exceeds prior 0.5s.
    recent = _subrange(ticks, arm_idx, 0.5, 0.0)
    prior = _subrange(ticks, arm_idx, 1.0, 0.5)
    m_recent = _move(ticks, recent, side)
    m_prior = _move(ticks, prior, side)
    accel_ok = m_recent >= 0.02 and m_recent > m_prior + 0.01

    close_ok = close_location >= 0.62
    body_ok = body_side >= 0.02 and body_ratio >= 0.20
    score = sum([close_ok, reclaim_ok, accel_ok, expand_ok, body_ok])
    return {
        'close_ok': bool(close_ok), 'reclaim_ok': bool(reclaim_ok),
        'accel_ok': bool(accel_ok), 'expand_ok': bool(expand_ok),
        'body_ok': bool(body_ok), 'score': int(score),
        'body_ratio': body_ratio, 'close_location': close_location,
        'reclaim_amt': reclaim_amt, 'last1_range': r1, 'prev2_range': r2,
        'recent_move': m_recent, 'prior_move': m_prior,
    }


def gate_ok(name, f):
    if name == 'ALL': return True
    if name == 'PA_CLOSE': return f['close_ok']
    if name == 'PA_RECLAIM': return f['reclaim_ok']
    if name == 'PA_ACCEL': return f['accel_ok']
    if name == 'PA_SCORE2': return f['score'] >= 2
    if name == 'PA_SCORE3': return f['score'] >= 3
    return False


def rank_key(r):
    return (float(r.get('expectancy_usd') or 0.0), float(r.get('pf') or 0.0), -float(r.get('max_dd_pct') or 999.0))


def compact(r):
    keys = ['gate','gate_signals','scenario','variant','tp','sl','hold_s','risk_mult',
            'signals','filled_tranches','avg_tranches_per_signal','wr','pf','expectancy_usd',
            'net_pnl_usd','return_pct','max_dd_pct','max_dd_usd','rf','monthly21_pct',
            'ending_equity','scale_allowed','selection_eligible']
    return {k:r.get(k) for k in keys}


def run(catalog, max_ticks=3_000_000):
    ticks = v65.v6.load_ticks(catalog, max_ticks)
    b15 = v65.v6.bars_n(ticks, 15)
    base_events = v65.arm_events(ticks, b15)
    dates = {datetime.fromtimestamp(int(t.ts_event)/1e9, tz=timezone.utc).date() for t in ticks}
    n_days = max(1, sum(1 for d in dates if d.weekday() < 5))

    feats = [(idx, side, pa_features(ticks, idx, side)) for idx, side in base_events]
    gate_events = {g:[(idx,side) for idx,side,f in feats if gate_ok(g,f)] for g in PA_GATES}
    gate_counts = {g:len(v) for g,v in gate_events.items()}

    rows = []
    # Risk is evaluated only after a positive 1.0x edge exists. Generate base rows first.
    for gate in PA_GATES:
        events = gate_events[gate]
        if len(events) < 50:
            continue
        for sc in v65.SCENARIOS:
            for var in VARIANTS:
                for tp in TPS:
                    for sl in SLS:
                        for hold in HOLDS:
                            pnls=[]; fills=[]
                            for arm_idx,side in events:
                                p,f,_,_,_=v65.simulate_signal(ticks,arm_idx,side,var,tp,sl,hold,sc)
                                pnls.append(p); fills.append(f)
                            m=v65.metrics(pnls,fills,1.0,n_days)
                            m.update({'gate':gate,'gate_signals':len(events),'scenario':sc['name'],
                                      'variant':var,'tp':tp,'sl':sl,'hold_s':hold,
                                      'scale_allowed':False,'selection_eligible':True})
                            rows.append(m)

    positive = [r for r in rows if r['expectancy_usd'] > 0 and (r['pf'] or 0) > 1.0]
    positive.sort(key=rank_key, reverse=True)
    near = sorted(rows, key=rank_key, reverse=True)

    # Adaptive risk: rescale only the top unique positive base configs, preserving their path DD logic.
    scaled=[]
    for base in positive[:20]:
        dd=base['max_dd_pct']
        allowed=[1.0,1.25,1.5,2.0] if dd<=2.0 else ([1.0,1.25] if dd<=3.5 else [1.0])
        base['scale_allowed']=len(allowed)>1
        for rm in allowed:
            s=dict(base)
            if rm != 1.0:
                # Linear lot-risk sensitivity from the measured base row. This does not alter N/WR/PF.
                s['risk_mult']=rm
                s['expectancy_usd']=base['expectancy_usd']*rm
                s['net_pnl_usd']=base['net_pnl_usd']*rm
                s['return_pct']=base['return_pct']*rm
                s['max_dd_pct']=base['max_dd_pct']*rm
                s['max_dd_usd']=base['max_dd_usd']*rm
                s['monthly21_pct']=base['monthly21_pct']*rm
                s['ending_equity']=v65.START_EQUITY+s['net_pnl_usd']
                s['rf']=s['net_pnl_usd']/s['max_dd_usd'] if s['max_dd_usd']>0 else None
            scaled.append(s)
    scaled.sort(key=lambda r:(r['return_pct']/max(r['max_dd_pct'],1e-9), r['return_pct']), reverse=True)

    return {
        'status':'CAUSAL_V652_RAW_TICK_PA_GATED_PROBE_EXECUTION_DIAGNOSTIC',
        'ticks':len(ticks),'bars15':len(b15),'base_signals':len(base_events),
        'business_days_observed':n_days,'start_equity':v65.START_EQUITY,'base_lot':v65.BASE_LOT,
        'gate_counts':gate_counts,
        'pa_definition':'Causal 3s raw-tick micro-PA at ARM: close location, adverse-probe reclaim, compression-expansion, probe-side acceleration, body/range. No future candle data.',
        'base_config_count':len(rows),
        'positive_pf_count':len(positive),
        'accepted_top20':[compact(r) for r in positive[:20]],
        'near_miss_top20':[compact(r) for r in near[:20]],
        'adaptive_risk_top20':[compact(r) for r in scaled[:20]],
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--max-ticks',type=int,default=3_000_000); a=ap.parse_args()
    res=run(a.catalog,a.max_ticks)
    d=Path('results/dexg-amd15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/'AMD_PROBE_PA_GATE_V652.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps({'status':res['status'],'base_signals':res['base_signals'],'gate_counts':res['gate_counts'],'positive_pf_count':res['positive_pf_count'],'accepted_top5':res['accepted_top20'][:5],'adaptive_risk_top5':res['adaptive_risk_top20'][:5]},indent=2))

if __name__=='__main__': main()
