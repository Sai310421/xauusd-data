from __future__ import annotations

import argparse
import json
from pathlib import Path

import amd_probe_risk_split_v65 as v65

# v6.5.1 correction:
# - keep the same causal ARM->inverse Probe signal and broker-cost model
# - widen gross payoff search so execution cost is not compared only with tiny targets
# - never hide all information behind a strict profitability gate
# - report gate failure counts and best near-miss configurations separately from accepted configs
v65.TPS = [0.30, 0.50, 0.75, 1.00, 1.25, 1.50]
v65.SLS = [0.10, 0.15, 0.20, 0.30]
v65.HOLDS = [3, 5, 10, 15, 30]


def finite_pf(row):
    pf = row.get('pf')
    return float(pf) if pf is not None else -1.0


def rank_score(row):
    # Diagnostic ranking only; this does NOT make a losing row eligible.
    # Prefer positive expectancy, then PF, then lower DD, then RF.
    exp = float(row.get('expectancy_usd') or 0.0)
    pf = finite_pf(row)
    dd = max(float(row.get('max_dd_pct') or 0.0), 1e-9)
    rf = float(row.get('rf') or -999.0)
    return (exp, pf, -dd, rf)


def compact(row):
    keys = [
        'scenario', 'variant', 'tp', 'sl', 'hold_s', 'risk_mult',
        'signals', 'filled_tranches', 'avg_tranches_per_signal',
        'wr', 'pf', 'expectancy_usd', 'net_pnl_usd', 'return_pct',
        'max_dd_pct', 'max_dd_usd', 'rf', 'monthly21_pct',
        'ending_equity', 'scale_allowed', 'selection_eligible',
    ]
    return {k: row.get(k) for k in keys}


def best_group(rows, key):
    out = {}
    for row in rows:
        value = str(row.get(key))
        current = out.get(value)
        if current is None or rank_score(row) > rank_score(current):
            out[value] = row
    return {k: compact(v) for k, v in out.items()}


def diagnose(result):
    rows = result['all_results']
    base_rows = [r for r in rows if abs(float(r.get('risk_mult', 0)) - 1.0) < 1e-12]

    positive_exp = [r for r in base_rows if float(r.get('expectancy_usd') or 0.0) > 0.0]
    pf_gt_1 = [r for r in base_rows if finite_pf(r) > 1.0]
    strict_base = [r for r in base_rows if float(r.get('expectancy_usd') or 0.0) > 0.0 and finite_pf(r) > 1.0]

    strict_all = [
        r for r in rows
        if r.get('selection_eligible')
        and float(r.get('expectancy_usd') or 0.0) > 0.0
        and finite_pf(r) > 1.0
    ]
    strict_all.sort(key=lambda r: (float(r.get('return_pct') or 0.0) / max(float(r.get('max_dd_pct') or 0.0), 1e-9)), reverse=True)

    near = sorted(base_rows, key=rank_score, reverse=True)

    dd_buckets = {
        'dd_le_2pct': sum(1 for r in base_rows if float(r.get('max_dd_pct') or 0.0) <= 2.0),
        'dd_le_3_5pct': sum(1 for r in base_rows if float(r.get('max_dd_pct') or 0.0) <= 3.5),
        'dd_le_5pct': sum(1 for r in base_rows if float(r.get('max_dd_pct') or 0.0) <= 5.0),
    }

    result['status'] = 'CAUSAL_V651_PROBE_3SPLIT_GATE_DIAGNOSTIC_AND_WIDER_PAYOFF_SEARCH'
    result['search_grid'] = {
        'tp': v65.TPS,
        'sl': v65.SLS,
        'hold_s': v65.HOLDS,
        'risk_mult': v65.RISK_MULTS,
        'variants': ['SINGLE', 'SPLIT_A', 'SPLIT_B', 'SPLIT_C'],
        'scenarios': [s['name'] for s in v65.SCENARIOS],
    }
    result['gate_diagnostics'] = {
        'base_config_count': len(base_rows),
        'positive_expectancy_count': len(positive_exp),
        'pf_gt_1_count': len(pf_gt_1),
        'strict_positive_pf_count': len(strict_base),
        'strict_all_risk_rows_count': len(strict_all),
        **dd_buckets,
        'meaning': 'A zero strict count means no configuration is accepted. near_miss_top20 is diagnostic only and must not be reported as profitable.',
    }
    result['accepted_top20'] = [compact(r) for r in strict_all[:20]]
    result['near_miss_top20'] = [compact(r) for r in near[:20]]
    result['best_by_variant_base_risk'] = best_group(base_rows, 'variant')
    result['best_by_scenario_base_risk'] = best_group(base_rows, 'scenario')

    # Adaptive risk is permitted only after a base-risk configuration has positive expectancy and PF>1.
    if strict_base:
        best_base = max(strict_base, key=rank_score)
        dd = float(best_base.get('max_dd_pct') or 0.0)
        if dd <= 2.0:
            allowed_mults = [1.0, 1.25, 1.5, 2.0]
        elif dd <= 3.5:
            allowed_mults = [1.0, 1.25]
        else:
            allowed_mults = [1.0]
        result['adaptive_risk_decision'] = {
            'base_candidate': compact(best_base),
            'allowed_risk_mults': allowed_mults,
            'reason': 'Risk is increased only after positive net expectancy and PF>1 are demonstrated at 1.0x.',
        }
    else:
        result['adaptive_risk_decision'] = {
            'base_candidate': None,
            'allowed_risk_mults': [1.0],
            'reason': 'No positive base edge; risk-up is blocked. Diagnose timing/PA/exit before leverage.',
        }

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--max-ticks', type=int, default=3_000_000)
    args = ap.parse_args()

    result = v65.run(args.catalog, args.max_ticks)
    result = diagnose(result)

    out_dir = Path('results/dexg-amd15') / args.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'AMD_PROBE_RISK_SPLIT_V651.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')

    print(json.dumps({
        'status': result['status'],
        'signals': result['signals'],
        'business_days_observed': result['business_days_observed'],
        'gate_diagnostics': result['gate_diagnostics'],
        'accepted_top20': result['accepted_top20'],
        'near_miss_top5': result['near_miss_top20'][:5],
        'adaptive_risk_decision': result['adaptive_risk_decision'],
    }, indent=2))


if __name__ == '__main__':
    main()
