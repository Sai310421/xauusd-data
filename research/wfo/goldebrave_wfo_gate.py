from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def profile_hash(profile: dict) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def median(xs):
    return float(statistics.median(xs)) if xs else 0.0


def evaluate(config: dict, records: list[dict]) -> dict:
    gate = config['oos_gate']
    min_windows = config['rolling_windows']['minimum_oos_windows']
    out = {'schema': config['schema'], 'profiles': {}, 'selection_basis': 'OOS_ONLY'}

    for name, frozen in config['frozen_profiles'].items():
        ph = profile_hash(frozen)
        rows = [r for r in records if r.get('profile') == name]
        is_rows = [r for r in rows if str(r.get('split', '')).upper() == 'IS']
        oos = [r for r in rows if str(r.get('split', '')).upper() == 'OOS']

        # Fail closed if a result claims a different parameter hash.
        mismatches = [r for r in rows if r.get('params_hash') not in (None, ph)]
        nets = [float(r.get('Net', 0.0)) for r in oos]
        pfs = [float(r.get('PF', 0.0)) for r in oos]
        dds = [float(r.get('max_floating_dd_pct', r.get('MaxDD_pct_virtual', 999.0))) for r in oos]
        nret = [float(r.get('n_retention', 1.0)) for r in oos]
        positive_ratio = (sum(x > 0 for x in nets) / len(nets)) if nets else 0.0

        is_net = sum(float(r.get('Net', 0.0)) for r in is_rows)
        oos_net = sum(nets)
        wfe = (100.0 * oos_net / is_net) if is_net > 0 else None

        checks = {
            'enough_oos_windows': len(oos) >= min_windows,
            'positive_window_ratio': positive_ratio >= gate['positive_window_ratio_min'],
            'median_pf': median(pfs) >= gate['median_pf_min'],
            'median_floating_dd': median(dds) <= gate['median_floating_dd_pct_max'],
            'worst_floating_dd': (max(dds) if dds else 999.0) <= gate['worst_floating_dd_pct_max'],
            'median_n_retention': median(nret) >= gate['median_n_retention_min'],
            'wfe': (wfe is not None and wfe >= gate['wfe_pct_min']),
            'frozen_parameter_hash': len(mismatches) == 0,
        }

        out['profiles'][name] = {
            'params_hash': ph,
            'oos_windows': len(oos),
            'is_windows': len(is_rows),
            'oos_positive_ratio': positive_ratio,
            'oos_net_sum': oos_net,
            'median_pf': median(pfs),
            'median_floating_dd_pct': median(dds),
            'worst_floating_dd_pct': max(dds) if dds else None,
            'median_n_retention': median(nret),
            'wfe_pct': wfe,
            'checks': checks,
            'pass': all(checks.values()),
        }

    passed = [k for k, v in out['profiles'].items() if v['pass']]
    out['adoptable_profiles'] = passed
    out['pass'] = bool(passed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='research/wfo/goldebrave_wfo_config.json')
    ap.add_argument('--records', required=True, help='JSON array of IS/OOS window result records')
    ap.add_argument('--out', default='research/results/wfo/wfo_gate_summary.json')
    ap.add_argument('--fail-on-gate', action='store_true')
    args = ap.parse_args()

    config = load_json(args.config)
    records = load_json(args.records)
    if not isinstance(records, list):
        raise SystemExit('records must be a JSON array')

    summary = evaluate(config, records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.fail_on_gate and not summary['pass']:
        sys.exit(2)


if __name__ == '__main__':
    main()
