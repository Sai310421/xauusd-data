from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def get_metrics(obj):
    if 'portfolio_realized_close_ordered' in obj:
        return obj['portfolio_realized_close_ordered']
    if 'metrics' in obj:
        return obj['metrics']
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--edge', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-n-drop-pct', type=float, default=20.0)
    ap.add_argument('--max-dd-worsen-pct', type=float, default=3.0)
    args = ap.parse_args()

    b0 = load(args.base)
    e0 = load(args.edge)
    b = get_metrics(b0)
    e = get_metrics(e0)

    def num(d, *names):
        for n in names:
            if n in d and d[n] is not None:
                return float(d[n])
        return None

    BN, EN = num(b,'N'), num(e,'N')
    BPF, EPF = num(b,'PF'), num(e,'PF')
    BRF, ERF = num(b,'RF'), num(e,'RF')
    BDD, EDD = num(b,'MaxDD_pct','DD'), num(e,'MaxDD_pct','DD')
    BRET, ERET = num(b,'Monthly21_pct','Monthly21'), num(e,'Monthly21_pct','Monthly21')

    n_drop = None if BN in (None,0) or EN is None else (BN-EN)/BN*100.0
    dd_worsen = None if BDD in (None,0) or EDD is None else (EDD-BDD)/BDD*100.0

    improves = []
    if BPF is not None and EPF is not None: improves.append(EPF > BPF)
    if BRF is not None and ERF is not None: improves.append(ERF > BRF)
    if BRET is not None and ERET is not None: improves.append(ERET > BRET)

    gates = {
        'n_preserved': n_drop is None or n_drop <= args.max_n_drop_pct,
        'dd_not_materially_worse': dd_worsen is None or dd_worsen <= args.max_dd_worsen_pct,
        'one_quality_or_return_metric_improves': any(improves) if improves else False,
        'raw_bidask_evidence': b0.get('ohlc_resample_used') is False and e0.get('ohlc_resample_used') is False,
        'same_data_kind': b0.get('data_kind') == e0.get('data_kind'),
    }
    passed = all(gates.values())
    rep = {
        'decision': 'PROMOTE_NEXT_GATE' if passed else 'HOLD_OR_REJECT',
        'gates': gates,
        'deltas': {
            'N_pct_drop': n_drop,
            'PF': None if BPF is None or EPF is None else EPF-BPF,
            'RF': None if BRF is None or ERF is None else ERF-BRF,
            'MaxDD_pct_points': None if BDD is None or EDD is None else EDD-BDD,
            'MaxDD_relative_worsen_pct': dd_worsen,
            'Monthly21_pct_points': None if BRET is None or ERET is None else ERET-BRET,
        },
        'base': b,
        'edge': e,
        'warning': 'Promotion requires matching catalog hash, strategy hash and exposure/risk metadata in experiment manifests; this script does not infer missing evidence.'
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(rep, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
