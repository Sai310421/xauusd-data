from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(m: dict) -> float:
    pf = float(m.get('PF') or 0.0)
    rf = float(m.get('RF') or 0.0)
    monthly = float(m.get('Monthly21_pct') or 0.0)
    dd = float(m.get('MaxFloatingDD_pct') or m.get('MaxDD_pct') or m.get('MaxClosedDD_pct') or 0.0)
    if dd <= 0:
        dd = 0.01
    return (monthly * max(pf, 0.0) * max(rf, 0.1)) / dd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for p in root.rglob('summary.json'):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        mode_metrics = obj.get('mode_metrics') or obj.get('metrics') or {}
        if isinstance(mode_metrics, dict) and any(isinstance(v, dict) for v in mode_metrics.values()):
            for mode, m in mode_metrics.items():
                if not isinstance(m, dict):
                    continue
                row = {'mode': mode, **m, 'source': str(p)}
                row['score'] = score(row)
                rows.append(row)
        elif isinstance(mode_metrics, dict):
            mode = obj.get('mode') or p.parent.name
            row = {'mode': mode, **mode_metrics, 'source': str(p)}
            row['score'] = score(row)
            rows.append(row)

    rows.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    out = {
        'verification_level': 'NAUTILUS_MATRIX_AGGREGATED',
        'ranking_formula': 'Monthly21 * PF * max(RF,0.1) / MaxFloatingDD(or fallback DD)',
        'rows': rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
