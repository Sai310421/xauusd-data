from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = [
    'commission',
    'slippage',
    'latency',
    'reject_fill',
    'floating_mark_to_market_drawdown',
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--allow-signal-gate-only', action='store_true')
    args = ap.parse_args()

    p = Path(args.evidence)
    if not p.exists():
        raise SystemExit(f'REALITY_GATE_FAIL\nevidence missing: {p}')
    obj = json.loads(p.read_text(encoding='utf-8'))
    impl = obj.get('implemented', {})
    missing = [k for k in REQUIRED if impl.get(k) is not True]

    out = {
        'status': 'PRODUCTION_REALITY_OK' if not missing else 'SIGNAL_GATE_ONLY',
        'required': REQUIRED,
        'missing': missing,
        'evidence': str(p),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if missing and not args.allow_signal_gate_only:
        raise SystemExit('REALITY_GATE_FAIL\nmissing: ' + ','.join(missing))


if __name__ == '__main__':
    main()
