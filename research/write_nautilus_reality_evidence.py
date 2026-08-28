from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--runner', required=True)
    ap.add_argument('--verification-level', default='RAW_SIGNAL_GATE')
    ap.add_argument('--native-spread', action='store_true')
    ap.add_argument('--commission', action='store_true')
    ap.add_argument('--slippage', action='store_true')
    ap.add_argument('--latency', action='store_true')
    ap.add_argument('--reject-fill', action='store_true')
    ap.add_argument('--floating-mtm-dd', action='store_true')
    args = ap.parse_args()

    implemented = {
        'native_bid_ask_spread': bool(args.native_spread),
        'commission': bool(args.commission),
        'slippage': bool(args.slippage),
        'latency': bool(args.latency),
        'reject_fill': bool(args.reject_fill),
        'floating_mark_to_market_drawdown': bool(args.floating_mtm_dd),
    }
    production = all(implemented[k] for k in (
        'commission', 'slippage', 'latency', 'reject_fill', 'floating_mark_to_market_drawdown'
    )) and implemented['native_bid_ask_spread']

    obj = {
        'schema': 'NAUTILUS_REALITY_EVIDENCE_v1',
        'runner': args.runner,
        'verification_level': 'PRODUCTION_CANDIDATE' if production else args.verification_level,
        'implemented': implemented,
        'production_candidate': production,
        'rule': 'Fail closed: missing reality components cannot be inferred from native spread or from post-hoc KPI adjustment.',
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
