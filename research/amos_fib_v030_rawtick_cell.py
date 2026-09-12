from __future__ import annotations
import argparse, json
from pathlib import Path

from g75_expected_action_rawtick_v5 import load_all
from amos_fib_v030_rawtick_parity import simulate, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--mode', choices=['CORE','ATR','PYR','STDV'], required=True)
    ap.add_argument('--stdv-target', type=float, default=2.65)
    ap.add_argument('--stdv-frac', type=float, default=0.24)
    ap.add_argument('--stdv-stop', type=float, default=0.55)
    ap.add_argument('--raw-bidask-only', action='store_true')
    a = ap.parse_args()
    if not a.raw_bidask_only:
        raise SystemExit('Raw BidAsk mandatory')

    _, ticks = load_all(a.catalog)
    split = int(len(ticks) * 0.40)
    rows = simulate(
        ticks, split, len(ticks), a.mode,
        a.stdv_target, a.stdv_frac, a.stdv_stop,
    )
    s = summarize(rows)
    s.update({
        'mode': a.mode,
        'stdv_target': a.stdv_target,
        'stdv_frac': a.stdv_frac,
        'stdv_stop': a.stdv_stop,
    })
    out = {
        'experiment_id': a.experiment_id,
        'source': 'AMOS_Fib_v0_30_STDV_LocalOpt.mq5',
        'execution': 'raw Bid/Ask tick-by-tick; completed M5 bars only for source-parity indicators/swing geometry',
        'split': 'first 40% warmup/train excluded; final 60% chronological OOS',
        'no_ohlc_execution': True,
        'pd_array_note': 'source placeholder reproduced as EMA9-distance gate (<=1.5 ATR effective bound)',
        'research_reference': {
            'N': 390,
            'WR_pct': 66.923,
            'PF': 8.012,
            'DD_pct': 4.0,
            'Monthly21_pct': 99.467,
        },
        'result': s,
        'verification_level': 'RAW_BIDASK_SOURCE_PARITY_SINGLE_CELL_V2',
    }
    d = Path('results/amos-fib-v030-rawtick-fast') / a.experiment_id
    d.mkdir(parents=True, exist_ok=True)
    fn = f"{a.mode.lower()}-t{a.stdv_target:.2f}-f{a.stdv_frac:.2f}-s{a.stdv_stop:.2f}.json"
    (d / fn).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
