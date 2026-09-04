from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research.minimumspike_raw6x3_bt import metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--symbols', nargs='+', required=True)
    ap.add_argument('--timeframes', nargs='+', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--raw-bidask-only', action='store_true')
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit('raw-bidask-only is mandatory')

    catalog_path = Path(args.catalog)
    manifest = json.loads((catalog_path / 'catalog_manifest.json').read_text(encoding='utf-8'))
    days = int(manifest['days'])
    outdir = Path('results/ae-bt') / args.experiment_id
    cells_dir = outdir / 'cells'
    cells_dir.mkdir(parents=True, exist_ok=True)

    all_trades: list[dict] = []
    cell_metrics: dict[str, dict] = {}
    raw_counts: dict[str, int] = {}

    for symbol in args.symbols:
        for tf in args.timeframes:
            child_id = f'{args.experiment_id}__{symbol}_{tf}'
            child_dir = Path('results/ae-bt') / child_id
            log_path = cells_dir / f'{symbol}_{tf}.log'
            cmd = [
                sys.executable, '-m', 'research.minimumspike_raw6x3_bt_compat',
                '--catalog', args.catalog,
                '--symbols', symbol,
                '--timeframes', tf,
                '--experiment-id', child_id,
                '--raw-bidask-only',
            ]
            with log_path.open('w', encoding='utf-8') as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-80:]
                print('\n'.join(tail), file=sys.stderr)
                raise SystemExit(f'CELL_FAILED {symbol}:{tf} rc={proc.returncode}')

            child_summary_path = child_dir / 'summary.json'
            child_trades_path = child_dir / 'trades.csv'
            if not child_summary_path.exists():
                raise SystemExit(f'CELL_MISSING_SUMMARY {symbol}:{tf}')
            child = json.loads(child_summary_path.read_text(encoding='utf-8'))
            key = f'{symbol}:{tf}'
            if key not in child.get('cell_metrics', {}):
                raise SystemExit(f'CELL_METRICS_MISSING {key}')
            cell_metrics[key] = child['cell_metrics'][key]
            if symbol in child.get('raw_tick_counts', {}):
                raw_counts[symbol] = int(child['raw_tick_counts'][symbol])

            shutil.copy2(child_summary_path, cells_dir / f'{symbol}_{tf}_summary.json')
            if child_trades_path.exists() and child_trades_path.stat().st_size > 0:
                try:
                    df = pd.read_csv(child_trades_path)
                except pd.errors.EmptyDataError:
                    df = pd.DataFrame()
                if not df.empty:
                    all_trades.extend(df.to_dict(orient='records'))
            shutil.rmtree(child_dir, ignore_errors=True)

    all_trades.sort(key=lambda x: (int(x.get('ts_closed', 0)), str(x.get('symbol', '')), str(x.get('tf', ''))))
    portfolio = metrics(all_trades, days=days)
    summary = {
        'verification_level': 'NAUTILUS_BT_RAW_BIDASK_PROCESS_ISOLATED',
        'engine': 'NautilusTrader BacktestEngine; one fresh OS process per symbol/TF cell',
        'nautilus_version': '1.230.0',
        'data_kind': 'RAW_BIDASK QuoteTick',
        'ohlc_resample_used': False,
        'process_isolation': True,
        'process_isolation_reason': 'NautilusTrader 1.230.0 Rust logger is process-global and aborts when multiple BacktestEngine loggers are initialized sequentially in one process.',
        'signal_bars': 'Nautilus INTERNAL BID bars built from raw QuoteTicks',
        'execution': 'Nautilus simulated venue MARKET orders on raw QuoteTicks; observed spread included; commission/slippage model not yet added',
        'symbols': args.symbols,
        'timeframes': args.timeframes,
        'period': {'start': manifest['start'], 'days': days, 'end_exclusive': manifest['end_exclusive']},
        'portfolio_realized_close_ordered': portfolio,
        'cell_metrics': cell_metrics,
        'raw_tick_counts': raw_counts,
        'limitations': [
            'Portfolio DD is reconstructed from realized closed-position PnL across independent symbol/TF engine cells, not synchronized mark-to-market portfolio equity.',
            'No explicit commission or probabilistic slippage model in this RAW_BIDASK screening gate; raw Bid/Ask spread is native.',
            'Each symbol/TF is run in a separate OS process and independent Nautilus BacktestEngine cell to avoid logger reinitialization and cross-TF position-netting contamination.',
        ],
    }

    pd.DataFrame(all_trades).to_csv(outdir / 'trades.csv', index=False)
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir / 'catalog_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
