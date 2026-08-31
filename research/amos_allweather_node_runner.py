from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.config import BacktestDataConfig, BacktestEngineConfig, BacktestRunConfig, BacktestVenueConfig
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import research.amos_allweather_raw_bidask_bt_compat as compat
from research.amos_allweather_autofit_strategy import AutoFitCompatStrat, fit_snapshot

base = compat.base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--symbols', nargs='+', required=True)
    ap.add_argument('--timeframes', nargs='+', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--raw-bidask-only', action='store_true')
    a = ap.parse_args()
    if not a.raw_bidask_only:
        raise SystemExit('raw-bidask-only is mandatory')

    path = Path(a.catalog)
    manifest = json.loads((path / 'catalog_manifest.json').read_text())
    days = int(manifest['days'])
    cat = ParquetDataCatalog(str(path))
    insts = {x.id.symbol.value.replace('/', ''): x for x in cat.instruments()}
    out = Path('results/ae-bt') / a.experiment_id
    out.mkdir(parents=True, exist_ok=True)

    allts, cells, scenes, counts, trans, raw, lifecycle, autofit = [], {}, {}, {}, {}, {}, {}, {}

    for symbol in [s for s in a.symbols if s == 'XAUUSD']:
        ins = insts.get(symbol)
        if ins is None:
            raise SystemExit(f'instrument missing: {symbol}')

        ticks = cat.query_quote_ticks(identifiers=[ins.id.value])
        raw[symbol] = len(ticks)
        if not ticks:
            raise SystemExit('no raw QuoteTicks: XAUUSD')

        for tf in a.timeframes:
            mins = base.TF_MIN[tf]
            venue_cfg = BacktestVenueConfig(
                name='SIM',
                oms_type='NETTING',
                account_type='MARGIN',
                book_type='L1_MBP',
                base_currency='USD',
                starting_balances=['1000 USD'],
                default_leverage=2000.0,
            )
            data_cfg = BacktestDataConfig(
                catalog_path=str(path),
                data_cls=QuoteTick,
                instrument_id=ins.id,
            )
            run_cfg = BacktestRunConfig(
                venues=[venue_cfg],
                data=[data_cfg],
                engine=BacktestEngineConfig(
                    logging=base.LoggingConfig(log_level='ERROR'),
                    risk_engine=base.RiskEngineConfig(bypass=True),
                ),
                dispose_on_completion=False,
                raise_exception=True,
            )

            node = BacktestNode(configs=[run_cfg])
            node.build()
            engine = node.get_engine(run_cfg.id)
            if engine is None:
                raise RuntimeError(f'BacktestNode failed to build engine for {run_cfg.id}')

            bt = base.BarType.from_str(f'{ins.id.value}-{mins}-MINUTE-BID-INTERNAL')
            st = AutoFitCompatStrat(base.Cfg(
                instrument_id=ins.id,
                bar_type=bt,
                trade_size=Decimal('1'),
                tf_minutes=mins,
            ))
            engine.add_strategy(st)
            node.run()

            tt = [dict(t, symbol=symbol, tf=tf) for t in st.closed_trades]
            allts += tt
            key = f'{symbol}:{tf}'
            cells[key] = {**base.metrics(tt, days=days), 'raw_ticks': len(ticks), 'signals_submitted': st.entries}
            lifecycle[key] = dict(st.lifecycle)
            lifecycle[key]['rejection_reasons'] = dict(getattr(st, 'rejection_reasons', {}))
            counts[key] = dict(st.scene_counts)
            trans[key] = dict(st.transitions)
            autofit[key] = fit_snapshot(st)
            for sc in sorted({t['scene'] for t in tt}):
                scenes[f'{key}:{sc}'] = base.metrics([t for t in tt if t['scene'] == sc], days=days)
            node.dispose()

    allts.sort(key=lambda x: (x['ts_closed'], x['symbol'], x['tf']))
    summary = {
        'verification_level': 'NAUTILUS_BT_RAW_BIDASK_NODE_AUTOFIT',
        'strategy': 'AMOS_AllWeather_XAUUSD_MetaBot_v0.3_AutoFit',
        'engine': 'NautilusTrader BacktestNode 1.230 + ExecutionAutoFit',
        'nautilus_version': getattr(base.nautilus_trader, '__version__', 'unknown'),
        'data_kind': 'RAW_BIDASK QuoteTick',
        'ohlc_resample_used': False,
        'book_type': 'L1_MBP',
        'execution': 'Catalog-driven QuoteTick execution with market-ready gate, dynamic spread gate, causal volatility sizing and TF execution weight',
        'symbols': ['XAUUSD'],
        'timeframes': a.timeframes,
        'period': {'start': manifest['start'], 'days': days, 'end_exclusive': manifest['end_exclusive']},
        'catalog_execution_contract': {
            'size_precision': manifest.get('size_precision'),
            'size_increment': manifest.get('size_increment'),
            'volume_policy': manifest.get('volume_policy'),
        },
        'portfolio_realized_close_ordered': base.metrics(allts, days=days),
        'cell_metrics': cells,
        'lifecycle_metrics': lifecycle,
        'autofit_snapshot': autofit,
        'scene_metrics': scenes,
        'scene_bar_counts': counts,
        'state_transitions': trans,
        'raw_tick_counts': raw,
        'limitations': [
            'Scheduled-news calendar is not present in the raw catalog; NEWS is not directly event-labelled in v0.3.',
            'IFVG/BPR/Breaker are reserved but not fully reconstructed in v0.3.',
            'Portfolio DD is reconstructed from realized closed PnL across independent TF cells, not synchronized mark-to-market equity.',
            'No explicit probabilistic latency/slippage model yet; raw Bid/Ask spread is native and dynamically gated.',
        ],
    }
    cols = ['scene', 'pnl', 'ts_closed', 'realized_return', 'symbol', 'tf']
    base.pd.DataFrame(allts, columns=cols).to_csv(out / 'trades.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / 'catalog_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('ENGINE_MODE=BACKTEST_NODE_1_230_EXECUTION_AUTOFIT')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
