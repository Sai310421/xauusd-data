from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from nautilus_trader.model.enums import BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import research.amos_allweather_raw_bidask_bt_compat as compat

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

    allts, cells, scenes, counts, trans, raw, lifecycle = [], {}, {}, {}, {}, {}, {}
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
            eng = base.BacktestEngine(config=base.BacktestEngineConfig(
                logging=base.LoggingConfig(log_level='ERROR'),
                risk_engine=base.RiskEngineConfig(bypass=True),
            ))
            eng.add_venue(
                venue=base.SIM,
                oms_type=base.OmsType.NETTING,
                account_type=base.AccountType.MARGIN,
                base_currency=base.USD,
                starting_balances=[base.Money(1000, base.USD)],
                default_leverage=Decimal('2000'),
                book_type=BookType.L1_MBP,
            )
            eng.add_instrument(ins)
            eng.add_data(ticks)
            bt = base.BarType.from_str(f'{ins.id.value}-{mins}-MINUTE-BID-INTERNAL')
            st = compat.CompatStrat(base.Cfg(
                instrument_id=ins.id,
                bar_type=bt,
                trade_size=Decimal('1'),
                tf_minutes=mins,
            ))
            eng.add_strategy(st)
            eng.run()

            tt = [dict(t, symbol=symbol, tf=tf) for t in st.closed_trades]
            allts += tt
            key = f'{symbol}:{tf}'
            cells[key] = {**base.metrics(tt, days=days), 'raw_ticks': len(ticks), 'signals_submitted': st.entries}
            lifecycle[key] = dict(st.lifecycle)
            counts[key] = dict(st.scene_counts)
            trans[key] = dict(st.transitions)
            for sc in sorted({t['scene'] for t in tt}):
                scenes[f'{key}:{sc}'] = base.metrics([t for t in tt if t['scene'] == sc], days=days)
            eng.dispose()

    allts.sort(key=lambda x: (x['ts_closed'], x['symbol'], x['tf']))
    summary = {
        'verification_level': 'NAUTILUS_BT_RAW_BIDASK',
        'strategy': 'AMOS_AllWeather_XAUUSD_MetaBot_v0.2',
        'engine': 'NautilusTrader BacktestEngine',
        'nautilus_version': getattr(base.nautilus_trader, '__version__', 'unknown'),
        'data_kind': 'RAW_BIDASK QuoteTick',
        'ohlc_resample_used': False,
        'book_type': 'L1_MBP',
        'signal_bars': 'Nautilus INTERNAL BID bars built from raw QuoteTicks',
        'execution': 'MARKET orders on L1 QuoteTick market; native observed spread included; explicit commission/slippage model not yet added',
        'symbols': ['XAUUSD'],
        'timeframes': a.timeframes,
        'period': {'start': manifest['start'], 'days': days, 'end_exclusive': manifest['end_exclusive']},
        'portfolio_realized_close_ordered': base.metrics(allts, days=days),
        'cell_metrics': cells,
        'lifecycle_metrics': lifecycle,
        'scene_metrics': scenes,
        'scene_bar_counts': counts,
        'state_transitions': trans,
        'raw_tick_counts': raw,
        'limitations': [
            'Scheduled-news calendar is not present in the raw catalog; NEWS is not directly event-labelled in v0.2.',
            'IFVG/BPR/Breaker are reserved but not fully reconstructed in v0.2.',
            'Portfolio DD is reconstructed from realized closed PnL across independent TF cells, not synchronized mark-to-market equity.',
            'No explicit commission/probabilistic slippage model yet; raw Bid/Ask spread is native.',
        ],
    }
    cols = ['scene', 'pnl', 'ts_closed', 'realized_return', 'symbol', 'tf']
    base.pd.DataFrame(allts, columns=cols).to_csv(out / 'trades.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / 'catalog_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('MARKET_BOOK_TYPE=L1_MBP')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
