from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.g75_vgrsi_raw_bidask_bt import G75VGRSIConfig, G75VGRSIStrategy

SIM = Venue('SIM')


def frame_to_records(df):
    if df is None:
        return []
    try:
        return json.loads(df.to_json(orient='records', date_format='iso'))
    except Exception:
        return [{'repr': repr(df)}]


def run_cell(catalog_path: Path, symbol: str, oms_type: OmsType):
    catalog = ParquetDataCatalog(str(catalog_path))
    inst_by_plain = {x.id.symbol.value.replace('/', ''): x for x in catalog.instruments()}
    instrument = inst_by_plain[symbol]
    ticks = catalog.quote_ticks(instrument_ids=[instrument.id.value])
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id=TraderId(f'G75PROBE-{oms_type.name[:3]}'),
        logging=LoggingConfig(log_level='ERROR'),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    engine.add_venue(
        venue=SIM,
        oms_type=oms_type,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(1000, USD)],
        default_leverage=Decimal('2000'),
    )
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    strat = G75VGRSIStrategy(G75VGRSIConfig(instrument_id=instrument.id, mode='BASE', max_layers=10))
    engine.add_strategy(strat)
    engine.run()
    orders = frame_to_records(engine.generate_orders_report())
    positions = frame_to_records(engine.generate_positions_report())
    out = {
        'oms_type': oms_type.name,
        'raw_ticks': len(ticks),
        'strategy': {
            'triggers': strat.triggers,
            'fills_seen': strat.entries + strat.adds,
            'entries_filled': strat.entries,
            'adds_filled': strat.adds,
            'exits': strat.exits,
            'pending_side': strat.pending_side,
            'entry_order_id': str(strat.entry_order_id) if strat.entry_order_id is not None else None,
            'pending_open_count': len(strat.pending_opens),
            'active_side': strat.active_side,
        },
        'orders_count': len(orders),
        'orders_tail': orders[-10:],
        'positions_count': len(positions),
        'positions_tail': positions[-10:],
    }
    engine.dispose()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--symbol', default='XAUUSD')
    ap.add_argument('--out', default='results/ae-bt/g75-execution-parity-probe.json')
    args = ap.parse_args()
    catalog_path = Path(args.catalog).resolve()
    data = {
        'purpose': 'Diagnose N=0 before changing G75 logic',
        'cells': [
            run_cell(catalog_path, args.symbol, OmsType.HEDGING),
            run_cell(catalog_path, args.symbol, OmsType.NETTING),
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
