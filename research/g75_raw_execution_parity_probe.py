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


def _event_record(event):
    rec = {'type': type(event).__name__, 'repr': repr(event)}
    for name in ('client_order_id', 'venue_order_id', 'reason', 'last_px', 'last_qty', 'order_side', 'ts_event'):
        if hasattr(event, name):
            try:
                rec[name] = str(getattr(event, name))
            except Exception:
                pass
    return rec


class DiagnosticG75VGRSIStrategy(G75VGRSIStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.order_events = []

    def on_order_submitted(self, event):
        self.order_events.append(_event_record(event))

    def on_order_accepted(self, event):
        self.order_events.append(_event_record(event))

    def on_order_denied(self, event):
        self.order_events.append(_event_record(event))
        oid = str(event.client_order_id)
        p = self.pending_opens.pop(oid, None)
        if p is not None:
            if p.tag == 'ENTRY':
                self.entry_order_id = None
            else:
                self.pending_adds = max(0, self.pending_adds - 1)

    def on_order_rejected(self, event):
        self.order_events.append(_event_record(event))
        oid = str(event.client_order_id)
        p = self.pending_opens.pop(oid, None)
        if p is not None:
            if p.tag == 'ENTRY':
                self.entry_order_id = None
            else:
                self.pending_adds = max(0, self.pending_adds - 1)

    def on_order_canceled(self, event):
        self.order_events.append(_event_record(event))

    def on_order_filled(self, event):
        self.order_events.append(_event_record(event))
        super().on_order_filled(event)


def frame_to_records(df):
    if df is None:
        return []
    try:
        return json.loads(df.to_json(orient='records', date_format='iso'))
    except Exception:
        return [{'repr': repr(df)}]


def safe_positions_report(engine: BacktestEngine):
    try:
        return frame_to_records(engine.generate_positions_report())
    except Exception as exc:
        return [{'report_error': f'{type(exc).__name__}: {exc}'}]


def _order_record(order):
    rec = {'type': type(order).__name__, 'repr': repr(order)}
    for name in ('client_order_id', 'venue_order_id', 'status', 'side', 'quantity', 'filled_qty', 'leaves_qty', 'instrument_id'):
        if hasattr(order, name):
            try:
                rec[name] = str(getattr(order, name))
            except Exception:
                pass
    return rec


def cache_order_snapshot(engine: BacktestEngine, instrument_id):
    out = {'orders_total': None, 'orders_open': None, 'positions_open': None, 'orders': [], 'errors': []}
    cache = getattr(engine, 'cache', None)
    if cache is None:
        out['errors'].append('engine.cache unavailable')
        return out
    for key, name in (
        ('orders_total', 'orders'),
        ('orders_open', 'orders_open'),
        ('positions_open', 'positions_open'),
    ):
        fn = getattr(cache, name, None)
        if fn is None:
            out['errors'].append(f'cache.{name} unavailable')
            continue
        try:
            try:
                xs = fn(instrument_id=instrument_id)
            except TypeError:
                xs = fn()
            xs = list(xs) if xs is not None else []
            out[key] = len(xs)
            if name == 'orders':
                out['orders'] = [_order_record(x) for x in xs[-10:]]
        except Exception as exc:
            out['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    return out


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
    strat = DiagnosticG75VGRSIStrategy(G75VGRSIConfig(instrument_id=instrument.id, mode='BASE', max_layers=10))
    engine.add_strategy(strat)
    engine.run()
    positions = safe_positions_report(engine)
    cache_snapshot = cache_order_snapshot(engine, instrument.id)
    out = {
        'oms_type': oms_type.name,
        'raw_ticks': len(ticks),
        'instrument': {
            'id': str(instrument.id),
            'size_precision': getattr(instrument, 'size_precision', None),
            'size_increment': str(getattr(instrument, 'size_increment', None)),
        },
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
            'cycle_count': len(strat.cycle_pnls),
            'max_layers_seen': strat.max_layers_seen,
            'order_events_tail': strat.order_events[-20:],
        },
        'cache': cache_snapshot,
        'positions_count': len(positions),
        'positions_tail': positions[-10:],
    }
    engine.dispose()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--symbol', default='XAUUSD')
    ap.add_argument('--oms', required=True, choices=('HEDGING', 'NETTING'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    oms_type = OmsType.HEDGING if args.oms == 'HEDGING' else OmsType.NETTING
    data = {
        'purpose': 'Diagnose G75 order lifecycle before changing strategy logic',
        'rule': 'One Nautilus engine per OS process; capture submit/accept/deny/reject/fill lifecycle',
        'cell': run_cell(Path(args.catalog).resolve(), args.symbol, oms_type),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
