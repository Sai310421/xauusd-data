"""Audit a Dukascopy XAUUSD Nautilus QuoteTick shard before reuse."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from nautilus_trader.persistence.catalog import ParquetDataCatalog
from shared_bidask.market_calendar import expected_trading_hour, SCHEDULE_SOURCE


def audit(catalog_path: Path, start: str, days: int) -> dict:
    manifest_path = catalog_path / 'catalog_manifest.json'
    result = {'status': 'BLOCKED', 'start': start, 'days': days, 'reasons': [],
              'schedule_source_url': SCHEDULE_SOURCE}
    if not manifest_path.is_file():
        result['reasons'].append('manifest missing')
        return result
    manifest = json.loads(manifest_path.read_text())
    result['manifest_status'] = manifest.get('status')
    if manifest.get('start') != start or manifest.get('days') != days:
        result['reasons'].append('requested date range mismatch')
    if manifest.get('coverage_schema_version') != 2 or manifest.get('unresolved_open_hours'):
        result['reasons'].append('source hourly coverage unresolved or unversioned')
    if manifest.get('status') != 'COMPLETE' or manifest.get('data_kind') != 'RAW_BIDASK':
        result['reasons'].append('source manifest not complete raw bid/ask')
    if manifest.get('symbols') != ['XAUUSD']:
        result['reasons'].append('unexpected instrument set')
    catalog = ParquetDataCatalog(str(catalog_path.resolve()))
    result['manifest_quote_count'] = manifest.get('stats', {}).get('XAUUSD', {}).get('ticks')
    origin = dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc)
    observed = set()
    previous = None
    count = 0
    first_utc = None
    last_utc = None
    for day_no in range(days):
        day_start = origin + dt.timedelta(days=day_no)
        day_end = day_start + dt.timedelta(days=1)
        # Keep memory bounded to a single day even for a 30-day shard.
        ticks = catalog.quote_ticks(instrument_ids=['XAUUSD.SIM'],
                                    start=day_start.isoformat(),
                                    end=(day_end - dt.timedelta(microseconds=1)).isoformat())
        for tick in ticks:
            stamp = dt.datetime.fromtimestamp(tick.ts_event / 1e9, tz=dt.timezone.utc)
            if previous is not None and tick.ts_event <= previous:
                result['reasons'].append('non-increasing quote timestamps')
                break
            previous = tick.ts_event
            if not day_start <= stamp < day_end:
                result['reasons'].append('quote outside queried day')
                break
            if float(tick.ask_price) < float(tick.bid_price):
                result['reasons'].append('negative bid/ask spread')
                break
            first_utc = first_utc or stamp.isoformat()
            last_utc = stamp.isoformat()
            count += 1
            observed.add(stamp.replace(minute=0, second=0, microsecond=0))
        if result['reasons'] and result['reasons'][-1] in ('non-increasing quote timestamps',
                                                            'quote outside queried day', 'negative bid/ask spread'):
            break
    result['quote_count'] = count
    if count != result['manifest_quote_count'] or not count:
        result['reasons'].append('quote count mismatch or empty')
    expected = {origin + dt.timedelta(hours=h) for h in range(days * 24)
                if expected_trading_hour('XAUUSD', origin + dt.timedelta(hours=h))}
    missing = sorted(expected - observed)
    result['expected_open_hours'] = len(expected)
    result['observed_open_hours'] = len(expected & observed)
    result['missing_open_hours'] = [h.isoformat() for h in missing]
    if missing:
        result['reasons'].append(f'{len(missing)} scheduled open hours have no quotes')
    if count:
        result['first_utc'] = first_utc
        result['last_utc'] = last_utc
    if not result['reasons']:
        result['status'] = 'SHARD_HOURLY_COVERAGE_VERIFIED'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--days', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.catalog, args.start, args.days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'SHARD_HOURLY_COVERAGE_VERIFIED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
