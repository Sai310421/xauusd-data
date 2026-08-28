from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import lzma
import shutil
import struct
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler

REC = struct.Struct('>3i2f')
HEADERS = {'User-Agent': 'raw6x3-nautilus/1.0', 'Accept': '*/*', 'Connection': 'close'}
SIM = Venue('SIM')

SYMBOLS = {
    'XAUUSD': {'pair': 'XAU/USD', 'scale': 1000.0, 'price_precision': 3},
    'EURUSD': {'pair': 'EUR/USD', 'scale': 100000.0, 'price_precision': 5},
    'GBPUSD': {'pair': 'GBP/USD', 'scale': 100000.0, 'price_precision': 5},
    'USDJPY': {'pair': 'USD/JPY', 'scale': 1000.0, 'price_precision': 3},
    'AUDUSD': {'pair': 'AUD/USD', 'scale': 100000.0, 'price_precision': 5},
    'USDCHF': {'pair': 'USD/CHF', 'scale': 100000.0, 'price_precision': 5},
}


def make_instrument(symbol: str, meta: dict) -> CurrencyPair:
    base, quote = meta['pair'].split('/')
    precision = int(meta['price_precision'])
    price_increment = '0.' + ('0' * (precision - 1)) + '1'
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol(symbol), SIM),
        raw_symbol=Symbol(symbol),
        base_currency=Currency.from_str(base),
        quote_currency=Currency.from_str(quote),
        price_precision=precision,
        size_precision=0,
        price_increment=Price.from_str(price_increment),
        size_increment=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


def iter_days(start: dt.datetime, days: int):
    for i in range(days):
        yield start + dt.timedelta(days=i)


def fetch_hour(symbol: str, scale: float, t: dt.datetime):
    urls = [
        f'https://datafeed.dukascopy.com/datafeed/{symbol}/{t.year}/{t.month-1:02d}/{t.day:02d}/{t.hour:02d}h_ticks.bi5',
        f'https://www.dukascopy.com/datafeed/{symbol}/{t.year}/{t.month-1:02d}/{t.day:02d}/{t.hour:02d}h_ticks.bi5',
    ]
    last = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            if not raw:
                continue
            dec = lzma.decompress(raw)
            rows = []
            for i in range(0, len(dec) - REC.size + 1, REC.size):
                ms, ask_i, bid_i, ask_v, bid_v = REC.unpack_from(dec, i)
                ts = t + dt.timedelta(milliseconds=ms)
                ask = ask_i / scale
                bid = bid_i / scale
                if ask <= 0 or bid <= 0 or ask < bid:
                    continue
                rows.append((ts, bid, ask, float(bid_v), float(ask_v)))
            return rows, 200
        except urllib.error.HTTPError as e:
            last = e.code
        except Exception:
            last = -1
    return [], last


def sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name == 'catalog_manifest.json':
            continue
        h.update(str(p.relative_to(root)).encode())
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-07-27')
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--catalog', default='catalog/raw_bidask')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    catalog_path = Path(args.catalog)
    manifest_path = catalog_path / 'catalog_manifest.json'

    if args.fresh and catalog_path.exists():
        shutil.rmtree(catalog_path)
    catalog_path.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        if m.get('status') == 'COMPLETE' and m.get('start') == args.start and m.get('days') == args.days:
            print(json.dumps({'status': 'CATALOG_CACHE_HIT', 'manifest': m}, indent=2))
            return

    catalog = ParquetDataCatalog(str(catalog_path))
    instruments = {symbol: make_instrument(symbol, meta) for symbol, meta in SYMBOLS.items()}
    catalog.write_instruments(list(instruments.values()))

    stats = {}
    for symbol, meta in SYMBOLS.items():
        instrument = instruments[symbol]
        wrangler = QuoteTickDataWrangler(instrument=instrument)
        total_ticks = 0
        status_counts = {}
        written_days = 0

        for day in iter_days(start, args.days):
            hours = [day.replace(hour=h) for h in range(24)]
            rows = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(fetch_hour, symbol, meta['scale'], h): h for h in hours}
                for fut in as_completed(futs):
                    r, status = fut.result()
                    rows.extend(r)
                    status_counts[str(status)] = status_counts.get(str(status), 0) + 1
            if not rows:
                continue
            rows.sort(key=lambda x: x[0])
            df = pd.DataFrame(rows, columns=['datetime', 'bid_price', 'ask_price', 'bid_size', 'ask_size'])
            df = df.drop_duplicates('datetime', keep='last').set_index('datetime')
            ticks = wrangler.process(df)
            if ticks:
                catalog.write_quote_ticks(ticks)
                total_ticks += len(ticks)
                written_days += 1

        stats[symbol] = {
            'instrument_id': instrument.id.value,
            'ticks': total_ticks,
            'written_days': written_days,
            'http_status_counts': status_counts,
        }

    missing = [s for s, v in stats.items() if v['ticks'] <= 0]
    manifest = {
        'status': 'COMPLETE' if not missing else 'INCOMPLETE',
        'data_kind': 'RAW_BIDASK',
        'source': 'Dukascopy BI5 QuoteTick Bid/Ask',
        'venue': 'SIM',
        'symbols': list(SYMBOLS),
        'timeframes': ['M1', 'M5', 'M15'],
        'start': args.start,
        'days': args.days,
        'end_exclusive': (start + dt.timedelta(days=args.days)).date().isoformat(),
        'ohlc_resample_used': False,
        'bar_policy': 'Nautilus INTERNAL bars built directly from raw QuoteTick stream; execution remains QuoteTick based',
        'instrument_provider': 'public-model CurrencyPair constructor; no nautilus_trader.testkit dependency',
        'stats': stats,
        'missing_symbols': missing,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    manifest['catalog_sha256'] = sha256_tree(catalog_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if missing:
        raise SystemExit('RAW CATALOG INCOMPLETE: ' + ','.join(missing))


if __name__ == '__main__':
    main()
