#!/usr/bin/env python3
"""Export the pinned Nautilus raw Bid/Ask cache and run Jev-inspired XAU OOS gate."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from xau_oos_gate import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--catalog',required=True)
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()

    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from nautilus_catalog_compat import select_instrument_compat,query_quote_ticks_compat

    root=Path(args.catalog);manifest_path=root/'catalog_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('status')!='COMPLETE' or manifest.get('data_kind')!='RAW_BIDASK' or manifest.get('ohlc_resample_used') is not False:
        raise SystemExit('RAW_CATALOG_INVALID')
    if manifest.get('start')!='2026-07-27' or manifest.get('days')!=30:
        raise SystemExit('PINNED_30D_DATASET_REQUIRED')
    source_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    catalog=ParquetDataCatalog(str(root.resolve()))
    instrument=select_instrument_compat(catalog,'XAUUSD')
    start=datetime(2026,7,27,tzinfo=timezone.utc)
    end=start+timedelta(days=30)
    csv_path=out/'raw_xau_for_gate.csv'
    n=0;prior_ns=None;bad=0;missing_weekdays=[];daily_counts={}
    with csv_path.open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['symbol','available_at','bid','ask'])
        day=start
        while day<end:
            stop=day+timedelta(days=1)
            ticks=query_quote_ticks_compat(catalog,identifiers=[instrument.id.value],
                                          start=day.isoformat(),end=stop.isoformat())
            day_n=0
            for tick in ticks:
                ns=int(tick.ts_event)
                if not int(day.timestamp()*1e9)<=ns<int(stop.timestamp()*1e9):continue
                if prior_ns is not None and ns<prior_ns:raise SystemExit('RAW_TICK_ORDER_INVALID')
                bid=float(tick.bid_price.as_double());ask=float(tick.ask_price.as_double())
                if not (0<bid<=ask):bad+=1;continue
                at=datetime.fromtimestamp(ns/1e9,timezone.utc).isoformat(timespec='microseconds')
                writer.writerow(['XAUUSD',at,bid,ask]);n+=1;day_n+=1;prior_ns=ns
            daily_counts[day.date().isoformat()]=day_n
            if day.weekday()<5 and day_n==0:missing_weekdays.append(day.date().isoformat())
            day=stop
    if n<10000:raise SystemExit(f'RAW_TICK_COUNT_TOO_LOW:{n}')
    result=evaluate(csv_path,'2026-08-13T23:59:59Z','2026-08-15T00:00:00Z',
                    source=f'Dukascopy BI5 Nautilus cache manifest_sha256={source_hash}',
                    capital=1000.,lot=.01,contract_oz=100.,
                    slippage_usd_per_side=.1,min_cal_trades=60,min_oos_trades=60)
    result['catalog_manifest_sha256']=source_hash
    result['raw_catalog_ticks']=n
    result['raw_invalid_quotes']=bad
    result['daily_tick_counts']=daily_counts
    result['missing_weekdays']=missing_weekdays
    # Thirty calendar days cannot establish the user's 90-day profitability gate.
    result['original_30d_gate']=result.pop('gate')
    result['gate']=('BLOCKED_RAW_WEEKDAY_GAPS' if missing_weekdays else
                    'PROVISIONAL_30D_REQUIRES_90D_AND_BROKER_PARITY')
    with csv_path.open('rb') as stream:
        result['raw_csv_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('oos','cost_stress_oos','calibration_trials')},indent=2))
    if result['gate'].startswith('BLOCKED'):
        raise SystemExit(result['gate'])


if __name__=='__main__':main()
