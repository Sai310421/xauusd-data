#!/usr/bin/env python3
"""Export broker XAUUSD bid/ask ticks from a locally running MT5 terminal."""
import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def utc(value):
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('UTC timestamp with timezone required')
    return dt.astimezone(timezone.utc)


def export(symbol,start,end,out,mt5):
    start,end=utc(start),utc(end)
    if not start<end:raise ValueError('start must precede end')
    if (end-start).days>120:raise ValueError('export at most 120 days per run')
    if not mt5.initialize():raise RuntimeError(f'MT5 initialize: {mt5.last_error()}')
    try:
        info=mt5.symbol_info(symbol)
        if info is None or not mt5.symbol_select(symbol,True):
            raise RuntimeError(f'symbol unavailable: {symbol}')
        manifest={'broker_symbol':symbol,'period_start':start.isoformat(),
                  'period_end_exclusive':end.isoformat(),
                  'trade_contract_size':getattr(info,'trade_contract_size',None),
                  'volume_min':getattr(info,'volume_min',None),
                  'volume_step':getattr(info,'volume_step',None),
                  'empty_days':[],'invalid_quotes':0,'ticks':0,
                  'warning':'Availability time is broker quote timestamp, not local strategy receipt time.'}
        previous_ms=None; day=start
        with open(out,'w',encoding='utf8',newline='') as f:
            writer=csv.writer(f);writer.writerow(('symbol','available_at','bid','ask'))
            while day<end:
                stop=min(day+timedelta(days=1),end)
                ticks=mt5.copy_ticks_range(symbol,day,stop,mt5.COPY_TICKS_INFO)
                if ticks is None:raise RuntimeError(f'MT5 copy_ticks_range {day}: {mt5.last_error()}')
                if not len(ticks):manifest['empty_days'].append(day.date().isoformat())
                for tick in ticks:
                    ms=int(tick['time_msc'])
                    if not (int(day.timestamp()*1000)<=ms<int(stop.timestamp()*1000)):
                        continue  # MT5 range endpoints may overlap.
                    if previous_ms is not None and ms<previous_ms:
                        raise RuntimeError('broker ticks out of order')
                    bid,ask=float(tick['bid']),float(tick['ask'])
                    if bid<=0 or ask<bid:
                        manifest['invalid_quotes']+=1
                        continue
                    at=datetime.fromtimestamp(ms/1000,timezone.utc).isoformat(timespec='milliseconds')
                    writer.writerow(('XAUUSD',at,bid,ask))
                    previous_ms=ms;manifest['ticks']+=1
                day=stop
    finally:mt5.shutdown()
    Path(str(out)+'.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--symbol',default='XAUUSD',help='Actual broker symbol, e.g. XAUUSDm')
    p.add_argument('--start',required=True);p.add_argument('--end',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    import MetaTrader5 as mt5
    print(json.dumps(export(args.symbol,args.start,args.end,args.out,mt5),indent=2))
