#!/usr/bin/env python3
"""Frozen AMOS LLM EA default candidate, diagnostic raw Bid/Ask quote replay.

Assumptions are explicit; this is not MT5/Nautilus order execution parity.
"""
import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def replay(catalog_path, commission_roundturn=0.07, slippage_side=0.10):
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from research.nautilus_catalog_compat import select_instrument_compat, query_quote_ticks_compat

    root = Path(catalog_path)
    manifest_path = root / 'catalog_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('data_kind') != 'RAW_BIDASK' or manifest.get('ohlc_resample_used') is not False:
        raise ValueError('Raw Bid/Ask provenance check failed')
    catalog = ParquetDataCatalog(str(root.resolve()))
    instrument = select_instrument_compat(catalog, 'XAUUSD')
    begin = datetime(2026, 7, 27, tzinfo=timezone.utc)
    end = begin + timedelta(days=30)
    mids = []
    pending = 0
    position = None
    held = 0
    equity = 1000.0
    peak = equity
    daily_open = equity
    day_key = None
    max_floating_dd = 0.0
    rows = []
    days = []
    last_ns = None
    n_ticks = 0
    risk_blocks = 0
    spread_blocks = 0
    all_spreads = []
    for i in range(30):
        day = begin + timedelta(days=i)
        stop = day + timedelta(days=1)
        ticks = query_quote_ticks_compat(catalog, identifiers=[instrument.id.value],
                                         start=day.isoformat(), end=stop.isoformat())
        count = 0
        for tick in ticks:
            ns = int(tick.ts_event)
            if not int(day.timestamp()*1e9) <= ns < int(stop.timestamp()*1e9):
                continue
            if last_ns is not None and ns < last_ns:
                raise ValueError('Raw quote order reversed')
            last_ns = ns
            bid = float(tick.bid_price.as_double())
            ask = float(tick.ask_price.as_double())
            if not 0 < bid <= ask:
                raise ValueError('Invalid bid/ask')
            count += 1
            n_ticks += 1
            at = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
            mid = (bid + ask) / 2
            spread = ask-bid
            all_spreads.append(spread)
            key = at.date()
            if key != day_key:
                daily_open = equity
                day_key = key

            # Open position marked to executable close price, including exit slippage/fees.
            floating = 0.0
            if position:
                side, entry, entry_ns = position
                closing = bid if side > 0 else ask
                floating = side*(closing-entry) - commission_roundturn - 2*slippage_side
            marked = equity + floating
            peak = max(peak, marked)
            max_floating_dd = max(max_floating_dd, (peak-marked)/peak)
            risk = (daily_open-marked)/daily_open < 0.03 and (peak-marked)/peak < 0.05
            if not risk:
                risk_blocks += 1
            if spread > 0.80:
                spread_blocks += 1
            window = at.weekday() < 5 and not (at.weekday() == 4 and at.hour >= 12)

            # Bid/Ask triggers; price gaps close at first observed executable quote.
            was_owned = position is not None
            if position:
                side, entry, entry_ns = position
                held += 1
                close_quote = bid if side > 0 else ask
                move = side*(close_quote-entry)
                reason = 'SL' if move <= -3.0 else 'TP' if move >= 4.5 else 'HOLD' if held >= 12 else None
                if reason:
                    net = move - commission_roundturn - 2*slippage_side
                    equity += net
                    rows.append({'entry_ns':entry_ns,'exit_ns':ns,'side':side,'entry':entry,
                                 'exit':close_quote,'gross':move,'net':net,'reason':reason})
                    position = None
                    held = 0
            # A signal from preceding quote executes at this quote only when flat.
            old_pending = pending
            pending = 0
            if old_pending and not was_owned and risk and spread <= 0.80 and window:
                position = (old_pending, ask if old_pending > 0 else bid, ns)
                held = 0
            if len(mids) >= 30 and not was_owned and risk and window:
                delta = mid - mids[-30]
                pending = 1 if delta > 0.035 else -1 if delta < -0.035 else 0
            mids.append(mid)
            if len(mids) > 30:
                mids.pop(0)
        days.append({'date':day.date().isoformat(),'quotes':count})

    # An unfinished trade is forcibly closed at the last quote for accounting.
    # Its last observed quote is already included in marked DD; track separately.
    open_position = position is not None
    gains = sum(x['net'] for x in rows if x['net'] > 0)
    losses = -sum(x['net'] for x in rows if x['net'] < 0)
    weekdays_missing = [x['date'] for x in days if x['quotes'] == 0 and
                        datetime.fromisoformat(x['date']).weekday() < 5]
    n = len(rows)
    spreads = sorted(all_spreads)
    return {
      'status':'REAL_RAW_QUOTE_DIAGNOSTIC_INCOMPLETE' if weekdays_missing else 'REAL_RAW_QUOTE_DIAGNOSTIC',
      'market_source':'Dukascopy BI5 in xauusd-data Nautilus raw Bid/Ask Actions cache',
      'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
      'period_utc':[begin.isoformat(),end.isoformat()], 'candidate':'EA defaults: momentum lookback=30 threshold=$0.035 hold=12',
      'cash_initial_usd':1000, 'lot':0.01, 'contract_oz_assumed':100,
      'commission_roundtrip_usd_assumed':commission_roundturn,
      'slippage_usd_per_side_assumed':slippage_side,
      'spread_in_fills':True, 'n_quotes':n_ticks, 'n_trades':n,
      'net_pnl_usd':sum(x['net'] for x in rows),
      'profit_factor':gains/losses if losses else None,
      'win_rate':sum(x['net'] > 0 for x in rows)/n if n else None,
      'expectancy_usd':sum(x['net'] for x in rows)/n if n else None,
      'max_floating_dd_pct':100*max_floating_dd,
      'spread_median_usd':spreads[len(spreads)//2] if spreads else None,
      'spread_p95_usd':spreads[int(.95*(len(spreads)-1))] if spreads else None,
      'weekdays_missing':weekdays_missing, 'daily_quote_counts':days,
      'risk_blocked_quote_count':risk_blocks,'wide_spread_quote_count':spread_blocks,
      'unclosed_position_at_end':open_position,
      'trades':rows,
      'promotion':'BLOCKED',
      'limitations':['Missing trading days invalidate a continuous OOS claim.',
                     'Retrospective frozen default, not an untouched sealed OOS window.',
                     'No broker margin, order rejection, swap, variable slippage or MT5 parity.',
                     'Held ticks and SL/TP emulate EA but broker fill sequence can differ.']}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=replay(args.catalog)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'trades'},indent=2))
