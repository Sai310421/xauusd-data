from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import lzma
import math
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# FROZEN BigPlayer formula constants copied from research/bigplayer_2edge_bt.py.
LOOKBACK = 120
VOL_SIGMA = 2.0
RANGE_MULT = 1.5
WICK_RATIO = 1.2
POINT = 0.01

REC = struct.Struct('>3i2f')
HOSTS = (
    'https://datafeed.dukascopy.com/datafeed',
    'https://www.dukascopy.com/datafeed',
)
HEADERS = {'User-Agent': 'bigplayer-rawtick-5tf/1.0', 'Accept': '*/*'}
TF_SECONDS = {'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800, 'H1': 3600}
OUT = Path('results/bigplayer_rawtick_5tf_21d')


def business_days(start: dt.date, n: int) -> list[dt.date]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def fetch_hour(day: dt.date, hour: int):
    origin = dt.datetime(day.year, day.month, day.day, hour, tzinfo=dt.timezone.utc)
    rel = f'XAUUSD/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5'
    last = None
    for host in HOSTS:
        try:
            req = urllib.request.Request(f'{host}/{rel}', headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                raw = r.read()
            dec = lzma.decompress(raw)
            rows = []
            for i in range(0, len(dec) - REC.size + 1, REC.size):
                ms, ask_i, bid_i, ask_v, bid_v = REC.unpack_from(dec, i)
                ask = ask_i / 1000.0
                bid = bid_i / 1000.0
                if ask <= 0 or bid <= 0 or ask < bid:
                    continue
                ts = origin + dt.timedelta(milliseconds=ms)
                rows.append((ts, bid, ask, float(bid_v), float(ask_v)))
            return rows, 200
        except urllib.error.HTTPError as e:
            last = e.code
        except Exception:
            last = -1
    return [], last


def load_ticks(days: list[dt.date], workers: int) -> pd.DataFrame:
    jobs = [(d, h) for d in days for h in range(24)]
    rows = []
    status = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_hour, d, h): (d, h) for d, h in jobs}
        for fut in cf.as_completed(futs):
            r, s = fut.result()
            rows.extend(r)
            status[str(s)] = status.get(str(s), 0) + 1
    if not rows:
        raise SystemExit('no raw ticks downloaded')
    rows.sort(key=lambda x: x[0])
    df = pd.DataFrame(rows, columns=['datetime','bid','ask','bid_size','ask_size'])
    df = df.drop_duplicates('datetime', keep='last').reset_index(drop=True)
    df['mid'] = (df.bid + df.ask) / 2.0
    df['raw_volume'] = df.bid_size + df.ask_size
    (OUT / 'download_status.json').write_text(json.dumps({'hours': len(jobs), 'status': status, 'ticks': len(df)}, indent=2), encoding='utf-8')
    return df


def build_bars_from_raw_ticks(ticks: pd.DataFrame, tf_sec: int) -> pd.DataFrame:
    # Direct tick->bar aggregation. This is NOT OHLC-to-OHLC resampling.
    ns = ticks.datetime.astype('int64').to_numpy()
    bucket_ns = tf_sec * 1_000_000_000
    key = (ns // bucket_ns) * bucket_ns
    x = ticks.assign(bucket=pd.to_datetime(key, utc=True))
    g = x.groupby('bucket', sort=True)
    bars = g.agg(open=('mid','first'), high=('mid','max'), low=('mid','min'), close=('mid','last'), volume=('raw_volume','sum')).reset_index().rename(columns={'bucket':'datetime'})
    return bars


def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    # BYTE-FOR-BYTE mathematical structure preserved from BigPlayer 2-edge baseline.
    out = df.copy()
    rng = out['high'] - out['low']
    vol = out['volume'].astype(float)
    vol_mean = vol.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).mean()
    vol_std = vol.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).std(ddof=0)
    range_mean = rng.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).mean()
    z = (vol - vol_mean) / vol_std.replace(0.0, np.nan)
    body = (out['close'] - out['open']).abs()
    body_safe = body.clip(lower=POINT)
    body_ratio = body / rng.replace(0.0, np.nan)
    upper = out['high'] - out[['open','close']].max(axis=1)
    lower = out[['open','close']].min(axis=1) - out['low']
    high_volume = (z >= VOL_SIGMA) & (rng > 0.0)
    imbalance = pd.Series(0, index=out.index, dtype='int8')
    imb_gate = high_volume & ((rng / range_mean) >= RANGE_MULT) & (body_ratio >= 0.60)
    imbalance.loc[imb_gate & (out['close'] > out['open'])] = 1
    imbalance.loc[imb_gate & (out['close'] < out['open'])] = -1
    absorption = pd.Series(0, index=out.index, dtype='int8')
    long_lower = high_volume & (lower >= body_safe * WICK_RATIO) & (lower > upper)
    long_upper = high_volume & (upper >= body_safe * WICK_RATIO) & (upper > lower)
    absorption.loc[long_lower] = 1
    absorption.loc[long_upper] = -1
    out['bp_imbalance'] = imbalance
    out['bp_absorption'] = absorption
    return out


def run_tf(df: pd.DataFrame, signal: pd.Series, tf: str, initial_balance: float, lot: float, contract_size: float, cost: float):
    horizon = 5  # same five-bar fixed horizon per TF; formula not retuned.
    qty = lot * contract_size
    trades = []
    next_free = 0
    sig = signal.to_numpy()
    for idx in np.flatnonzero(sig != 0):
        if idx < next_free or idx + 1 + horizon >= len(df):
            continue
        direction = int(sig[idx])
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon - 1
        entry = float(df.open.iat[entry_idx])
        exit_ = float(df.close.iat[exit_idx])
        pnl = direction * (exit_ - entry) * qty - cost
        trades.append((tf, idx, entry_idx, exit_idx, direction, entry, exit_, pnl, df.datetime.iat[entry_idx]))
        next_free = exit_idx + 1
    return trades


def metrics(trades: pd.DataFrame, initial_balance: float):
    if trades.empty:
        return dict(N=0,N_per_day=0.0,WR=0.0,PF=0.0,RF=0.0,net_profit=0.0,return_pct=0.0,max_dd_pct=0.0,max_dd_usd=0.0,final_balance=initial_balance)
    t = trades.sort_values('entry_time').reset_index(drop=True)
    gp = float(t.loc[t.pnl > 0, 'pnl'].sum())
    gl = float(-t.loc[t.pnl < 0, 'pnl'].sum())
    net = float(t.pnl.sum())
    pf = gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0)
    wr = float((t.pnl > 0).mean() * 100)
    eq = initial_balance + t.pnl.cumsum()
    peak = pd.concat([pd.Series([initial_balance]), eq], ignore_index=True).cummax().iloc[1:].to_numpy()
    dd = peak - eq.to_numpy()
    maxdd = float(dd.max()) if len(dd) else 0.0
    ddpct = np.where(peak > 0, dd / peak * 100.0, 0.0)
    maxddpct = float(ddpct.max()) if len(ddpct) else 0.0
    rf = net / maxdd if maxdd > 0 else (math.inf if net > 0 else 0.0)
    return dict(N=int(len(t)),N_per_day=float(len(t)/21.0),WR=wr,PF=pf,RF=rf,net_profit=net,return_pct=net/initial_balance*100.0,max_dd_pct=maxddpct,max_dd_usd=maxdd,final_balance=initial_balance+net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-07-27')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--initial-balance', type=float, default=1000.0)
    ap.add_argument('--lot', type=float, default=0.01)
    ap.add_argument('--contract-size', type=float, default=100.0)
    ap.add_argument('--round-trip-cost-usd', type=float, default=0.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = business_days(dt.date.fromisoformat(args.start), 21)
    ticks = load_ticks(days, args.workers)
    all_rows = []
    all_trade_frames = []
    for tf, sec in TF_SECONDS.items():
        bars = build_edges(build_bars_from_raw_ticks(ticks, sec))
        for edge, col in [('IMBALANCE','bp_imbalance'),('ABSORPTION','bp_absorption')]:
            tr = run_tf(bars, bars[col], tf, args.initial_balance, args.lot, args.contract_size, args.round_trip_cost_usd)
            td = pd.DataFrame(tr, columns=['tf','signal_idx','entry_idx','exit_idx','direction','entry','exit','pnl','entry_time'])
            if not td.empty:
                td['edge'] = edge
                all_trade_frames.append(td)
            m = metrics(td, args.initial_balance)
            all_rows.append({'scope':'TF_EDGE','tf':tf,'edge':edge,'business_days':21,'raw_tick_to_bar':True,'ohlc_resample_used':False,**m})
    combined = pd.concat(all_trade_frames, ignore_index=True) if all_trade_frames else pd.DataFrame(columns=['pnl','entry_time'])
    cm = metrics(combined, args.initial_balance)
    all_rows.append({'scope':'ALL_5TF_2EDGE','tf':'ALL','edge':'BOTH_INDEPENDENT','business_days':21,'raw_tick_to_bar':True,'ohlc_resample_used':False,**cm})
    pd.DataFrame(all_rows).to_csv(OUT/'summary_21d.csv', index=False)
    combined.to_csv(OUT/'trades_all.csv', index=False)
    (OUT/'provenance.json').write_text(json.dumps({'start':days[0].isoformat(),'end':days[-1].isoformat(),'business_days':[d.isoformat() for d in days],'formula_policy':'FROZEN_BIGPLAYER_2EDGE','volume_input':'sum(bid_size+ask_size) per direct raw-tick bar','execution':'next bar open, fixed 5 bars, non-overlap within each TF-edge, independent across TF-edge','cost_usd':args.round_trip_cost_usd}, indent=2), encoding='utf-8')
    print(pd.DataFrame(all_rows).to_string(index=False))

if __name__ == '__main__':
    main()
