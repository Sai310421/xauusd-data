from __future__ import annotations

import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from nautilus_trader.persistence.catalog import ParquetDataCatalog

POINT = 0.001          # USDJPY point under the current Dukascopy catalog precision
GRID = 0.100           # CRYSTAL observed OrderDistance=100 points
TP = 0.050             # CRYSTAL observed TakeProfitPoints=50
MAX_POS = 55

@dataclass
class Pos:
    side: int           # +1 buy, -1 sell
    entry: float
    ts: int


def px_float(x):
    return float(x.as_double()) if hasattr(x, 'as_double') else float(x)


def pnl_points(pos: Pos, bid: float, ask: float) -> float:
    close = bid if pos.side > 0 else ask
    return pos.side * (close - pos.entry) / POINT


def threshold_now(base: float, policy: str, gross: int, mids: list[float]) -> float:
    if policy == 'absolute':
        return base
    if policy == 'per_position':
        # Normalize the tolerated absolute debt by inventory size; avoids treating 5 and 50 positions equally.
        return base * max(1.0, math.sqrt(max(gross, 1) / 10.0))
    if policy == 'adaptive':
        if len(mids) < 300:
            return base
        v_short = abs(mids[-1] - mids[-60])
        v_long = abs(mids[-1] - mids[-300]) + 1e-12
        persistence = min(2.0, v_short / v_long * 5.0)
        # Persistent adverse movement narrows the boundary; choppy/ranging movement widens it.
        return base * float(np.clip(1.20 - 0.30 * persistence, 0.65, 1.35))
    raise ValueError(policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--policy', choices=['absolute','per_position','adaptive'], required=True)
    ap.add_argument('--threshold', type=float, required=True)
    ap.add_argument('--max-ticks', type=int, default=0, help='0 = all ticks')
    args = ap.parse_args()

    root = Path(args.catalog)
    manifest = json.loads((root/'catalog_manifest.json').read_text())
    if manifest.get('data_kind') != 'RAW_BIDASK' or manifest.get('ohlc_resample_used') is not False:
        raise SystemExit('FAIL-CLOSED: RAW_BIDASK catalog required and OHLC resample forbidden')

    catalog = ParquetDataCatalog(str(root))
    inst = next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','') == 'USDJPY'), None)
    if inst is None:
        raise SystemExit('FAIL-CLOSED: USDJPY instrument missing')
    ticks = catalog.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:
        raise SystemExit('FAIL-CLOSED: no USDJPY QuoteTicks')
    if args.max_ticks > 0:
        ticks = ticks[:args.max_ticks]

    positions: list[Pos] = []
    mids: list[float] = []
    realized = 0.0
    peak_equity = 0.0
    max_dd = 0.0
    next_buy = next_sell = None
    locked = False
    lock_ts = None
    lock_debt = 0.0
    recovery_profit = 0.0
    recovery_done_ts = None
    lock_gross = 0
    lock_threshold = None
    natural_recovery_after_lock = False
    orig_positions_at_lock: list[Pos] = []
    orig_lock_mark = None
    orig_recovery_ts = None
    events = []

    # Independent post-lock recovery harvester. It is intentionally simple and isolated from the locked basket.
    rec_positions: list[Pos] = []
    rec_next_buy = rec_next_sell = None

    for i, t in enumerate(ticks):
        bid, ask = px_float(t.bid_price), px_float(t.ask_price)
        mid = (bid + ask) * 0.5
        ts = int(t.ts_event)
        mids.append(mid)
        if len(mids) > 600:
            mids.pop(0)

        if next_buy is None:
            next_buy = mid - GRID
            next_sell = mid + GRID

        # Counterfactual: after lock, ask whether the original frozen basket would naturally return to lock mark/economic BE.
        if locked and orig_recovery_ts is None and orig_positions_at_lock:
            cf = sum(pnl_points(p, bid, ask) for p in orig_positions_at_lock)
            if cf >= orig_lock_mark:
                orig_recovery_ts = ts
                natural_recovery_after_lock = True

        if not locked:
            # TP recycle.
            keep = []
            for p in positions:
                pp = pnl_points(p, bid, ask)
                if pp >= TP / POINT:
                    realized += pp
                else:
                    keep.append(p)
            positions = keep

            # Symmetric mixed-grid reconstruction candidate. This is a parity hypothesis, not claimed original source logic.
            while len(positions) < MAX_POS and mid <= next_buy:
                positions.append(Pos(+1, ask, ts)); next_buy -= GRID
            while len(positions) < MAX_POS and mid >= next_sell:
                positions.append(Pos(-1, bid, ts)); next_sell += GRID

            floating = sum(pnl_points(p, bid, ask) for p in positions)
            equity = realized + floating
            peak_equity = max(peak_equity, equity)
            max_dd = max(max_dd, peak_equity - equity)
            th = threshold_now(args.threshold, args.policy, len(positions), mids)

            if positions and floating <= -th:
                locked = True
                lock_ts = ts
                lock_debt = -floating
                lock_gross = len(positions)
                lock_threshold = th
                orig_positions_at_lock = [Pos(p.side,p.entry,p.ts) for p in positions]
                orig_lock_mark = floating
                events.append({'type':'FULL_HEDGE_LOCK','ts':ts,'debt_points':lock_debt,'gross_positions':lock_gross,'effective_threshold':th})
                rec_next_buy = mid - GRID
                rec_next_sell = mid + GRID
                # Full hedge semantics: subsequent directional PnL of locked basket is frozen; costs are accounted in later broker-reality layer.
        else:
            # Independent recovery harvester: new inventory only; locked basket itself is untouched.
            keep = []
            for p in rec_positions:
                pp = pnl_points(p, bid, ask)
                if pp >= TP / POINT:
                    recovery_profit += pp
                else:
                    keep.append(p)
            rec_positions = keep
            while len(rec_positions) < 10 and mid <= rec_next_buy:
                rec_positions.append(Pos(+1, ask, ts)); rec_next_buy -= GRID
            while len(rec_positions) < 10 and mid >= rec_next_sell:
                rec_positions.append(Pos(-1, bid, ts)); rec_next_sell += GRID
            if recovery_done_ts is None and recovery_profit >= lock_debt:
                recovery_done_ts = ts
                events.append({'type':'ECONOMIC_BE','ts':ts,'recovery_profit_points':recovery_profit})

    recovery_seconds = None
    if lock_ts is not None and recovery_done_ts is not None:
        recovery_seconds = (recovery_done_ts - lock_ts) / 1e9
    natural_seconds = None
    if lock_ts is not None and orig_recovery_ts is not None:
        natural_seconds = (orig_recovery_ts - lock_ts) / 1e9

    result = {
        'verification_level':'RAW_BIDASK_LOCK_FRONTIER_RECONSTRUCTION',
        'strategy_status':'CRYSTAL behavior reconstruction candidate; not source-code parity proven',
        'ohlc_resample_used':False,
        'symbol':'USDJPY',
        'period':{'start':manifest.get('start'),'days':manifest.get('days'),'end_exclusive':manifest.get('end_exclusive')},
        'policy':args.policy,'base_threshold_points':args.threshold,
        'raw_ticks':len(ticks),'locked':locked,'lock_debt_points':lock_debt,'lock_gross_positions':lock_gross,
        'effective_lock_threshold_points':lock_threshold,
        'recovery_profit_points':recovery_profit,'economic_be_recovered':recovery_done_ts is not None,
        'recovery_seconds':recovery_seconds,'natural_recovery_after_lock':natural_recovery_after_lock,
        'natural_recovery_seconds':natural_seconds,'max_dd_points_prelock':max_dd,
        'events':events,
        'score_components':{
            'recovery_ease': (1.0/(1.0+lock_debt)) if locked else 0.0,
            'natural_recovery_preserved': 1.0 if not locked else 0.0,
            'false_lock_flag': bool(locked and natural_recovery_after_lock and (natural_seconds is not None and natural_seconds <= 3600)),
        },
        'limitations':[
            'GridMode=2/TradeSide=2 mapping remains unproven; symmetric mixed-grid is a reconstruction hypothesis.',
            'Lock is modeled as full directional freeze; commission/slippage are not yet injected in this frontier worker.',
            'Independent recovery harvester is deliberately separated from the locked basket and must be replaced/validated against the final recovery engine.'
        ]
    }
    out = Path('results/crystal-x-v10')/args.experiment_id/f'{args.policy}-{int(args.threshold)}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()
