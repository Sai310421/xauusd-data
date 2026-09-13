from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_liquidity_cycle_v6 as v6

CONTRACT_SIZE_OZ = 100.0
LOT = 0.01

# Rebate evidence status:
# - TariTali Zero XAUUSD $1.35/lot: published in TariTali Exness-Zero-AutoRebate PDF.
# - PaybackFX XAUUSD $2.43750/lot: published in PaybackFX Exness symbol-rate PDF.
#   The PDF excerpt does not independently prove separate Raw-vs-Zero mapping, so it is modeled
#   as a sensitivity scenario on both account types and tagged accordingly.
# - No unverified TariTali Raw XAUUSD value is hard-coded here.

SCENARIOS = [
    {'name':'EXNESS_RAW_NO_CB_BASE','account':'RAW','commission_per_lot_side':3.5,'spread_usd':0.10,'slippage_usd':0.02,'rebate_per_lot_rt':0.0,'evidence':'broker_cost_model'},
    {'name':'EXNESS_ZERO_NO_CB_BASE','account':'ZERO','commission_per_lot_side':6.0,'spread_usd':0.02,'slippage_usd':0.02,'rebate_per_lot_rt':0.0,'evidence':'broker_cost_model'},
    {'name':'EXNESS_ZERO_TARITALI_1_35','account':'ZERO','commission_per_lot_side':6.0,'spread_usd':0.02,'slippage_usd':0.02,'rebate_per_lot_rt':1.35,'evidence':'taritali_official_pdf_xauusd_zero'},
    {'name':'EXNESS_RAW_PAYBACKFX_2_4375_SENS','account':'RAW','commission_per_lot_side':3.5,'spread_usd':0.10,'slippage_usd':0.02,'rebate_per_lot_rt':2.4375,'evidence':'paybackfx_symbol_pdf_account_mapping_not_independently_proven'},
    {'name':'EXNESS_ZERO_PAYBACKFX_2_4375_SENS','account':'ZERO','commission_per_lot_side':6.0,'spread_usd':0.02,'slippage_usd':0.02,'rebate_per_lot_rt':2.4375,'evidence':'paybackfx_symbol_pdf_account_mapping_not_independently_proven'},
]


def commission_price_equiv(c):
    qty_oz = CONTRACT_SIZE_OZ * LOT
    return (2.0 * c * LOT) / qty_oz


def rebate_price_equiv(rebate_per_lot_rt):
    qty_oz = CONTRACT_SIZE_OZ * LOT
    return (rebate_per_lot_rt * LOT) / qty_oz


def all_in_cost(s):
    return max(0.0, s['spread_usd'] + commission_price_equiv(s['commission_per_lot_side']) + 2.0 * s['slippage_usd'] - rebate_price_equiv(s['rebate_per_lot_rt']))


def first_passage_mid(ticks, start_idx, side, seconds, dist, cost=0.0):
    entry = v6.mid(ticks[start_idx])
    end = int(ticks[start_idx].ts_event) + int(seconds * 1e9)
    mfe = -cost
    mae = cost
    for j in range(start_idx + 1, len(ticks)):
        if int(ticks[j].ts_event) > end:
            break
        raw = (v6.mid(ticks[j]) - entry) * side
        net = raw - cost
        mfe = max(mfe, net)
        mae = max(mae, -net)
        if net >= dist:
            return 1, mfe, mae
        if -net >= dist:
            return -1, mfe, mae
    return 0, mfe, mae


def summarize(stats, horizons, dists):
    rows = []
    for h in horizons:
        for d in dists:
            s = stats[(h, d)]
            dec = s['w'] + s['l']
            n = dec + s['none']
            rows.append({
                'horizon_s': h,
                'distance': d,
                'wins_first': s['w'],
                'losses_first': s['l'],
                'none': s['none'],
                'first_passage_wr': s['w'] / dec if dec else None,
                'coverage': dec / n if n else 0.0,
                'avg_mfe': s['mfe'] / n if n else 0.0,
                'avg_mae': s['mae'] / n if n else 0.0,
            })
    return rows


def run(catalog, max_ticks=3000000):
    ticks = v6.load_ticks(catalog, max_ticks)
    b15 = v6.bars_n(ticks, 15)
    variants = ['V63_ARM_BASELINE','V63_AE05_SECOND','V63_AE10_SECOND','V63_FINAL_RETEST','V63_INVERSE_ARM']
    horizons = [1,3,5,10,15,30]
    dists = [0.10,0.20,0.30,0.50,0.75,1.00]

    def zstats():
        return {(h,d):{'w':0,'l':0,'none':0,'mfe':0.0,'mae':0.0} for h in horizons for d in dists}

    out = {v:{'arms':0,'entries':0,'latencies':[],'adverse':[],'struct':zstats(),
              'broker':{s['name']:zstats() for s in SCENARIOS}} for v in variants}

    for i in range(15, len(b15)-1):
        if not v6.accumulation_ok(b15, i):
            continue
        A,S,H,D = b15[i-3],b15[i-2],b15[i-1],b15[i]
        side = v6.initial_sweep_side(A,S)
        if not side:
            continue
        hm = v6.opposite_harvest_meta(A,S,H,side)
        if not hm['broad']:
            continue
        arm_idx,_ = v6.find_arm(ticks,D,side,hm['return_level'])
        if arm_idx is None or arm_idx >= len(ticks)-2:
            continue

        for name in variants:
            o = out[name]
            o['arms'] += 1
            entry_idx = None
            trade_side = side
            meta = {'max_adverse':0.0,'latency_s':0.0}
            if name == 'V63_ARM_BASELINE':
                entry_idx = arm_idx
            elif name == 'V63_INVERSE_ARM':
                entry_idx = arm_idx
                trade_side = -side
            elif name == 'V63_AE05_SECOND':
                entry_idx,meta = v6.seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.05,False)
            elif name == 'V63_AE10_SECOND':
                entry_idx,meta = v6.seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.10,False)
            elif name == 'V63_FINAL_RETEST':
                entry_idx,meta = v6.seek_second_entry(ticks,arm_idx,side,hm['return_level'],hm['harvest_extreme'],0.05,True)
            if entry_idx is None:
                continue

            o['entries'] += 1
            o['latencies'].append(meta.get('latency_s',0.0))
            o['adverse'].append(meta.get('max_adverse',0.0))

            for h in horizons:
                for d in dists:
                    r,mfe,mae = first_passage_mid(ticks,entry_idx,trade_side,h,d,0.0)
                    s = o['struct'][(h,d)]
                    if r > 0: s['w'] += 1
                    elif r < 0: s['l'] += 1
                    else: s['none'] += 1
                    s['mfe'] += mfe; s['mae'] += mae

                    for sc in SCENARIOS:
                        cost = all_in_cost(sc)
                        r,mfe,mae = first_passage_mid(ticks,entry_idx,trade_side,h,d,cost)
                        s = o['broker'][sc['name']][(h,d)]
                        if r > 0: s['w'] += 1
                        elif r < 0: s['l'] += 1
                        else: s['none'] += 1
                        s['mfe'] += mfe; s['mae'] += mae

    result = []
    for name in variants:
        o = out[name]
        broker = {}
        for sc in SCENARIOS:
            broker[sc['name']] = {
                'assumptions': {
                    **sc,
                    'roundtrip_commission_price_equiv': commission_price_equiv(sc['commission_per_lot_side']),
                    'rebate_price_equiv': rebate_price_equiv(sc['rebate_per_lot_rt']),
                    'all_in_cost_price_equiv': all_in_cost(sc),
                },
                'first_passage': summarize(o['broker'][sc['name']], horizons, dists),
            }
        result.append({
            'variant':name,
            'arms':o['arms'],
            'entries':o['entries'],
            'conversion':o['entries']/o['arms'] if o['arms'] else 0.0,
            'latency_s':v6.summarize_latency(o['latencies']),
            'adverse_before_entry':{
                'mean':sum(o['adverse'])/len(o['adverse']) if o['adverse'] else None,
                'median':sorted(o['adverse'])[len(o['adverse'])//2] if o['adverse'] else None,
            },
            'structural_first_passage':summarize(o['struct'],horizons,dists),
            'exness_rebate_sensitivity':broker,
        })

    return {
        'status':'CAUSAL_V63_EXNESS_REBATE_SENSITIVITY_NOT_EXACT_EXNESS_TICK_REPLAY',
        'raw_source':'Existing Raw Bid/Ask catalog drives state detection; Exness execution/rebate is a cost-sensitivity layer, not empirical Exness historical tick replay.',
        'ticks':len(ticks),
        'bars15':len(b15),
        'lot':LOT,
        'contract_size_oz':CONTRACT_SIZE_OZ,
        'rebate_evidence':{
            'taritali_zero_xauusd_usd_per_lot_rt':1.35,
            'paybackfx_xauusd_usd_per_lot_rt':2.4375,
            'paybackfx_note':'Published XAUUSD symbol rate; Raw-vs-Zero mapping is not independently proven by the symbol PDF excerpt, therefore sensitivity-only.',
            'taritali_raw_note':'No unverified Raw value hard-coded.'
        },
        'cost_formula':'net_mid_move = directional_mid_move - spread - roundtrip_commission_price_equiv - 2*slippage + rebate_price_equiv',
        'sequence':'Accumulation -> InitialSweep -> OppositeHarvest -> ARM -> adverse excursion -> recovery/final retest -> second confirmation -> entry',
        'variants':result,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--max-ticks', type=int, default=3000000)
    a = ap.parse_args()
    res = run(a.catalog, a.max_ticks)
    d = Path('results/dexg-amd15') / a.experiment_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / 'AMD_LIQUIDITY_CYCLE_V63_EXNESS_REBATE.json'
    p.write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
