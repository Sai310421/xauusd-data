#!/usr/bin/env python3
"""Frozen, small-candidate XAU tick research comparison. Never authorizes trading."""
import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

from xau_tick_research import run, timestamp


# Fixed before inspecting the OOS segment. Changes require a new experiment ID.
CANDIDATES = (
    {'name':'fixed_20','signal_mode':'fixed_momentum','lookback':20,'threshold_usd':.5},
    {'name':'trend_20','signal_mode':'adaptive_trend','lookback':20,'threshold_usd':.5,
     'vol_k':1.5,'min_efficiency':.45,'spread_budget_fraction':.25},
    {'name':'trend_40','signal_mode':'adaptive_trend','lookback':40,'threshold_usd':.7,
     'vol_k':1.5,'min_efficiency':.55,'spread_budget_fraction':.2},
)
FIELDS = ('symbol','available_at','bid','ask')


def split(path, calibration_end, oos_start, cal_path, oos_path):
    cal_end, start=timestamp(calibration_end),timestamp(oos_start)
    if cal_end>=start:raise ValueError('OOS start must follow calibration end; gap required')
    counts={'calibration':0,'embargo':0,'oos':0}
    prior=None
    with open(path,encoding='utf8',newline='') as f, open(cal_path,'w',encoding='utf8',newline='') as c, open(oos_path,'w',encoding='utf8',newline='') as o:
        reader=csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):raise ValueError('missing tick columns')
        wc,wo=csv.DictWriter(c,fieldnames=FIELDS),csv.DictWriter(o,fieldnames=FIELDS)
        wc.writeheader();wo.writeheader()
        for row in reader:
            at=timestamp(row['available_at'])
            if prior is not None and at<prior:raise ValueError('unsorted input')
            prior=at
            if row['symbol']!='XAUUSD':raise ValueError('XAUUSD only')
            if at<=cal_end:wc.writerow({k:row[k] for k in FIELDS});counts['calibration']+=1
            elif at<start:counts['embargo']+=1
            else:wo.writerow({k:row[k] for k in FIELDS});counts['oos']+=1
    return counts


def evaluate(path, calibration_end, oos_start, source, capital=1000.,
             lot=.01, contract_oz=100., commission_round_usd=0.,
             min_cal_trades=60, min_oos_trades=60,
             slippage_usd_per_side=.1):
    if not source or source.lower() in ('synthetic','unknown','example'):
        raise ValueError('specific external tick data source required')
    if min_cal_trades<1 or min_oos_trades<1:raise ValueError('positive trade minimum required')
    with open(path,'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        cal,oos=Path(tmp)/'cal.csv',Path(tmp)/'oos.csv'
        counts=split(path,calibration_end,oos_start,cal,oos)
        if min(counts['calibration'],counts['oos'])<41:
            raise ValueError('insufficient ticks in calibration or OOS')
        shared=dict(capital=capital,lot=lot,contract_oz=contract_oz,
                    commission_round_usd=commission_round_usd,
                    slippage_usd_per_side=slippage_usd_per_side,
                    execution_delay_ticks=1)
        trials=[]
        for candidate in CANDIDATES:
            report=run(cal,**shared,**{k:v for k,v in candidate.items() if k!='name'})
            trials.append({'candidate':candidate['name'],'closed_trades':report['closed_trades'],
                           'profit_factor':report['profit_factor'],
                           'net_pnl':report['cash']-capital,'max_marked_equity_dd':report['max_marked_equity_dd']})
        eligible=[t for t in trials if t['closed_trades']>=min_cal_trades and
                  t['profit_factor'] is not None and t['profit_factor']>1]
        chosen=max(eligible,key=lambda t:(t['profit_factor'],t['net_pnl'])) if eligible else None
        if chosen is None:oos_result=None;stress_result=None
        else:
            spec=next(c for c in CANDIDATES if c['name']==chosen['candidate'])
            oos_result=run(oos,**shared,**{k:v for k,v in spec.items() if k!='name'})
            stress={**shared,'slippage_usd_per_side':slippage_usd_per_side*2+.1,
                    'commission_round_usd':commission_round_usd*2+1.}
            stress_result=run(oos,**stress,**{k:v for k,v in spec.items() if k!='name'})
    oos_pass=bool(oos_result and oos_result['closed_trades']>=min_oos_trades and
                  oos_result['profit_factor'] is not None and oos_result['profit_factor']>=1.5 and
                  oos_result['max_marked_equity_dd']<=.035 and not oos_result['risk_halted'] and
                  stress_result['profit_factor'] is not None and stress_result['profit_factor']>1 and
                  stress_result['cash']>capital and not stress_result['risk_halted'])
    return {'classification':'UNVERIFIED_EXTERNAL_TICK_REPLAY',
            'source_claim':source,'source_sha256':digest,'split_counts':counts,
            'calibration_end':calibration_end,'oos_start':oos_start,
            'candidate_set':'FROZEN_THREE_V1','calibration_trials':trials,
            'selected':chosen['candidate'] if chosen else None,
            'oos':oos_result,'cost_stress_oos':stress_result,
            'cost_stress_assumptions':{'slippage_usd_per_side':slippage_usd_per_side*2+.1,
                                       'commission_round_usd':commission_round_usd*2+1.},
            'gate':'REVIEW_EXECUTION_AND_DATA' if oos_pass else 'BLOCKED',
            'note':'OOS was not used for selection. Tick source, CFD execution, latency, margin and independent rerun still require verification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ticks',required=True)
    parser.add_argument('--calibration-end',required=True)
    parser.add_argument('--oos-start',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--capital',type=float,default=1000.)
    parser.add_argument('--lot',type=float,default=.01)
    parser.add_argument('--contract-oz',type=float,default=100.)
    parser.add_argument('--commission-round-usd',type=float,default=0.)
    parser.add_argument('--slippage-usd-per-side',type=float,default=.1)
    parser.add_argument('--min-cal-trades',type=int,default=60)
    parser.add_argument('--min-oos-trades',type=int,default=60)
    a=parser.parse_args()
    result=evaluate(a.ticks,a.calibration_end,a.oos_start,a.source,capital=a.capital,
                    lot=a.lot,contract_oz=a.contract_oz,
                    commission_round_usd=a.commission_round_usd,
                    min_cal_trades=a.min_cal_trades,min_oos_trades=a.min_oos_trades,
                    slippage_usd_per_side=a.slippage_usd_per_side)
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='oos'},indent=2))
