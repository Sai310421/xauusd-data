#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amos_reversal_v1_v4_rawtick_bt_v2 as m
import amos_reversal_v8_ablation_m1_standard as v8

TF='M1'; MODE='STANDARD'; RR=2.0
STAGES=['S1_SEQUENCE','S3_POI_RETRACE']
VARIANTS=['CURRENT','FLIP_BOTH','FLIP_BULL_ONLY','FLIP_BEAR_ONLY']

def epoch_ns(values):
    return np.fromiter((pd.Timestamp(x).value for x in values), dtype=np.int64, count=len(values))

def exec_side(signal_side:int, variant:str)->int:
    if variant=='CURRENT': return signal_side
    if variant=='FLIP_BOTH': return -signal_side
    if variant=='FLIP_BULL_ONLY': return -signal_side if signal_side==1 else signal_side
    if variant=='FLIP_BEAR_ONLY': return -signal_side if signal_side==-1 else signal_side
    raise ValueError(variant)

def entry_index(stage,c,tv,bid,ask):
    if stage=='S1_SEQUENCE':
        return int(np.searchsorted(tv,np.int64(pd.Timestamp(c['mss_time']).value),side='right'))
    if c['disp_i'] is None or c['zone'] is None: return None
    lo,hi,_=c['zone']; st=pd.Timestamp(c['disp_time']); en=st+pd.Timedelta(minutes=6)
    return v8.first_touch(tv,bid,ask,st.value,en.value,c['side'],lo,hi)

def simulate_one(t,z,cands,stage,variant):
    tv=epoch_ns(t['time']); bid=t.bid.to_numpy(float,copy=False); ask=t.ask.to_numpy(float,copy=False)
    trades=[]; last_exit=np.int64(-1)
    diag={'candidates':len(cands),'entry_candidates':0,'accepted':0,'executed':0,'overlap':0}
    for c in cands:
        k=entry_index(stage,c,tv,bid,ask)
        if k is None or k>=len(t): continue
        diag['entry_candidates']+=1
        touch=pd.Timestamp(t.time.iloc[k])
        if touch.value<=last_exit:
            diag['overlap']+=1; continue
        signal_side=int(c['side']); side=exec_side(signal_side,variant)
        entry=float(ask[k] if side==1 else bid[k])
        ref_i=c['mss_i'] if stage=='S1_SEQUENCE' else c['disp_i']
        av=float(z.atr.iloc[ref_i])
        risk=0.20*av
        if not np.isfinite(risk) or risk<=0: continue
        stop=entry-side*risk; target=entry+side*RR*risk
        horizon=touch+pd.Timedelta(minutes=20)
        end=min(int(np.searchsorted(tv,np.int64(horizon.value),side='right')),len(t))
        if end<=k+1: continue
        exit_px=None; exit_t=None; rval=None; result='TIME'
        for j in range(k+1,end):
            px=float(bid[j] if side==1 else ask[j])
            if (side==1 and px<=stop) or (side==-1 and px>=stop):
                exit_px=px; exit_t=pd.Timestamp(t.time.iloc[j]); rval=side*(px-entry)/risk; result='LOSS'; break
            if (side==1 and px>=target) or (side==-1 and px<=target):
                exit_px=px; exit_t=pd.Timestamp(t.time.iloc[j]); rval=side*(px-entry)/risk; result='WIN'; break
        if exit_px is None:
            j=end-1; exit_px=float(bid[j] if side==1 else ask[j]); exit_t=pd.Timestamp(t.time.iloc[j]); rval=side*(exit_px-entry)/risk
        tr=m.Trade(f'V9_{variant}',TF,MODE,side,str(touch),str(exit_t),entry,stop,target,exit_px,float(rval),result,
                   f'SIGNAL_SIDE_{signal_side}',float(c['score']),0.0,0.0)
        trades.append(tr); last_exit=np.int64(exit_t.value); diag['accepted']+=1; diag['executed']+=1
    return trades,diag

def extended_metrics(trades):
    x=m.metrics(trades); rs=np.array([t.r for t in trades],float)
    pos=rs[rs>0]; neg=rs[rs<0]
    x['avg_win_R']=float(pos.mean()) if len(pos) else 0.0
    x['avg_loss_R']=float(neg.mean()) if len(neg) else 0.0
    x['median_R']=float(np.median(rs)) if len(rs) else 0.0
    return x

def main():
    src=m.resolve_source('XAUUSD','dukascopy_raw'); t=m.load_ticks(src); z=m.feat(m.bars(t,TF)); cands=v8.detect_candidates(z)
    out=Path(os.environ.get('AMOS_V9_OUT','results/amos-v9-polarity')); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for stage in STAGES:
        for variant in VARIANTS:
            tr,diag=simulate_one(t,z,cands,stage,variant); met=extended_metrics(tr)
            met.update({'stage':stage,'variant':variant,'diagnostics':diag})
            rows.append(met)
            pd.DataFrame([asdict(x) for x in tr]).to_csv(out/f'{stage}_{variant}_trades.csv',index=False)
            (out/f'{stage}_{variant}_metrics.json').write_text(json.dumps(met,indent=2),encoding='utf-8')
    flat=[]
    for r in rows:
        flat.append({k:v for k,v in r.items() if k!='diagnostics'} | {f'diag_{k}':v for k,v in r['diagnostics'].items()})
    pd.DataFrame(flat).to_csv(out/'summary.csv',index=False)
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    manifest={
        'purpose':'Direction polarity isolation', 'symbol':'XAUUSD','timeframe':'M1','mode':'STANDARD','rr':RR,
        'raw_tick_source':str(src),'verification_level':'RAW_DUKASCOPY_BIDASK','stages':STAGES,'variants':VARIANTS,
        'entry_clock':'identical per stage before polarity transform','risk_model':'symmetric 0.20 ATR to isolate direction',
        'horizon_minutes':20,'bars_label':'right','bars_closed':'right','rows':len(t)
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
