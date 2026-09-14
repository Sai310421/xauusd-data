#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amos_reversal_v1_v4_rawtick_bt_v2 as m
import amos_reversal_v1_v4_rawtick_bt_v6_true_mtf as v6

TF='M1'; MODE='STANDARD'; RR=2.0
STAGES=['S1_SEQUENCE','S2_DISPLACEMENT','S3_POI_RETRACE','S4_TRUE_MTF','S5_CONTEXT']

def epoch_ns(values):
    return np.fromiter((pd.Timestamp(x).value for x in values), dtype=np.int64, count=len(values))

def detect_candidates(z):
    out=[]; states={1:None,-1:None}
    for i in range(30,len(z)-1):
        r=z.iloc[i]; p=z.iloc[i-1]
        if not np.isfinite(r.atr) or r.atr<=0: continue
        for side in (1,-1):
            s=states[side]
            if s and i-s['sweep_i']>28: states[side]=None
        bull=bool(r.low<r.swing_lo and r.close>r.swing_lo)
        bear=bool(r.high>r.swing_hi and r.close<r.swing_hi)
        if bull:
            states[1]={'sweep_i':i,'extreme':float(r.low),'mss_ref':float(z.high.iloc[max(0,i-3):i].max()),'cisd_i':None}
        if bear:
            states[-1]={'sweep_i':i,'extreme':float(r.high),'mss_ref':float(z.low.iloc[max(0,i-3):i].min()),'cisd_i':None}
        for side in (1,-1):
            s=states[side]
            if not s: continue
            if s['cisd_i'] is None:
                if i>s['sweep_i'] and i-s['sweep_i']<=8:
                    cisd=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
                    if cisd: s['cisd_i']=i
                continue
            if i<=s['cisd_i']: continue
            if i-s['cisd_i']>12:
                states[side]=None; continue
            mss=(r.close>s['mss_ref']) if side==1 else (r.close<s['mss_ref'])
            if not mss: continue
            cand={'side':side,'sweep_i':s['sweep_i'],'cisd_i':s['cisd_i'],'mss_i':i,'extreme':float(s['extreme']),
                  'mss_time':pd.Timestamp(r.time),'disp_i':None,'zone':None,'score':3.2}
            # Displacement may confirm on MSS bar or during the next four completed bars.
            for j in range(i,min(i+5,len(z)-1)):
                q=z.iloc[j]; prev=z.iloc[j-1]
                if np.isfinite(q.disp) and float(q.disp)>=0.65:
                    cand['disp_i']=j; cand['disp_time']=pd.Timestamp(q.time)
                    zone=v6._poi_zone(z,j,side)
                    cand['zone']=zone
                    vel=min(1.5,abs(float(q.close)-float(prev.close))/(float(q.atr)+1e-12))
                    poi_type=zone[2] if zone else ''
                    cand['score']=4.2+0.8*(poi_type=='FVG')+1.0*(poi_type=='IFVG')+1.2*(poi_type=='BPR')+0.35*vel
                    break
            out.append(cand); states[side]=None
    return out

def first_touch(tv,bid,ask,start_ns,end_ns,side,lo,hi):
    a=int(np.searchsorted(tv,np.int64(start_ns),side='right')); b=int(np.searchsorted(tv,np.int64(end_ns),side='right'))
    if b<=a: return None
    px=ask[a:b] if side==1 else bid[a:b]
    hit=np.flatnonzero((px>=lo)&(px<=hi))
    return None if hit.size==0 else a+int(hit[0])

def trade_from_tick(t,tv,bid,ask,z,cand,k,stage,last_exit_ns):
    q=t.iloc[k]; touch=pd.Timestamp(q.time); side=cand['side']
    if touch.value<=last_exit_ns: return None,last_exit_ns,'overlap'
    entry=float(ask[k] if side==1 else bid[k]); ref_i=cand['disp_i'] if cand['disp_i'] is not None else cand['mss_i']
    av=float(z.atr.iloc[ref_i]); stop=min(cand['extreme'],entry-.20*av) if side==1 else max(cand['extreme'],entry+.20*av)
    risk=abs(entry-stop)
    if not np.isfinite(risk) or risk<=0: return None,last_exit_ns,'risk'
    target=entry+side*RR*risk; horizon=touch+pd.Timedelta(minutes=20)
    end=int(np.searchsorted(tv,np.int64(horizon.value),side='right')); end=min(end,len(t))
    if end<=k+1: return None,last_exit_ns,'horizon'
    exit_px=None; exit_t=None; rval=None; result='TIME'
    for j in range(k+1,end):
        px=float(bid[j] if side==1 else ask[j])
        if (side==1 and px<=stop) or (side==-1 and px>=stop):
            exit_px=px; exit_t=pd.Timestamp(t.time.iloc[j]); rval=side*(px-entry)/risk; result='LOSS'; break
        if (side==1 and px>=target) or (side==-1 and px<=target):
            exit_px=px; exit_t=pd.Timestamp(t.time.iloc[j]); rval=side*(px-entry)/risk; result='WIN'; break
    if exit_px is None:
        j=end-1; exit_px=float(bid[j] if side==1 else ask[j]); exit_t=pd.Timestamp(t.time.iloc[j]); rval=side*(exit_px-entry)/risk
    tr=m.Trade(stage,TF,MODE,side,str(touch),str(exit_t),entry,stop,target,exit_px,float(rval),result,'',float(cand['score']),0.0,0.0)
    return tr,np.int64(exit_t.value),'ok'

def run_stage(stage,t,z,cands,all_events):
    tv=epoch_ns(t['time']); bid=t.bid.to_numpy(float,copy=False); ask=t.ask.to_numpy(float,copy=False); ztv=epoch_ns(z.time)
    trades=[]; last_exit=np.int64(-1); d={'candidates':len(cands),'displacement':0,'poi':0,'poi_touch':0,'mtf_pass':0,'context_pass':0,'accepted':0,'executed':0,'overlap':0}
    for c in cands:
        if c['disp_i'] is not None: d['displacement']+=1
        if c['zone'] is not None: d['poi']+=1
        # S1: enter after MSS close. S2: enter after displacement close. S3+: first executable raw tick in actual POI zone.
        if stage=='S1_SEQUENCE':
            k=int(np.searchsorted(tv,np.int64(pd.Timestamp(c['mss_time']).value),side='right'))
        elif stage=='S2_DISPLACEMENT':
            if c['disp_i'] is None: continue
            k=int(np.searchsorted(tv,np.int64(pd.Timestamp(c['disp_time']).value),side='right'))
        else:
            if c['disp_i'] is None or c['zone'] is None: continue
            lo,hi,_=c['zone']; st=pd.Timestamp(c['disp_time']); en=st+pd.Timedelta(minutes=6)
            k=first_touch(tv,bid,ask,st.value,en.value,c['side'],lo,hi)
            if k is None: continue
            d['poi_touch']+=1
            touch=pd.Timestamp(t.time.iloc[k])
            if stage in ('S4_TRUE_MTF','S5_CONTEXT'):
                net,sup,conf,hconf,leader,details=v6.mtf_fusion(all_events,touch,TF,c['side'])
                # Diagnostic uses the current strict V6 gate unchanged.
                if not (net>=1.10 and not hconf): continue
                d['mtf_pass']+=1
                if stage=='S5_CONTEXT':
                    bi=int(np.searchsorted(ztv,np.int64(touch.value),side='right')-1); bi=max(0,min(bi,len(z)-1))
                    ctx=v6.context_score(z,bi,c['side'])
                    if ctx<0.0: continue
                    d['context_pass']+=1
        if k is None or k>=len(t): continue
        d['accepted']+=1
        tr,last_exit,why=trade_from_tick(t,tv,bid,ask,z,c,k,stage,last_exit)
        if why=='overlap': d['overlap']+=1
        if tr is not None: trades.append(tr); d['executed']+=1
    return trades,d

def extended_metrics(trades):
    x=m.metrics(trades); rs=np.array([t.r for t in trades],float)
    pos=rs[rs>0]; neg=rs[rs<0]
    x['avg_win_R']=float(pos.mean()) if len(pos) else 0.0
    x['avg_loss_R']=float(neg.mean()) if len(neg) else 0.0
    x['median_R']=float(np.median(rs)) if len(rs) else 0.0
    x['wins_R2_count']=int((rs>=1.5).sum()) if len(rs) else 0
    return x

def main():
    src=m.resolve_source('XAUUSD','dukascopy_raw'); t=m.load_ticks(src)
    bars_by={tf:m.bars(t,tf) for tf in v6.TFS}; feats={tf:m.feat(bars_by[tf]) for tf in v6.TFS}
    z=feats[TF]; cands=detect_candidates(z); all_events={tf:v6.build_tf_events(feats[tf],tf) for tf in v6.TFS}
    out=Path(os.environ.get('AMOS_ABLATION_OUT','results/amos-v8-ablation')); out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for stage in STAGES:
        tr,diag=run_stage(stage,t,z,cands,all_events); met=extended_metrics(tr); met.update({'stage':stage,'diagnostics':diag})
        summary.append(met); pd.DataFrame([asdict(x) for x in tr]).to_csv(out/f'{stage}_trades.csv',index=False)
        (out/f'{stage}_metrics.json').write_text(json.dumps(met,indent=2),encoding='utf-8')
    flat=[]
    for s in summary:
        flat.append({k:v for k,v in s.items() if k!='diagnostics'} | {f'diag_{k}':v for k,v in s['diagnostics'].items()})
    pd.DataFrame(flat).to_csv(out/'summary.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={'symbol':'XAUUSD','timeframe':'M1','mode':'STANDARD','rr':RR,'raw_tick_source':str(src),'verification_level':'RAW_DUKASCOPY_BIDASK','bars_label':'right','bars_closed':'right','stages':STAGES,'sequence':'SWEEP->CISD->MSS','displacement_window_bars':4,'poi_types':['FVG','IFVG','BPR'],'poi_touch_window_bars':6,'mtf_gate':'V6 strict net>=1.10 and no higher-TF conflict','context_gate':'score>=0.0','horizon_bars':20,'rows':len(t)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
