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

TF='M1'; MODE='STANDARD'; RR=4.0
OTE_MODES={
    'NO_OTE': None,
    'OTE_062_079': (0.62,0.79),
    'OTE_065_079': (0.65,0.79),
    'OTE_0675_0735': (0.675,0.735),
}
POI_GROUPS=['ALL','FVG','IFVG','BPR']

def ns(x):
    return np.fromiter((pd.Timestamp(v).value for v in x),dtype=np.int64,count=len(x))

def first_touch(tv,bid,ask,c):
    if c['disp_i'] is None or c['zone'] is None: return None
    lo,hi,_=c['zone']; st=pd.Timestamp(c['disp_time']); en=st+pd.Timedelta(minutes=6)
    return v8.first_touch(tv,bid,ask,st.value,en.value,int(c['side']),lo,hi)

def ote_retracement(z,c,entry):
    a=int(c['sweep_i']); b=int(c['disp_i'])
    if b<a: return np.nan
    side=int(c['side'])
    if side==1:
        leg_lo=float(c['extreme'])
        leg_hi=float(z.high.iloc[a:b+1].max())
        den=leg_hi-leg_lo
        return np.nan if den<=0 else (leg_hi-entry)/den
    leg_hi=float(c['extreme'])
    leg_lo=float(z.low.iloc[a:b+1].min())
    den=leg_hi-leg_lo
    return np.nan if den<=0 else (entry-leg_lo)/den

def pass_ote(r, band):
    if band is None: return True
    return np.isfinite(r) and band[0] <= r <= band[1]

def simulate(t,z,cands,mode,band,poi_group):
    tv=ns(t.time); bid=t.bid.to_numpy(float,copy=False); ask=t.ask.to_numpy(float,copy=False)
    out=[]; last=-1; diag={'candidates':0,'poi_type_pass':0,'touch':0,'ote_pass':0,'executed':0,'overlap':0}
    for c in cands:
        if c['disp_i'] is None or c['zone'] is None: continue
        poi=str(c['zone'][2])
        diag['candidates']+=1
        if poi_group!='ALL' and poi!=poi_group: continue
        diag['poi_type_pass']+=1
        k=first_touch(tv,bid,ask,c)
        if k is None or k>=len(t): continue
        diag['touch']+=1
        et=pd.Timestamp(t.time.iloc[k]); side=int(c['side']); entry=float(ask[k] if side==1 else bid[k])
        r=ote_retracement(z,c,entry)
        if not pass_ote(r,band): continue
        diag['ote_pass']+=1
        if et.value<=last:
            diag['overlap']+=1; continue
        av=float(z.atr.iloc[c['disp_i']]); sweep=float(c['extreme'])
        risk=max(abs(entry-sweep),0.50*av); stop=entry-side*risk; target=entry+side*RR*risk
        end=min(int(np.searchsorted(tv,(et+pd.Timedelta(minutes=120)).value,side='right')),len(t))
        if end<=k+1: continue
        result='TIME'; xp=xt=rval=None
        for j in range(k+1,end):
            px=float(bid[j] if side==1 else ask[j])
            if (side==1 and px<=stop) or (side==-1 and px>=stop):
                result='LOSS'; xp=px; xt=pd.Timestamp(t.time.iloc[j]); rval=side*(px-entry)/risk; break
            if (side==1 and px>=target) or (side==-1 and px<=target):
                result='WIN'; xp=px; xt=pd.Timestamp(t.time.iloc[j]); rval=side*(px-entry)/risk; break
        if xp is None:
            j=end-1; xp=float(bid[j] if side==1 else ask[j]); xt=pd.Timestamp(t.time.iloc[j]); rval=side*(xp-entry)/risk
        label=f'V11_{mode}_{poi_group}'
        note=f'POI_{poi};OTE={r:.6f}' if np.isfinite(r) else f'POI_{poi};OTE=nan'
        out.append(m.Trade(label,TF,MODE,side,str(et),str(xt),entry,stop,target,xp,float(rval),result,note,float(c['score']),float(r) if np.isfinite(r) else 0.,0.))
        last=xt.value; diag['executed']+=1
    return out,diag

def metrics(tr):
    x=m.metrics(tr); rs=np.array([q.r for q in tr],float); pos=rs[rs>0]; neg=rs[rs<0]
    x.update(avg_win_R=float(pos.mean()) if len(pos) else 0., avg_loss_R=float(neg.mean()) if len(neg) else 0., median_R=float(np.median(rs)) if len(rs) else 0.)
    return x

def main():
    src=m.resolve_source('XAUUSD','dukascopy_raw'); t=m.load_ticks(src); z=m.feat(m.bars(t,TF)); cands=v8.detect_candidates(z)
    out=Path(os.environ.get('AMOS_V11_OUT','results/amos-v11-ote-ablation')); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for mode,band in OTE_MODES.items():
        for poi_group in POI_GROUPS:
            tr,diag=simulate(t,z,cands,mode,band,poi_group); met=metrics(tr)
            met.update({'OTE_mode':mode,'OTE_lo':None if band is None else band[0],'OTE_hi':None if band is None else band[1],'POI_group':poi_group,'RR_target':RR})
            met.update({f'diag_{k}':v for k,v in diag.items()}); rows.append(met)
            pd.DataFrame([asdict(q) for q in tr]).to_csv(out/f'{mode}_{poi_group}_trades.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(out/'summary.csv',index=False); (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    manifest={
        'purpose':'OTE ablation on V10 S3 CURRENT raw bid/ask setup',
        'symbol':'XAUUSD','tf':TF,'mode':MODE,'rr':RR,
        'entry':'actual FVG/IFVG/BPR first raw-tick touch within 6 minutes of displacement',
        'execution':'raw Dukascopy bid/ask','stop':'sweep extreme with minimum 0.50 ATR','horizon_minutes':120,
        'ote_definition':'Sweep-to-displacement impulse retracement measured at executable POI-touch entry. Bull=(impulse_high-entry)/(impulse_high-sweep_low); Bear=(entry-impulse_low)/(sweep_high-impulse_low).',
        'ote_modes':{k:v for k,v in OTE_MODES.items()},'poi_groups':POI_GROUPS,
        'important':'Each cell applies its own no-overlap clock, so N is not a perfectly paired comparison. Use as ablation screening; confirm winners on common signal IDs.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(df.to_json(orient='records',indent=2))
if __name__=='__main__': main()
