#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import amos_reversal_v1_v4_rawtick_bt_v2 as m
import amos_reversal_v8_ablation_m1_standard as v8
TF='M1'; MODE='STANDARD'; RRS=[2.0,3.0,4.0,5.0,6.0,8.0]
def ns(x): return np.fromiter((pd.Timestamp(v).value for v in x),dtype=np.int64,count=len(x))
def touch(tv,bid,ask,c):
    if c['disp_i'] is None or c['zone'] is None:return None
    lo,hi,_=c['zone'];st=pd.Timestamp(c['disp_time']);en=st+pd.Timedelta(minutes=6);return v8.first_touch(tv,bid,ask,st.value,en.value,int(c['side']),lo,hi)
def simulate(t,z,cands,rr):
    tv=ns(t.time);bid=t.bid.to_numpy(float,copy=False);ask=t.ask.to_numpy(float,copy=False);out=[];last=-1
    for c in cands:
        k=touch(tv,bid,ask,c)
        if k is None or k>=len(t):continue
        et=pd.Timestamp(t.time.iloc[k])
        if et.value<=last:continue
        side=int(c['side']);entry=float(ask[k] if side==1 else bid[k]);av=float(z.atr.iloc[c['disp_i']]);sweep=float(c['extreme']);risk=max(abs(entry-sweep),.50*av);stop=entry-side*risk;target=entry+side*rr*risk
        end=min(int(np.searchsorted(tv,(et+pd.Timedelta(minutes=120)).value,side='right')),len(t))
        if end<=k+1:continue
        result='TIME';xp=xt=rval=None
        for j in range(k+1,end):
            px=float(bid[j] if side==1 else ask[j])
            if (side==1 and px<=stop) or (side==-1 and px>=stop):result='LOSS';xp=px;xt=pd.Timestamp(t.time.iloc[j]);rval=side*(px-entry)/risk;break
            if (side==1 and px>=target) or (side==-1 and px<=target):result='WIN';xp=px;xt=pd.Timestamp(t.time.iloc[j]);rval=side*(px-entry)/risk;break
        if xp is None:j=end-1;xp=float(bid[j] if side==1 else ask[j]);xt=pd.Timestamp(t.time.iloc[j]);rval=side*(xp-entry)/risk
        poi=c['zone'][2] if c.get('zone') is not None and len(c['zone'])>2 else 'POI';out.append(m.Trade(f'V10_RR{rr:g}',TF,MODE,side,str(et),str(xt),entry,stop,target,xp,float(rval),result,f'POI_{poi}',float(c['score']),0.,0.));last=xt.value
    return out
def metrics(tr):
    x=m.metrics(tr);rs=np.array([q.r for q in tr],float);pos=rs[rs>0];neg=rs[rs<0];x.update(avg_win_R=float(pos.mean()) if len(pos) else 0.,avg_loss_R=float(neg.mean()) if len(neg) else 0.,median_R=float(np.median(rs)) if len(rs) else 0.);return x
def main():
    src=m.resolve_source('XAUUSD','dukascopy_raw');t=m.load_ticks(src);z=m.feat(m.bars(t,TF));c=v8.detect_candidates(z);out=Path(os.environ.get('AMOS_V10_OUT','results/amos-v10-ict-rr-sweep'));out.mkdir(parents=True,exist_ok=True);rows=[]
    for rr in RRS:
        tr=simulate(t,z,c,rr);met=metrics(tr);met['RR_target']=rr;rows.append(met);pd.DataFrame([asdict(q) for q in tr]).to_csv(out/f'RR_{rr:g}_trades.csv',index=False);(out/f'RR_{rr:g}_metrics.json').write_text(json.dumps(met,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);(out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'manifest.json').write_text(json.dumps({'purpose':'ICT high-RR sweep on verified S3 CURRENT direction','symbol':'XAUUSD','tf':TF,'mode':MODE,'rr_targets':RRS,'entry':'actual FVG/IFVG/BPR first raw-tick touch','execution':'raw bid/ask','stop':'sweep extreme with minimum 0.50 ATR','horizon_minutes':120,'verification':'RAW_DUKASCOPY_BIDASK'},indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
