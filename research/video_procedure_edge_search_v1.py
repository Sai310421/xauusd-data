from __future__ import annotations
"""Video-procedure statistical EDGE search v1.0.

Implements only the procedure supported by the uploaded video:
candidate lanes -> distribution filter -> Monte Carlo risk comparison -> shortlist.

The video's exact P0/P1/P2/P3 parameter semantics are not visible, so this
module does NOT invent them. It sweeps explicit, already-defined AMOS EDGE
signals and execution parameters instead.
"""
import argparse, json, math, random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
import numpy as np
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.nautilus_catalog_compat import query_quote_ticks_compat

EDGE_IDS=['T1_EMA21','T2_MTF_ALIGN','T3_BREAKOUT_CONT','T4_ACCEL','R1_RANGE_BIAS','R2_COMP_EXP','R3_MICRO_BREAK','V1_SWEEP','V2_MEAN_EXTREME','V3_REJECTION']
TF_MIN={'M1':1,'M5':5,'M15':15}
SCHEMA='AMOS.VideoProcedureEdgeSearch.v1.0'

def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)

def bars(ticks,minutes):
    ns=minutes*60*1_000_000_000; d={}
    for t in ticks:
        k=(int(t.ts_event)//ns)*ns; bid=f(t.bid_price); ask=f(t.ask_price); mid=(bid+ask)/2
        if k not in d:d[k]=[mid,mid,mid,mid,ask-bid]
        else:
            d[k][1]=max(d[k][1],mid);d[k][2]=min(d[k][2],mid);d[k][3]=mid;d[k][4]=ask-bid
    return [(k,*d[k]) for k in sorted(d)]

def ema(a,n):
    if len(a)<n:return None
    z=list(a)[-n:]; al=2/(n+1); v=z[0]
    for x in z[1:]:v=al*x+(1-al)*v
    return v

def rsi(a,n=14):
    if len(a)<n+1:return 50.
    d=np.diff(np.asarray(list(a)[-(n+1):],float));up=np.maximum(d,0).mean();dn=np.maximum(-d,0).mean()
    return 100. if dn<=1e-12 else 100-100/(1+up/dn)

def scores(c,h,l,tr):
    if len(c)<70:return {}
    C=np.asarray(c,float);H=np.asarray(h,float);L=np.asarray(l,float);atr=max(float(np.mean(list(tr)[-20:])),1e-12)
    e21=ema(c,21);e50=ema(c,50);e21p=float(np.mean(C[-22:-1]));loc=np.clip((C[-1]-e21)/atr,-1,1);slope=np.clip((e21-e21p)/atr*8,-1,1)
    hh=float(H[-21:-1].max());ll=float(L[-21:-1].min());mom=float(C[-1]-C[-4]);mom0=float(C[-4]-C[-7]);den=max(abs(mom0),atr*.05)
    vals=np.asarray(list(tr),float);cur=float(np.mean(vals[-20:]));hist=np.asarray([np.mean(vals[max(0,i-19):i+1]) for i in range(19,len(vals))]);atrpct=float(np.mean(hist<=cur)) if len(hist) else .5
    rh=float(H[-8:-1].max());rl=float(L[-8:-1].min());rr=rsi(c);rng=max(H[-1]-L[-1],1e-12);body=abs(C[-1]-C[-2]);upper=H[-1]-max(C[-1],C[-2]);lower=min(C[-1],C[-2])-L[-1]
    return {
      'T1_EMA21':float(np.clip(.65*loc+.35*slope,-1,1)),
      'T2_MTF_ALIGN':float(np.sign(e21-e50)),
      'T3_BREAKOUT_CONT':float(np.clip((C[-1]-hh)/atr*2,-1,1)) if C[-1]>hh else float(-np.clip((ll-C[-1])/atr*2,-1,1)) if C[-1]<ll else 0.,
      'T4_ACCEL':float(np.clip((mom-mom0)/den,-1,1)),
      'R1_RANGE_BIAS':float(np.clip(.75*np.clip((C[-1]-e21)/atr*.8,-1,1)+.25*np.clip((e21-e50)/atr,-1,1),-1,1)),
      'R2_COMP_EXP':float((1 if mom>0 else -1 if mom<0 else 0)*np.clip(1-atrpct,0,1)),
      'R3_MICRO_BREAK':float(np.clip((C[-1]-rh)/atr*3,-1,1)) if C[-1]>rh else float(-np.clip((rl-C[-1])/atr*3,-1,1)) if C[-1]<rl else 0.,
      'V1_SWEEP':float((1 if L[-1]<ll and C[-1]>ll else 0)-(1 if H[-1]>hh and C[-1]<hh else 0)),
      'V2_MEAN_EXTREME':float(-np.clip((rr-50)/35,0,1)) if rr>=50 else float(np.clip((50-rr)/35,0,1)),
      'V3_REJECTION':float(np.clip((lower-upper)/rng,-1,1)) if body/rng<.8 else 0.,
    }

def trades_for_lane(B,edge,direction,threshold,tp_atr,sl_atr,horizon):
    c=deque(maxlen=240);h=deque(maxlen=240);l=deque(maxlen=240);tr=deque(maxlen=240);prev=None;out=[];i=0
    while i<len(B):
        _,o,hi,lo,cl,spread=B[i];tv=max(hi-lo,abs(hi-prev) if prev is not None else 0,abs(lo-prev) if prev is not None else 0);tr.append(tv);prev=cl;c.append(cl);h.append(hi);l.append(lo)
        s=scores(c,h,l,tr)
        if s and direction*s[edge]>=threshold:
            atr=max(float(np.mean(list(tr)[-20:])),1e-12);entry=cl+direction*spread/2;end=min(len(B)-1,i+horizon);exitpx=B[end][4]-direction*B[end][5]/2;reason='HORIZON'
            for j in range(i+1,end+1):
                _,_,H,L,C,S=B[j];fav=(H-entry) if direction>0 else (entry-L);adv=(entry-L) if direction>0 else (H-entry)
                if adv>=sl_atr*atr:exitpx=entry-direction*sl_atr*atr;reason='SL';end=j;break
                if fav>=tp_atr*atr:exitpx=entry+direction*tp_atr*atr;reason='TP';end=j;break
            pnl=(exitpx-entry)*direction; out.append(pnl/atr);i=end
        i+=1
    return out

def metrics(rs):
    n=len(rs);gw=sum(x for x in rs if x>0);gl=-sum(x for x in rs if x<0);pf=gw/gl if gl>0 else (999. if gw>0 else 0.);wr=sum(x>0 for x in rs)/n if n else 0.;ev=sum(rs)/n if n else 0.
    return {'N':n,'WR':wr,'PF':pf,'EV_R':ev,'median_R':median(rs) if rs else 0.,'no_loss_share':sum(x>=0 for x in rs)/n if n else 0.}

def monte_carlo(rs,risk_pct,paths=2000,trades=400,seed=1):
    if not rs:return {'risk_pct':risk_pct,'paths':paths,'trades':trades,'median_terminal_return_pct':0.,'median_maxdd_pct':0.,'p90_maxdd_pct':0.}
    rng=random.Random(seed);terms=[];dds=[]
    for _ in range(paths):
        eq=1.;peak=1.;mdd=0.
        for _t in range(trades):
            r=rng.choice(rs);eq*=max(1e-9,1+r*risk_pct/100);peak=max(peak,eq);mdd=max(mdd,(peak-eq)/peak)
        terms.append((eq-1)*100);dds.append(mdd*100)
    return {'risk_pct':risk_pct,'paths':paths,'trades':trades,'median_terminal_return_pct':median(terms),'median_maxdd_pct':median(dds),'p90_maxdd_pct':float(np.quantile(dds,.9))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--symbol',default='XAUUSD');ap.add_argument('--experiment-id',required=True);a=ap.parse_args()
    cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')==a.symbol);raw=query_quote_ticks_compat(cat,identifiers=[inst.id.value])
    rows=[]
    for tf in ('M1','M5','M15'):
      B=bars(raw,TF_MIN[tf])
      for edge in EDGE_IDS:
       for direction in (-1,1):
        for threshold in (.25,.5,.75):
         for rr in ((.75,.75),(1.,.75),(1.5,1.)):
          rs=trades_for_lane(B,edge,direction,threshold,rr[0],rr[1],60//TF_MIN[tf])
          m=metrics(rs);rows.append({'tf':tf,'edge':edge,'direction':'LONG' if direction>0 else 'SHORT','threshold':threshold,'tp_atr':rr[0],'sl_atr':rr[1],**m,'returns_R':rs})
    # Video procedure: distribution-first shortlist. Exact video dollar thresholds are not portable to R-normalized AMOS lanes.
    eligible=[x for x in rows if x['N']>=100 and x['PF']>=1.20 and x['EV_R']>0]
    eligible.sort(key=lambda x:(x['PF'],x['EV_R'],x['N']),reverse=True)
    top=eligible[:20]
    for k,x in enumerate(top):
        x['monte_carlo']=[monte_carlo(x['returns_R'],r,seed=20260921+k) for r in (.5,1.,2.)]
        x.pop('returns_R',None)
    compact=[]
    for x in rows:
        y={k:v for k,v in x.items() if k!='returns_R'};compact.append(y)
    out=Path('results/video-edge-search')/a.experiment_id;out.mkdir(parents=True,exist_ok=True)
    summary={'schema':SCHEMA,'source_procedure':'UPLOADED_VIDEO_DISTRIBUTION_FIRST_EDGE_SEARCH','raw_ticks':len(raw),'symbol':a.symbol,'lanes_tested':len(rows),'eligible_count':len(eligible),'shortlist':top,
      'video_supported_steps':['candidate lane sweep','distribution metrics','filtering','Monte Carlo risk comparison','chart interpretation after statistical selection'],
      'unknown_from_video':['exact P0/P1/P2/P3 semantics','exact source entry/exit rules behind the displayed 96 lanes'],
      'amos_mapping':'Explicit existing AMOS EDGE definitions are swept instead of inventing unknown video parameters.',
      'gate':{'N_min':100,'PF_min':1.20,'EV_R_min_exclusive':0},'production_weighting_allowed':False,'execution_allowed':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8');(out/'all_lanes.json').write_text(json.dumps(compact,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
