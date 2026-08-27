#!/usr/bin/env python3
import argparse, json, lzma, struct, math
from pathlib import Path
import numpy as np

REC=struct.Struct('>3i2f'); SCALE=1000.0; TARGET=72.71
H=(50,75,100,125,150,200,250,300)

def load(root):
    rows=[]; files=sorted(Path(root).rglob('*h_ticks.bi5')); hour=0
    for f in files:
        try: dec=lzma.decompress(f.read_bytes())
        except Exception: hour+=1; continue
        for i in range(0,len(dec)-REC.size+1,REC.size):
            ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i)
            ask=ask_i/SCALE; bid=bid_i/SCALE
            if ask>=bid>0: rows.append((hour*3600000+ms,bid,ask))
        hour+=1
    a=np.asarray(rows,float)
    return a[:,0].astype(np.int64),a[:,1],a[:,2],len(files)

def cooldown(t,idx,ms=3000):
    out=[]; nxt=-10**30
    for i in idx:
        if t[i]>=nxt: out.append(i); nxt=t[i]+ms
    return np.asarray(out,np.int64)

def stats(t,mid,side,h,split=.5):
    idx=np.flatnonzero(side!=0); idx=cooldown(t,idx,3000)
    fut=np.searchsorted(t,t[idx]+h,'left'); ok=fut<len(t); idx=idx[ok]; fut=fut[ok]
    if len(idx)<1000:return None
    rr=side[idx]*(mid[fut]-mid[idx]); cut=int(split*len(t)); dm=idx<cut; vm=~dm
    def one(m):
        n=int(m.sum())
        if n<300:return None
        x=rr[m]; return {'N':n,'WR_pct':100*float(np.sum(x>0))/n,'wins':int(np.sum(x>0)),'zero':int(np.sum(x==0))}
    d=one(dm); v=one(vm)
    if not d or not v:return None
    score=max(abs(d['WR_pct']-TARGET),abs(v['WR_pct']-TARGET))+0.35*abs(d['WR_pct']-v['WR_pct'])
    return d,v,score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('-o','--out',default='micro_wr.json'); a=ap.parse_args()
    t,bid,ask,nfiles=load(a.root); mid=(ask+bid)/2; spread=ask-bid; n=len(t)
    if n<5000: raise SystemExit('insufficient ticks')
    # 5-point source-like features
    def lag(x,k):
        y=np.zeros(n); y[k:]=x[k:]-x[:-k]; return y
    dm1=lag(mid,1); dm2=lag(mid,2); dm4=lag(mid,4)
    ds1=lag(spread,1); ds2=lag(spread,2); ds4=lag(spread,4)
    da1=lag(ask,1); db1=lag(bid,1)
    # exact OLS on last 5 ticks
    ols=np.zeros(n); ols[4:]=(-2*mid[:-4]-mid[1:-3]+mid[3:-1]+2*mid[4:])/10.0
    ols_a=np.zeros(n); ols_a[4:]=(-2*ask[:-4]-ask[1:-3]+ask[3:-1]+2*ask[4:])/10.0
    ols_b=np.zeros(n); ols_b[4:]=(-2*bid[:-4]-bid[1:-3]+bid[3:-1]+2*bid[4:])/10.0
    acc=dm1-(np.r_[0,dm1[:-1]])
    curvature=np.zeros(n); curvature[4:]=(2*mid[:-4]-mid[1:-3]-2*mid[2:-2]-mid[3:-1]+2*mid[4:])/14.0
    eps=1e-12
    feats={
      'mid_v1':dm1,'mid_v2':dm2,'mid_v4':dm4,'ols_mid':ols,
      'acc':acc,'curv':curvature,'spread_v1':ds1,'spread_v2':ds2,'spread_v4':ds4,
      'ask_bid_delta_diff':da1-db1,'ask_bid_delta_sum':da1+db1,
      'ols_coherence':np.minimum(np.abs(ols_a),np.abs(ols_b))*np.sign(ols_a+ols_b),
    }
    results=[]
    lambdas=(-4,-2,-1,-.5,0,.5,1,2,4)
    gate_q=(0.0,0.25,0.5,0.7,0.8,0.9,0.95)
    # Family 1: OLS improved or reversed with spread/acceleration penalties.
    combos=[]
    for base_name in ('ols_mid','mid_v1','mid_v2','mid_v4'):
      base=feats[base_name]
      for aux_name in ('spread_v1','spread_v2','acc','curv','ask_bid_delta_diff'):
        aux=feats[aux_name]
        scale=np.nanmedian(np.abs(base[4:]))/(np.nanmedian(np.abs(aux[4:]))+eps)
        for lam in lambdas:
          sc=base-lam*scale*aux
          combos.append((f'{base_name}-({lam})*{aux_name}',sc))
          combos.append((f'REVERSE[{base_name}-({lam})*{aux_name}]',-sc))
    # pure source-like features also
    for name,x in feats.items(): combos.append((name,x)); combos.append((f'REVERSE[{name}]',-x))
    seen=set()
    for name,sc in combos:
      key=name
      if key in seen: continue
      seen.add(key)
      mag=np.abs(sc)
      valid=mag[4:]
      for q in gate_q:
        thr=float(np.quantile(valid,q)) if q>0 else 0.0
        # coherence gate: Ask/Bid must agree in direction; source downstream already implies this idea.
        coh=((ols_a>0)&(ols_b>0))|((ols_a<0)&(ols_b<0))
        for use_coh in (False,True):
          side=np.where(sc>thr,1,np.where(sc<-thr,-1,0)).astype(np.int8)
          if use_coh: side=np.where(coh,side,0).astype(np.int8)
          for h in H:
            z=stats(t,mid,side,h)
            if not z: continue
            d,v,s=z
            results.append({'formula':name,'gate_quantile':q,'threshold':thr,'coherence_gate':use_coh,'horizon_ms':h,'discovery':d,'validation':v,'score':s,'N_report_only':d['N']+v['N']})
    results.sort(key=lambda x:x['score'])
    out={'status':'MICROSTRUCTURE_WR_FIRST','target_WR_pct':TARGET,'ticks':n,'files':nfiles,'objective':'WR discovery+validation only; N excluded','tested':len(results),'top50':results[:50]}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
