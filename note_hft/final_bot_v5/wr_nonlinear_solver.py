#!/usr/bin/env python3
import argparse,json,lzma,struct,math,itertools
from pathlib import Path
import numpy as np
REC=struct.Struct('>3i2f'); SCALE=1000.0; TARGET=72.71

def load(root):
    out=[]; hour=0
    for f in sorted(Path(root).rglob('*h_ticks.bi5')):
        try: dec=lzma.decompress(f.read_bytes())
        except Exception: hour+=1; continue
        for i in range(0,len(dec)-REC.size+1,REC.size):
            ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i); ask=ask_i/SCALE; bid=bid_i/SCALE
            if ask>=bid>0: out.append((hour*3600000+ms,bid,ask))
        hour+=1
    return np.asarray(out,float)

def cooldown(times,idx,ms=3000):
    out=[]; nxt=-10**30
    for i in idx:
        if times[i]>=nxt: out.append(i); nxt=times[i]+ms
    return np.asarray(out,np.int64)

def wr_eval(times,bid,ask,score_a,score_b,h,split=0.6,cons=None):
    side=np.where((score_a>0)&(score_b>0),1,np.where((score_a<0)&(score_b<0),-1,0)).astype(np.int8)
    idx=np.flatnonzero(side!=0)
    if cons is not None: idx=idx[cons[idx]]
    idx=cooldown(times,idx)
    fut=np.searchsorted(times,times[idx]+h,side='left'); ok=fut<len(times); idx=idx[ok]; fut=fut[ok]
    mid=(bid+ask)/2; r=side[idx]*(mid[fut]-mid[idx]); cut=int(split*len(times))
    def one(mask):
        n=int(mask.sum()); rr=r[mask]; return {'N':n,'WR_pct':100*float(np.sum(rr>0))/n if n else 0.0}
    return one(idx<cut),one(idx>=cut)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('-o','--out',default='out.json'); a=ap.parse_args()
    x=load(a.root); times=x[:,0].astype(np.int64); bid=x[:,1]; ask=x[:,2]; n=len(x)
    A=np.full((5,n),np.nan); B=np.full((5,n),np.nan)
    for k in range(5): A[k,k:]=ask[:n-k]; B[k,k:]=bid[:n-k]
    valid=np.arange(n)>=4
    # 5-point OLS slope
    ols_a=(-2*A[4]-A[3]+A[1]+2*A[0])/10.0; ols_b=(-2*B[4]-B[3]+B[1]+2*B[0])/10.0
    # quadratic curvature on 5 points
    curv_a=(2*A[4]-A[3]-2*A[2]-A[1]+2*A[0])/14.0; curv_b=(2*B[4]-B[3]-2*B[2]-B[1]+2*B[0])/14.0
    # direction consistency 3/4 and 4/4
    da=np.vstack([ask-np.roll(ask,k) for k in range(1,5)]); db=np.vstack([bid-np.roll(bid,k) for k in range(1,5)])
    signa=np.sign(ols_a); signb=np.sign(ols_b)
    ca=np.sum(np.sign(da)==signa,axis=0); cb=np.sum(np.sign(db)==signb,axis=0)
    rows=[]
    horizons=(50,75,100,125,150,200,250,300)
    lambdas=(-4,-2,-1,-0.5,0,0.5,1,2,4)
    for h in horizons:
      # OLS only
      D,V=wr_eval(times,bid,ask,ols_a,ols_b,h); rows.append(('OLS',0,None,h,D,V))
      for q in (3,4):
        cons=valid&(ca>=q)&(cb>=q); D,V=wr_eval(times,bid,ask,ols_a,ols_b,h,cons=cons); rows.append((f'OLS_CONS_{q}of4',0,q,h,D,V))
      for lam in lambdas:
        sa=ols_a+lam*curv_a; sb=ols_b+lam*curv_b
        D,V=wr_eval(times,bid,ask,sa,sb,h); rows.append(('VEL_ACC',lam,None,h,D,V))
        cons=valid&(ca>=3)&(cb>=3); D,V=wr_eval(times,bid,ask,sa,sb,h,cons=cons); rows.append(('VEL_ACC_CONS',lam,3,h,D,V))
    out=[]
    for kind,lam,q,h,D,V in rows:
        if D['N']<500 or V['N']<500: continue
        score=max(abs(D['WR_pct']-TARGET),abs(V['WR_pct']-TARGET))+0.35*abs(D['WR_pct']-V['WR_pct'])
        out.append({'kind':kind,'lambda':lam,'consistency':q,'horizon_ms':h,'discovery':D,'validation':V,'score':score})
    out.sort(key=lambda z:z['score'])
    result={'target_WR_pct':TARGET,'ticks':n,'top30':out[:30]}
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
