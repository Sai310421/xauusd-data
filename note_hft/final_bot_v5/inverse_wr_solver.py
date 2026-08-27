#!/usr/bin/env python3
import argparse,json,lzma,struct,bisect,itertools,math
from pathlib import Path
import numpy as np

REC=struct.Struct('>3i2f'); SCALE=1000.0; TARGET_WR=72.71

# Unknown slots are treated as a constrained symbolic family:
# score_ask = sum_i c_i * (Ask_t-Ask_t-i)
# score_bid = sum_i c_i * (Bid_t-Bid_t-i)
# SLOT3 = -score_ask, SLOT4 = +score_bid
# Visible source predicates then become:
# BUY: score_ask>0 and score_bid>0; SELL: both <0.
# N is NEVER used in the objective.

def load_ticks(root):
    out=[]; files=sorted(Path(root).rglob('*h_ticks.bi5')); hour=0
    for f in files:
        try: dec=lzma.decompress(f.read_bytes())
        except Exception: hour+=1; continue
        for i in range(0,len(dec)-REC.size+1,REC.size):
            ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i)
            ask=ask_i/SCALE; bid=bid_i/SCALE
            if ask>=bid>0: out.append((hour*3600000+ms,bid,ask))
        hour+=1
    return out,files

def candidate_coeffs():
    # Parsimonious source-like expressions first: <=2 non-zero integer weights.
    vals=(-2,-1,1,2); seen=set(); out=[]
    for k in range(4):
        for v in vals:
            c=[0,0,0,0]; c[k]=v; seen.add(tuple(c))
    for i in range(4):
        for j in range(i+1,4):
            for a in vals:
                for b in vals:
                    c=[0,0,0,0]; c[i]=a; c[j]=b
                    # normalize sign/scale duplicates
                    g=math.gcd(abs(a),abs(b)); c=[x//g for x in c]
                    t=tuple(c)
                    if t not in seen: seen.add(t)
    # Explicit low-complexity shapes implied by 5-sample warmup.
    extras=[(1,1,1,1),(4,3,2,1),(1,2,3,4),(1,-1,0,0),(0,1,-1,0),(0,0,1,-1),(1,0,0,-1),(1,-2,1,0),(0,1,-2,1)]
    for x in extras: seen.add(tuple(x))
    return sorted(seen,key=lambda c:(sum(x!=0 for x in c),sum(abs(x) for x in c),c))

def cooldown_select(times, idx, cooldown=3000):
    if idx.size==0:return idx
    chosen=[]; next_t=-10**30
    for i in idx:
        t=times[i]
        if t>=next_t:
            chosen.append(i); next_t=t+cooldown
    return np.asarray(chosen,dtype=np.int64)

def eval_candidate(times,bid,ask,features_a,features_b,c,horizon,split):
    ca=np.asarray(c,dtype=float)
    sa=ca@features_a; sb=ca@features_b
    side=np.where((sa>0)&(sb>0),1,np.where((sa<0)&(sb<0),-1,0)).astype(np.int8)
    raw=np.flatnonzero(side!=0)
    sel=cooldown_select(times,raw,3000)
    if sel.size<2000:return None
    future=np.searchsorted(times,times[sel]+horizon,side='left')
    ok=future<len(times); sel=sel[ok]; future=future[ok]
    if sel.size<2000:return None
    mid=(bid+ask)/2.0
    r=side[sel]*(mid[future]-mid[sel])
    # zero moves are neither wins nor losses; source WR fingerprint is positive PnL trades / N,
    # so zero is counted as non-win.
    cut=int(split*len(times))
    dmask=sel<cut; vmask=sel>=cut
    def one(mask):
        n=int(mask.sum())
        if n<500:return None
        rr=r[mask]; wr=100.0*float(np.sum(rr>0))/n
        return {'N':n,'WR_pct':wr,'wins':int(np.sum(rr>0)),'zero':int(np.sum(rr==0))}
    D=one(dmask); V=one(vmask)
    if not D or not V:return None
    complexity=sum(x!=0 for x in c)+0.1*sum(abs(x) for x in c)
    # WR first. Generalization is second. N does not appear.
    score=max(abs(D['WR_pct']-TARGET_WR),abs(V['WR_pct']-TARGET_WR))+0.35*abs(D['WR_pct']-V['WR_pct'])+0.002*complexity
    return {'coeff':list(c),'horizon_ms':horizon,'slot3':'-sum(c_i*(ask_t-ask_t-i))','slot4':'+sum(c_i*(bid_t-bid_t-i))','discovery':D,'validation':V,'score':score,'complexity':complexity,'N_total_report_only':D['N']+V['N']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('-o','--out',default='wr_inverse.json'); ap.add_argument('--split',type=float,default=0.6); a=ap.parse_args()
    ticks,files=load_ticks(a.root)
    if len(ticks)<10000: raise SystemExit('insufficient ticks')
    arr=np.asarray(ticks,dtype=float); times=arr[:,0].astype(np.int64); bid=arr[:,1]; ask=arr[:,2]
    # d_k = current - k ticks back, k=1..4. First 4 samples are zeroed / unavailable.
    fa=np.zeros((4,len(ask)),dtype=float); fb=np.zeros((4,len(bid)),dtype=float)
    for k in range(1,5):
        fa[k-1,k:]=ask[k:]-ask[:-k]; fb[k-1,k:]=bid[k:]-bid[:-k]
    results=[]
    horizons=(50,75,100,125,150,200,250,300)
    coeffs=candidate_coeffs()
    for c in coeffs:
        for h in horizons:
            r=eval_candidate(times,bid,ask,fa,fb,c,h,a.split)
            if r: results.append(r)
    results.sort(key=lambda x:x['score'])
    out={'status':'WR_FIRST_INVERSE_SOLVER','target_WR_pct':TARGET_WR,'objective':'WR discovery+validation only; N excluded from optimization','ticks':len(ticks),'files':len(files),'candidate_count':len(coeffs)*len(horizons),'top20':results[:20]}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
