#!/usr/bin/env python3
import argparse,lzma,struct,json,bisect
from pathlib import Path
import numpy as np
REC=struct.Struct('>3i2f'); SCALE=1000.0
HORIZONS=(50,75,100,125,150,200,250,300)

def load(root):
    out=[]; hour=0
    for f in sorted(Path(root).rglob('*h_ticks.bi5')):
        try: dec=lzma.decompress(f.read_bytes())
        except Exception: hour+=1; continue
        for i in range(0,len(dec)-REC.size+1,REC.size):
            ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i)
            ask=ask_i/SCALE; bid=bid_i/SCALE
            if ask>=bid>0: out.append((hour*3600000+ms,bid,ask))
        hour+=1
    return np.asarray(out,dtype=float)

def cooldown(times,idx,ms=3000):
    out=[]; nxt=-10**30
    for i in idx:
        if times[i]>=nxt: out.append(i); nxt=times[i]+ms
    return np.asarray(out,dtype=np.int64)

def split_eval(times,bid,ask,side,h):
    idx=cooldown(times,np.flatnonzero(side!=0),3000)
    fut=np.searchsorted(times,times[idx]+h,side='left'); ok=fut<len(times); idx=idx[ok]; fut=fut[ok]
    mid=(bid+ask)/2; rr=side[idx]*(mid[fut]-mid[idx]); cut=int(len(times)*0.6)
    ans={}
    for name,mask in [('discovery',idx<cut),('validation',idx>=cut)]:
        n=int(mask.sum()); r=rr[mask]; ans[name]={'N':n,'WR_pct':100*float(np.sum(r>0))/n if n else 0.0,'wins':int(np.sum(r>0)),'zero':int(np.sum(r==0))}
    return ans

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('-o','--out',default='quick_wr_probe.json'); a=ap.parse_args()
    arr=load(a.root)
    if len(arr)<5000: raise SystemExit('insufficient ticks')
    t=arr[:,0].astype(np.int64); bid=arr[:,1]; ask=arr[:,2]
    # Exact Candidate A: 5-point OLS weights [-2,-1,0,1,2]/10
    ask_s=np.zeros(len(ask)); bid_s=np.zeros(len(bid))
    ask_s[4:]=(-2*ask[:-4]-ask[1:-3]+ask[3:-1]+2*ask[4:])/10.0
    bid_s[4:]=(-2*bid[:-4]-bid[1:-3]+bid[3:-1]+2*bid[4:])/10.0
    side_a=np.where((ask_s>0)&(bid_s>0),1,np.where((ask_s<0)&(bid_s<0),-1,0)).astype(np.int8)
    # 3/4 direction consistency gate
    da=np.zeros((4,len(ask))); db=np.zeros((4,len(bid)))
    for k in range(1,5): da[k-1,k:]=ask[k:]-ask[:-k]; db[k-1,k:]=bid[k:]-bid[:-k]
    sa=np.sign(ask_s); sb=np.sign(bid_s)
    ca=np.sum(np.sign(da)==sa,axis=0)/4.0; cb=np.sum(np.sign(db)==sb,axis=0)/4.0
    side_b=np.where((ask_s>0)&(bid_s>0)&(ca>=0.75)&(cb>=0.75),1,np.where((ask_s<0)&(bid_s<0)&(ca>=0.75)&(cb>=0.75),-1,0)).astype(np.int8)
    out={'ticks':int(len(arr)),'candidate_A_OLS':{},'candidate_B_OLS_consistency':{}}
    for h in HORIZONS:
        out['candidate_A_OLS'][str(h)]=split_eval(t,bid,ask,side_a,h)
        out['candidate_B_OLS_consistency'][str(h)]=split_eval(t,bid,ask,side_b,h)
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
