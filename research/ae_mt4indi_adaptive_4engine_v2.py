"""AE MT4Indi Adaptive 4Engine v1 — clean-room Raw Bid/Ask research runner.
Modes: TREND, RANGE, REVERSAL, INDICATOR, AUTO.
No proprietary indicator source is used. Public concepts are reconstructed with standard math.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

# v2 replacement branch: negative v1 engines are intentionally replaced\nMODES=("TREND","RANGE","REVERSAL","INDICATOR","AUTO")  # rerun-v2-20260922

def ema(x,n):
    a=2/(n+1); out=np.empty(len(x)); out[0]=x[0]
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out
def rsi(c,n=14):
    d=np.diff(c,prepend=c[0]); up=np.maximum(d,0); dn=np.maximum(-d,0)
    au=ema(up,n); ad=ema(dn,n); return 100-100/(1+au/(ad+1e-12))
def atr(h,l,c,n=14):
    pc=np.roll(c,1); pc[0]=c[0]; tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc)))
    return ema(tr,n)
def bars_from_quotes(ts,bid,ask,minutes):
    bucket=(ts//(minutes*60_000_000_000)).astype(np.int64)
    u,idx=np.unique(bucket,return_index=True); end=np.r_[idx[1:]-1,len(ts)-1]
    mid=(bid+ask)/2
    return dict(t=ts[end],o=mid[idx],h=np.maximum.reduceat(mid,idx),l=np.minimum.reduceat(mid,idx),
                c=mid[end],bid=bid[end],ask=ask[end])

def signals_v1(b):
    c,h,l=b["c"],b["h"],b["l"]; n=len(c); A=atr(h,l,c); R=rsi(c); e21=ema(c,21); e55=ema(c,55)
    # compact ADX proxy: directional efficiency over ATR; router only, not claimed as canonical ADX.
    slope=np.abs(e21-np.roll(e21,2))/(2*A+1e-12)
    eff=np.abs(c-np.roll(c,14))/(np.array([np.sum(np.abs(np.diff(c[max(0,i-14):i+1]))) for i in range(n)])+1e-12)
    trend=np.clip(.55*eff+.45*np.clip(slope/.10,0,1),0,1)
    width=np.array([np.std(c[max(0,i-19):i+1])*4 for i in range(n)])/(A+1e-12)
    rang=np.clip(.55*(1-eff)+.45*np.clip((2.2-width)/2.2,0,1),0,1)
    ext=np.where(R>=72,(R-72)/28,np.where(R<=28,(28-R)/28,0))
    sweep=np.zeros(n); sweep[2:]=((h[2:]>h[1:-1])&(c[2:]<h[1:-1]) | (l[2:]<l[1:-1])&(c[2:]>l[1:-1]))
    rev=np.clip(.65*ext+.35*sweep,0,1)
    mom=np.clip(np.abs(R-50)/50,0,1); indi=np.clip(.55*mom+.45*np.clip(width/3,0,1),0,1)
    scores=np.vstack([trend,rang,rev,indi]).T
    # 3-component trading engines
    T=np.sign((c>e21).astype(int)+(e21>e55).astype(int)+(c>np.roll(c,20)).astype(int)-1.5)
    mid=ema(c,20); z=(c-mid)/(A+1e-12)
    G=np.where(((z<-1.35).astype(int)+(R<35).astype(int))>=2,1,np.where(((z>1.35).astype(int)+(R>65).astype(int))>=2,-1,0))
    candle=np.sign(c-np.roll(c,1)); sw=np.where(sweep>0,-np.sign(c-np.roll(c,1)),0)
    V=np.where(((R<=28).astype(int)+(sw>0).astype(int)+(candle>0).astype(int))>=2,1,
      np.where(((R>=72).astype(int)+(sw<0).astype(int)+(candle<0).astype(int))>=2,-1,0))
    I=np.where(R>52,1,np.where(R<48,-1,0))
    return scores,{"TREND":T,"RANGE":G,"REVERSAL":V,"INDICATOR":I}

def simulate(b,mode,capital=300.0):
    scores,eng=signals(b); eq=capital; peak=eq; dd=0; pnls=[]; pos=0; entry=0.; a=atr(b["h"],b["l"],b["c"])
    current=0
    for i in range(60,len(b["c"])-1):
        if mode=="AUTO": current=int(np.argmax(scores[i])); sig=eng[MODES[current]][i]
        else: sig=eng[mode][i]
        if pos==0 and sig:
            pos=int(sig); entry=b["ask"][i] if pos>0 else b["bid"][i]
        elif pos:
            px=b["bid"][i] if pos>0 else b["ask"][i]; move=(px-entry)*pos
            if move<=-1.6*a[i] or move>=2.4*a[i] or (sig and sig!=pos):
                pnl=move*0.01*100.0; eq+=pnl; pnls.append(pnl); pos=0
                peak=max(peak,eq); dd=max(dd,(peak-eq)/peak*100)
    p=np.array(pnls); wins=p[p>0].sum() if len(p) else 0.; losses=-p[p<0].sum() if len(p) else 0.
    return {"mode":mode,"N":len(p),"WR":float((p>0).mean()*100) if len(p) else 0,
      "PF":float(wins/losses) if losses else (999.0 if wins else 0),"EV":float(p.mean()) if len(p) else 0,
      "Net":float(p.sum()),"ReturnPct":float((eq/capital-1)*100),"MaxDDPct":float(dd)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--npz",required=True); ap.add_argument("--tf",type=int,default=1)
    ap.add_argument("--mode",choices=MODES,default="AUTO"); ap.add_argument("--out",required=True); a=ap.parse_args()
    q=np.load(a.npz); b=bars_from_quotes(q["ts_ns"],q["bid"],q["ask"],a.tf)
    r=simulate(b,a.mode); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2))
    print(json.dumps(r))
if __name__=="__main__": main()


# v2: replace RANGE/REVERSAL/INDICATOR signals; preserve TREND baseline.
def signals(b):
    scores,eng=signals_v1(b)
    c,h,l=b["c"],b["h"],b["l"]; A=atr(h,l,c); R=rsi(c); e9=ema(c,9); e21=ema(c,21); e55=ema(c,55)
    G=np.zeros(len(c)); V=np.zeros(len(c)); I=np.zeros(len(c))
    for i in range(60,len(c)):
        hh=np.max(h[i-20:i]); ll=np.min(l[i-20:i]); width=(hh-ll)/(A[i]+1e-12)
        compression=width<5.5
        if compression and l[i]<ll and c[i]>ll and R[i]<48: G[i]=1
        elif compression and h[i]>hh and c[i]<hh and R[i]>52: G[i]=-1
        hi10=np.max(h[i-10:i]); lo10=np.min(l[i-10:i])
        if l[i]<lo10 and c[i]>lo10 and c[i]>e9[i]: V[i]=1
        elif h[i]>hi10 and c[i]<hi10 and c[i]<e9[i]: V[i]=-1
        if e9[i]>e21[i]>e55[i] and R[i]>52: I[i]=1
        elif e9[i]<e21[i]<e55[i] and R[i]<48: I[i]=-1
        slope=abs(e21[i]-e21[i-3])/(A[i]+1e-12)
        scores[i,0]=min(1,.55*min(1,slope/.35)+.45*(1 if I[i] else 0))
        scores[i,1]=min(1,.6*(1 if compression else 0)+.4*(1 if G[i] else 0))
        scores[i,2]=1 if V[i] else 0
        scores[i,3]=min(1,.5*(1 if I[i] else 0)+.5*min(1,slope/.25))
    eng["RANGE"]=G; eng["REVERSAL"]=V; eng["INDICATOR"]=I
    return scores,eng
