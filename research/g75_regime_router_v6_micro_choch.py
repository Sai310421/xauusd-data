from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from g75_regime_router_v5_sweep_reclaim import load,bias30,prep,state,met,TR,ADD,REV,MAXL,NS,MIN,M5

def run(t,bid,ask,bias,m5,m15,variant):
    bucket=-1;anchor=None;cm=None;started=False;eng=None;side=0;entries=[];last=None;ext=None;startreg=None;target=stop=None
    phase=0;rs=0;rh0=rl0=atr0=None;sweep_ext=None;reclaim_ext=None;pull_ext=None;phase_t=0
    pn=[];by={'UP':[],'DOWN':[],'RANGE':[]};real=0.;peak=1000.;mdd=0.;adds=0;blocked=0;starts={k:0 for k in ['UP','DOWN','RANGE','TRANSITION']}
    def reset_range():
        nonlocal phase,rs,rh0,rl0,atr0,sweep_ext,reclaim_ext,pull_ext,phase_t
        phase=0;rs=0;rh0=rl0=atr0=sweep_ext=reclaim_ext=pull_ext=None;phase_t=0
    def close(b,a):
        nonlocal eng,side,entries,last,ext,startreg,target,stop,real
        if eng is None:return
        z=b if side>0 else a;p=sum((z-e)*side for e in entries);pn.append(p);by[startreg].append(p);real+=p
        eng=None;side=0;entries=[];last=None;ext=None;startreg=None;target=stop=None
    for i,t0 in enumerate(t):
        tt=int(t0);b=float(bid[i]);a=float(ask[i]);mid=(b+a)/2;reg,rr=state(tt,m5,m15);bk=(tt//NS)//300
        if bucket<0:bucket=bk;anchor=mid
        elif bk!=bucket:
            if eng=='TREND':
                z=b if side>0 else a
                if (z<=ext-REV if side>0 else z>=ext+REV):close(b,a)
            anchor=cm;bucket=bk;started=False
        cm=mid
        if eng=='RANGE':
            z=b if side>0 else a
            if reg!='RANGE' or (side>0 and (z>=target or z<=stop)) or (side<0 and (z<=target or z>=stop)):close(b,a)
        elif eng=='TREND':
            z=b if side>0 else a;ext=max(ext,z) if side>0 else min(ext,z)
            while len(entries)<MAXL:
                tar=last+side*ADD;cross=z>=tar if side>0 else z<=tar
                if not cross:break
                entries.append(a if side>0 else b);last=tar;adds+=1
        if reg!='RANGE' or rr is None:
            reset_range()
        elif eng is None and variant!='TREND_ONLY':
            rh,rl,atr=rr;w=rh-rl
            if np.isfinite(w) and np.isfinite(atr) and w>max(.8,2*atr):
                sweep=max(.03,.03*atr);reclaim=max(.05,.07*atr);pull=max(.04,.06*atr);brk=max(.03,.04*atr);expiry=6*M5
                if phase and tt-phase_t>expiry:reset_range()
                if phase==0:
                    if mid<=rl-sweep: phase=1;rs=1;rh0=rh;rl0=rl;atr0=atr;sweep_ext=mid;phase_t=tt
                    elif mid>=rh+sweep: phase=1;rs=-1;rh0=rh;rl0=rl;atr0=atr;sweep_ext=mid;phase_t=tt
                elif phase==1 and rs==1:
                    sweep_ext=min(sweep_ext,mid)
                    if mid>=rl0+reclaim: phase=2;reclaim_ext=mid;pull_ext=mid
                elif phase==1 and rs==-1:
                    sweep_ext=max(sweep_ext,mid)
                    if mid<=rh0-reclaim: phase=2;reclaim_ext=mid;pull_ext=mid
                elif phase==2 and rs==1:
                    reclaim_ext=max(reclaim_ext,mid);pull_ext=min(pull_ext,mid)
                    if reclaim_ext-mid>=pull: phase=3;pull_ext=mid
                elif phase==2 and rs==-1:
                    reclaim_ext=min(reclaim_ext,mid);pull_ext=max(pull_ext,mid)
                    if mid-reclaim_ext>=pull: phase=3;pull_ext=mid
                elif phase==3 and rs==1:
                    pull_ext=min(pull_ext,mid)
                    if mid>=reclaim_ext+brk:
                        side=1;eng='RANGE';entries=[a];startreg='RANGE';starts['RANGE']+=1
                        target=rl0+(0.50 if variant=='CHOCH_STRICT' else 0.42)*(rh0-rl0);stop=sweep_ext-max(.08,.06*atr0);reset_range()
                elif phase==3 and rs==-1:
                    pull_ext=max(pull_ext,mid)
                    if mid<=reclaim_ext-brk:
                        side=-1;eng='RANGE';entries=[b];startreg='RANGE';starts['RANGE']+=1
                        target=rh0-(0.50 if variant=='CHOCH_STRICT' else 0.42)*(rh0-rl0);stop=sweep_ext+max(.08,.06*atr0);reset_range()
        if eng is None and rr is not None and reg in ('UP','DOWN') and not started and anchor is not None:
            cand=1 if mid>=anchor+TR else(-1 if mid<=anchor-TR else 0)
            allow=cand!=0 and cand==int(bias[i]) and cand==(1 if reg=='UP' else -1)
            if allow:
                side=cand;eng='TREND';entry=a if side>0 else b;entries=[entry];last=entry;ext=b if side>0 else a;startreg=reg;starts[reg]+=1;started=True
            elif cand:blocked+=1
        if eng is not None:
            z=b if side>0 else a;mark=sum((z-e)*side for e in entries);eq=1000+real+mark;peak=max(peak,eq);mdd=max(mdd,max(0.,(peak-eq)/max(peak,1e-9)*100))
    if eng is not None:close(float(bid[-1]),float(ask[-1]))
    r=met(pn,mdd);r.update({'variant':variant,'adds':adds,'blocked':blocked,'starts':starts,'by_engine':{k:met(v,0.) for k,v in by.items()}});return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--events',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    o=Path(a.out);o.mkdir(parents=True,exist_ok=True);t,bid,ask=load(a.catalog);bi=bias30(t,a.events);m5,m15=prep(t,bid,ask);rows=[]
    for v in ['TREND_ONLY','CHOCH_RECLAIM','CHOCH_STRICT']:
        r=run(t,bid,ask,bi,m5,m15,v);r['raw_ticks']=len(t);rows.append(r);print(json.dumps(r))
    (o/'summary.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(o/'summary.csv',index=False)
if __name__=='__main__':main()
