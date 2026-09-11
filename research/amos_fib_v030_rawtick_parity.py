from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from g75_expected_action_rawtick_v5 import f, load_all

NS=1_000_000_000
M5=300

@dataclass
class Bar:
    ts:int; o:float; h:float; l:float; c:float


def mid(t): return (f(t.bid_price)+f(t.ask_price))*0.5

def spread(t): return max(0.0,f(t.ask_price)-f(t.bid_price))

def bucket(ts,sec=M5): return (ts//(sec*NS))*(sec*NS)

def ema(vals,p):
    if len(vals)<p: return None
    a=2.0/(p+1.0); e=float(np.mean(vals[:p]))
    for x in vals[p:]: e=a*x+(1-a)*e
    return e

def atr(bars,p):
    if len(bars)<p+1: return None
    tr=[]
    for i in range(1,len(bars)):
        b,pr=bars[i],bars[i-1]
        tr.append(max(b.h-b.l,abs(b.h-pr.c),abs(b.l-pr.c)))
    return float(np.mean(tr[-p:])) if len(tr)>=p else None

def build_geom(bars,lookback=48):
    if len(bars)<lookback: return None
    w=bars[-lookback:]
    hi=max(range(len(w)),key=lambda i:w[i].h); lo=min(range(len(w)),key=lambda i:w[i].l)
    H=w[hi].h; L=w[lo].l
    if hi>lo: # high is newer in chronological array => bearish
        side=-1; A=L; B=H
    else:
        side=1; A=H; B=L
    wave=abs(A-B)
    if wave<=0: return None
    entry=B+side*0.5*wave
    sl=B-side*0.20*wave
    tp1=B+side*1.0*wave; tp2=B+side*1.272*wave; tp3=B+side*1.618*wave
    return side,A,B,wave,entry,sl,tp1,tp2,tp3

def regime(bars):
    a14=atr(bars,14); a50=atr(bars,50)
    if not a14 or not a50: return 'NORMAL',a14
    closes=[b.c for b in bars]
    e1=ema(closes[-20:],9); e6=ema(closes[:-5][-20:],9) if len(closes)>=14 else e1
    drive=abs(e1-e6)/a14 if e1 is not None and e6 is not None and a14>0 else 0
    vr=a14/a50
    if vr>=1.10 and drive>=0.60: return 'EXPANSION',a14
    if vr<=0.90 or drive<=0.35: return 'SHORT_RUN',a14
    return 'NORMAL',a14

def weights(r):
    return (0.15,0.15,0.70) if r in ('EXPANSION','SHORT_RUN') else (0.20,0.15,0.65)

def px_entry(t,side): return f(t.ask_price) if side>0 else f(t.bid_price)
def px_exit(t,side): return f(t.bid_price) if side>0 else f(t.ask_price)


def summarize(rows):
    pn=[r['pnl'] for r in rows]; n=len(pn); gp=sum(x for x in pn if x>0); gl=abs(sum(x for x in pn if x<0))
    pf=gp/gl if gl else (math.inf if gp else 0.0); wr=100*sum(x>0 for x in pn)/max(1,n)
    eq=1000.; peak=eq; dd=0.
    for x in pn:
        eq+=x; peak=max(peak,eq); dd=max(dd,(peak-eq)/peak*100 if peak>0 else 0)
    return {'N':n,'WR_pct':wr,'PF':pf,'expectancy':float(np.mean(pn)) if pn else 0.0,'net':float(sum(pn)),'maxDD_pct':dd,
            'avg_mfe':float(np.mean([r['mfe'] for r in rows])) if rows else 0.0,'avg_mae':float(np.mean([r['mae'] for r in rows])) if rows else 0.0}

def simulate(ticks,start,end,mode,stdv_target,stdv_frac,stdv_stop):
    bars=[]; cur=None; rows=[]; active=None; last_signal_bucket=None
    for i in range(start,end):
        t=ticks[i]; ts=int(t.ts_event); m=mid(t); bkt=bucket(ts)
        if cur is None or bkt!=cur.ts:
            if cur is not None: bars.append(cur)
            cur=Bar(bkt,m,m,m,m)
        else:
            cur.h=max(cur.h,m); cur.l=min(cur.l,m); cur.c=m

        if active:
            side=active['side']; x=px_exit(t,side); move=(x-active['avg_entry'])*side
            active['mfe']=max(active['mfe'],move); active['mae']=min(active['mae'],move)
            # close fixed legs on TP/SL
            for leg in active['legs']:
                if leg['closed']: continue
                hit_sl=(x-leg['sl'])*side<=0
                hit_tp=leg['tp'] is not None and (x-leg['tp'])*side>=0
                if hit_sl or hit_tp:
                    ex=leg['sl'] if hit_sl else leg['tp']; leg['pnl']=(ex-leg['entry'])*side*leg['w']; leg['closed']=True
            # ATR runner trail
            if mode in ('ATR','PYR','STDV'):
                rg,a14=regime(bars)
                if a14:
                    k=0.40 if rg=='SHORT_RUN' else 0.50
                    for leg in active['legs']:
                        if leg['kind']=='ATR' and not leg['closed']:
                            ns=x-side*k*a14
                            if side>0: leg['sl']=max(leg['sl'],ns)
                            else: leg['sl']=min(leg['sl'],ns)
            # profit-funded pyramid after F1 closed and ATR positive
            if mode in ('PYR','STDV') and not active['pyr_added']:
                f1=next(z for z in active['legs'] if z['kind']=='F1'); ar=next(z for z in active['legs'] if z['kind']=='ATR')
                if f1['closed'] and not ar['closed'] and (x-ar['entry'])*side>0 and (ar['sl']-ar['entry'])*side>=0:
                    rg,a14=regime(bars)
                    if a14:
                        en=px_entry(t,side); active['legs'].append({'kind':'PYR','entry':en,'w':0.45,'sl':en-side*0.75*a14,'tp':None,'closed':False,'pnl':0.0}); active['pyr_added']=True
            # STDV after F2 closed and ATR positive, once only
            if mode=='STDV' and not active['stdv_added']:
                f2=next(z for z in active['legs'] if z['kind']=='F2'); ar=next(z for z in active['legs'] if z['kind']=='ATR')
                g=build_geom(bars)
                if f2['closed'] and not ar['closed'] and (x-ar['entry'])*side>0 and g and g[0]==side:
                    _,A,B,wave,entry,sl,tp1,tp2,tp3=g
                    target=B+side*stdv_target*wave; ssl=B+side*stdv_stop*wave
                    en=px_entry(t,side)
                    if (target-en)*side>0:
                        if (en-ssl)*side<=0: ssl=tp1
                        active['legs'].append({'kind':'STDV','entry':en,'w':stdv_frac,'sl':ssl,'tp':target,'closed':False,'pnl':0.0}); active['stdv_added']=True
            if all(z['closed'] for z in active['legs']):
                pnl=sum(z['pnl'] for z in active['legs']); rows.append({'pnl':pnl,'mfe':active['mfe'],'mae':active['mae']}); active=None
                continue

        # one signal per completed M5 bar; source is M5-hardcoded
        if active is None and len(bars)>=55 and last_signal_bucket!=bkt:
            g=build_geom(bars)
            if g:
                side,A,B,wave,en0,sl,tp1,tp2,tp3=g; rg,a14=regime(bars)
                if a14:
                    closes=[b.c for b in bars]; e9=ema(closes[-30:],9)
                    pd_ok=e9 is not None and abs(en0-e9)/a14<=1.5
                    en=px_entry(t,side)
                    if pd_ok and abs(en-en0)<=0.35*a14:
                        w1,w2,wa=weights(rg)
                        active={'side':side,'avg_entry':en,'mfe':0.0,'mae':0.0,'pyr_added':False,'stdv_added':False,
                                'legs':[{'kind':'F1','entry':en,'w':w1,'sl':sl,'tp':tp1,'closed':False,'pnl':0.0},
                                        {'kind':'F2','entry':en,'w':w2,'sl':sl,'tp':tp2,'closed':False,'pnl':0.0},
                                        {'kind':'ATR','entry':en,'w':wa,'sl':sl,'tp':None,'closed':False,'pnl':0.0}]}
                        last_signal_bucket=bkt
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--raw-bidask-only',action='store_true')
    a=ap.parse_args();
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    _,ticks=load_all(a.catalog); split=int(len(ticks)*0.40)
    grid=[]
    for mode in ['CORE','ATR','PYR','STDV']:
        params=[(2.65,.24,.55)] if mode!='STDV' else [(t,s,st) for t in (2.45,2.65,2.85) for s in (.18,.24,.30) for st in (.45,.55,.65)]
        for tar,sz,stp in params:
            rows=simulate(ticks,split,len(ticks),mode,tar,sz,stp); s=summarize(rows); s.update({'mode':mode,'stdv_target':tar,'stdv_frac':sz,'stdv_stop':stp}); grid.append(s)
    grid.sort(key=lambda x:(x['expectancy']>0,x['PF'],x['expectancy'],-x['maxDD_pct']),reverse=True)
    out={'experiment_id':a.experiment_id,'source':'AMOS_Fib_v0_30_STDV_LocalOpt.mq5','execution':'raw Bid/Ask tick-by-tick; completed M5 bars only for source-parity indicators/swing geometry','split':'first 40% warmup/train excluded; final 60% chronological OOS','no_ohlc_execution':True,'pd_array_note':'source placeholder reproduced as EMA9-distance gate (<=1.5 ATR effective bound)','research_reference':{'N':390,'WR_pct':66.923,'PF':8.012,'DD_pct':4.0,'Monthly21_pct':99.467},'top20':grid[:20],'all':grid,'verification_level':'RAW_BIDASK_SOURCE_PARITY_CANDIDATE_V1'}
    d=Path('results/amos-fib-v030-rawtick')/a.experiment_id; d.mkdir(parents=True,exist_ok=True); (d/'result.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
