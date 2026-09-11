from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

@dataclass
class Trade:
    variant:str; side:int; entry_time:str; exit_time:str; entry:float; exit:float
    pnl_price:float; r:float; result:str; layer:str; atr:float; regime:str
    mae:float; mfe:float; partial1:bool=False; partial2:bool=False

def load_ticks(catalog_path:str):
    cat=ParquetDataCatalog(catalog_path)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick, identifiers=[inst.id.value])
    rows=[]
    for t in ticks:
        f=lambda x: float(x.as_double()) if hasattr(x,'as_double') else float(x)
        rows.append((pd.to_datetime(t.ts_event,unit='ns',utc=True),f(t.bid_price),f(t.ask_price)))
    d=pd.DataFrame(rows,columns=['time','bid','ask']).sort_values('time').drop_duplicates('time').reset_index(drop=True)
    d['mid']=(d.bid+d.ask)/2; d['spread']=d.ask-d.bid
    return d

def bars(t,mins):
    x=t.set_index('time'); m=x.mid
    o=m.resample(f'{mins}min',label='right',closed='right').ohlc(); o['spread']=x.spread.resample(f'{mins}min',label='right',closed='right').mean()
    return o.dropna().reset_index()

def atr(b,n=14):
    pc=b.close.shift(1); tr=pd.concat([b.high-b.low,(b.high-pc).abs(),(b.low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx_proxy(b,n=14):
    net=(b.close-b.close.shift(n)).abs(); tr=atr(b,n)*n
    return (100*net/(tr+1e-12)).clip(0,100)

def zigzag_points(b,depth=12,deviation_points=5,point=0.01,backstep=3):
    pts=[]; last_hi_i=-10**9; last_lo_i=-10**9
    for i in range(depth,len(b)-2):
        lo=float(b.low.iloc[i]); hi=float(b.high.iloc[i]); w=b.iloc[i-depth+1:i+1]
        if lo <= float(w.low.min()) + 1e-12:
            if i-last_lo_i>backstep or (pts and lo < pts[-1][1]): pts.append((i,lo,'L')); last_lo_i=i
        if hi >= float(w.high.max()) - 1e-12:
            if i-last_hi_i>backstep or (pts and hi > pts[-1][1]): pts.append((i,hi,'H')); last_hi_i=i
    pts.sort(key=lambda z:z[0]); out=[]
    for p in pts:
        if not out or p[2]!=out[-1][2]: out.append(p)
        elif (p[2]=='H' and p[1]>out[-1][1]) or (p[2]=='L' and p[1]<out[-1][1]): out[-1]=p
    return out

def shifted_hour(ts): return int(ts.hour)

def build_entries(ticks):
    h1=bars(ticks,60); m15=bars(ticks,15)
    h1['atr14']=atr(h1,14); h1['atr480']=atr(h1,480); h1['adx']=adx_proxy(h1,14)
    pA=zigzag_points(h1); pB=zigzag_points(m15)
    entries=[]; trade_hours={11,15,16,17,18}; seen=set()
    for layer,b,pts,cap in [('A',h1,pA,5),('B',m15,pB,3)]:
        for i,price,typ in pts:
            if i<2: continue
            ts=b.time.iloc[i]
            if shifted_hour(ts) not in trade_hours: continue
            day=b.time.dt.date.iloc[i]; prior=b[(b.time.dt.date==day) & (b.time<=ts)]
            dhi=float(prior.high.max()); dlo=float(prior.low.min())
            j=max(0,np.searchsorted(h1.time.values,np.datetime64(ts.to_datetime64()),side='right')-1)
            av=float(h1.atr14.iloc[j]) if np.isfinite(h1.atr14.iloc[j]) else 4.0
            al=float(h1.atr480.iloc[j]) if np.isfinite(h1.atr480.iloc[j]) and h1.atr480.iloc[j]>0 else av
            vol=min(max(av/al if al else 1.0,0.6),2.5)
            ad=float(h1.adx.iloc[j]) if np.isfinite(h1.adx.iloc[j]) else 20.0
            regime='TREND' if ad>=25 else ('RANGE' if ad<=18 else 'NEUTRAL')
            off=0.3*vol; mind=0.4*vol
            if typ=='H' and price>dhi+mind: side=1; stop_level=price-off
            elif typ=='L' and price<dlo-mind: side=-1; stop_level=price+off
            else: continue
            key=(str(ts),round(stop_level,3),side,layer)
            if key in seen: continue
            seen.add(key)
            j0=np.searchsorted(ticks.time.values,np.datetime64(ts.to_datetime64()),side='right'); end=ts+pd.Timedelta(hours=24)
            for k in range(j0,len(ticks)):
                q=ticks.iloc[k]
                if q.time>end: break
                if shifted_hour(q.time) not in trade_hours: continue
                if (side==1 and q.ask>=stop_level) or (side==-1 and q.bid<=stop_level):
                    entry=float(q.ask if side==1 else q.bid)
                    tp_mult=1.25 if regime=='TREND' else (0.8 if regime=='RANGE' else 1.0)
                    sl=max(min(av*1.2,12.0),4.0); tp=max(min(av*2.4,30.0),9.0)*tp_mult
                    entries.append(dict(time=q.time,side=side,entry=entry,atr=av,regime=regime,sl=sl,tp=tp,layer=layer,idx=k)); break
    entries.sort(key=lambda x:x['time']); out=[]
    for e in entries:
        if out and e['time']==out[-1]['time'] and e['side']==out[-1]['side'] and abs(e['entry']-out[-1]['entry'])<0.05: continue
        out.append(e)
    return out

def simulate_native(ticks,e):
    side=e['side']; entry=e['entry']; sl0=e['sl']; tpdist=e['tp']; idx=e['idx']; av=e['atr']
    sl=entry-side*sl0; tp=entry+side*tpdist; risk=sl0; best=entry; mae=mfe=0.0; k=idx+1
    for k in range(idx+1,len(ticks)):
        q=ticks.iloc[k]; px=float(q.bid if side==1 else q.ask); ex=side*(px-entry); mfe=max(mfe,ex); mae=min(mae,ex)
        if ex>=1.2:
            be=entry+side*0.2; sl=max(sl,be) if side==1 else min(sl,be)
        if ex>=2.0:
            best=max(best,px) if side==1 else min(best,px); nsl=best-side*3.0; sl=max(sl,nsl) if side==1 else min(sl,nsl)
        stop_hit=(px<=sl) if side==1 else (px>=sl); tp_hit=(px>=tp) if side==1 else (px<=tp)
        if stop_hit or tp_hit:
            exitp=px; rr=side*(exitp-entry)/risk
            return Trade('A_NATIVE',side,str(e['time']),str(q.time),entry,exitp,side*(exitp-entry),rr,'TP' if tp_hit else 'SL',e['layer'],av,e['regime'],mae,mfe)
        if q.time-e['time']>pd.Timedelta(days=3): break
    q=ticks.iloc[min(len(ticks)-1,k)]; px=float(q.bid if side==1 else q.ask); rr=side*(px-entry)/risk
    return Trade('A_NATIVE',side,str(e['time']),str(q.time),entry,px,side*(px-entry),rr,'TIME',e['layer'],av,e['regime'],mae,mfe)

def simulate_adaptive(ticks,e):
    side=e['side']; entry=e['entry']; idx=e['idx']; av=e['atr']; risk=max(1.5*av,0.01)
    sl=entry-side*risk; mae=mfe=0.0; t1=entry+side*1.0*av; t2=entry+side*2.0*av
    rem=1.0; pnl=0.0; p1=p2=False; best=entry; k=idx+1
    for k in range(idx+1,len(ticks)):
        q=ticks.iloc[k]; px=float(q.bid if side==1 else q.ask); ex=side*(px-entry); mfe=max(mfe,ex); mae=min(mae,ex)
        if not p1 and ((px>=t1) if side==1 else (px<=t1)):
            pnl += 0.40*side*(px-entry); rem-=0.40; p1=True; sl=entry+side*float(q.spread)
        if p1 and not p2 and ((px>=t2) if side==1 else (px<=t2)):
            pnl += 0.30*side*(px-entry); rem-=0.30; p2=True
        if p2:
            best=max(best,px) if side==1 else min(best,px); nsl=best-side*av
            if side==1 and nsl>sl+0.2*av: sl=nsl
            if side==-1 and nsl<sl-0.2*av: sl=nsl
        stop_hit=(px<=sl) if side==1 else (px>=sl)
        if stop_hit:
            pnl += rem*side*(px-entry); rr=pnl/risk
            return Trade('B_ADAPTIVE_REPRO',side,str(e['time']),str(q.time),entry,px,pnl,rr,'SL/TRAIL',e['layer'],av,e['regime'],mae,mfe,p1,p2)
        if q.time-e['time']>pd.Timedelta(days=3): break
    q=ticks.iloc[min(len(ticks)-1,k)]; px=float(q.bid if side==1 else q.ask); pnl+=rem*side*(px-entry); rr=pnl/risk
    return Trade('B_ADAPTIVE_REPRO',side,str(e['time']),str(q.time),entry,px,pnl,rr,'TIME',e['layer'],av,e['regime'],mae,mfe,p1,p2)

def metrics(ts):
    r=np.array([x.r for x in ts],float); p=np.array([x.pnl_price for x in ts],float); n=len(r)
    wins=int((p>0).sum()); wr=100*wins/n if n else 0.0; gp=p[p>0].sum() if n else 0.; gl=-p[p<0].sum() if n else 0.; pf=gp/gl if gl>0 else (999 if gp>0 else 0)
    curve=np.cumsum(p) if n else np.array([0.]); peak=np.maximum.accumulate(curve); dd=peak-curve; mdd=float(dd.max()) if len(dd) else 0.; net=float(p.sum()) if n else 0.
    return dict(N=n,WR_pct=wr,PF=float(pf),net_price=net,max_DD_price=mdd,RF=net/mdd if mdd>0 else (999 if net>0 else 0),avg_R=float(r.mean()) if n else 0.,avg_MAE=float(np.mean([x.mae for x in ts])) if n else 0.,avg_MFE=float(np.mean([x.mfe for x in ts])) if n else 0.)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    ticks=load_ticks(a.catalog); entries=build_entries(ticks)
    A=[simulate_native(ticks,e) for e in entries]; B=[simulate_adaptive(ticks,e) for e in entries]
    summary={'verification_level':'RAW_BIDASK_EXIT_AB_REPRODUCTION','raw_ticks':len(ticks),'period_start':str(ticks.time.iloc[0]),'period_end':str(ticks.time.iloc[-1]),'entry_count':len(entries),'A_GoldeBrave_native':metrics(A),'B_AdaptiveCloser_public_reproduction':metrics(B),'limitations':['GoldeBrave ZigZag is deterministic approximation of Examples\\ZigZag; Layer C omitted from parity-grade entry generation','Adaptive Closer is EX5-only: public/default feature reproduction, not byte-identical EX5','Adaptive target distances are not publicly documented; reproduction uses TP1=1 ATR, TP2=2 ATR, runner=1 ATR step trail','Final deployment decision requires MT5 real-tick tester parity run']}
    pd.DataFrame([asdict(x) for x in A+B]).to_csv(out/'trades_ab.csv',index=False)
    pd.DataFrame(entries).drop(columns=['idx']).to_csv(out/'entries.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
