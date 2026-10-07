#!/usr/bin/env python3
"""AMOS v1.14: restore intended entry architecture.
M5 corrected MSS+ ordered context -> M1 Close-line POI pattern -> Raw Bid/Ask entry signal.
M15 AMD remains a separate upstream branch and is not silently mixed into this normal setup audit.

M1 six-pattern image interpretation, frozen from prior v19:
Classic V Buy / Classic A Sell
Quasimodo Buy / Sell
Open-Close-Level Buy / Sell
Structure is read from M1 Close line. POI touch is raw price. Entry requires a favorable M1 Close
reaction after the POI touch. The entry timestamp is the first raw quote after that confirming M1 close.
"""
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

def fpx(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def atr(d,n=14):
    p=d.close.shift(1)
    tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()
def feature(d):
    z=d.copy();z["atr"]=atr(z);z["body"]=(z.close-z.open).abs();z["disp"]=z.body/(z.atr+1e-12)
    z["vma"]=z.volume.rolling(20).mean()
    z["swing_lo"]=z.low.shift(1).rolling(8).min();z["swing_hi"]=z.high.shift(1).rolling(8).max()
    z["bull_fvg"]=z.low>z.high.shift(2);z["bear_fvg"]=z.high<z.low.shift(2)
    return z
def fvg_zone(z,i,side):
    r=z.iloc[i]
    if side==1 and bool(r.bull_fvg):
        lo=float(z.high.iloc[i-2]);hi=float(r.low);return min(lo,hi),max(lo,hi),"FVG"
    if side==-1 and bool(r.bear_fvg):
        lo=float(r.high);hi=float(z.low.iloc[i-2]);return min(lo,hi),max(lo,hi),"FVG"
    return None
def prior_fvgs(z,i,side,lookback=12):
    out=[]
    for k in range(max(2,i-lookback),i):
        q=z.iloc[k]
        if side==1 and bool(q.bear_fvg):
            lo=float(q.high);hi=float(z.low.iloc[k-2]);out.append((k,min(lo,hi),max(lo,hi)))
        elif side==-1 and bool(q.bull_fvg):
            lo=float(z.high.iloc[k-2]);hi=float(q.low);out.append((k,min(lo,hi),max(lo,hi)))
    return out
def poi_zone(z,i,side):
    cur=fvg_zone(z,i,side);inv=[];r=z.iloc[i]
    for k,lo,hi in prior_fvgs(z,i,side):
        if (float(r.close)>hi if side==1 else float(r.close)<lo):inv.append((k,lo,hi))
    if cur and inv:
        clo,chi,_=cur
        ov=[(max(clo,lo),min(chi,hi)) for _,lo,hi in inv if max(clo,lo)<=min(chi,hi)]
        if ov:return (*ov[-1],"BPR")
    if cur:return cur
    if inv:
        _,lo,hi=inv[-1];return lo,hi,"IFVG"
    return None
def eqzone(di,se,de):
    if di<0:
        x=se-de;return (de+x*.50,de+x*.79) if x>0 else None
    x=de-se;return (de-x*.79,de-x*.50) if x>0 else None
def bars(raw,minutes):
    g=raw.set_index("datetime").bid.resample(f"{minutes}min")
    return pd.DataFrame({"open":g.first(),"high":g.max(),"low":g.min(),"close":g.last(),"volume":g.count()}).dropna().reset_index()

def m5_context(z):
    c=Counter();states={1:None,-1:None};down=[];contexts=[]
    for i in range(30,len(z)):
        r=z.iloc[i];p=z.iloc[i-1]
        if not np.isfinite(r.atr) or r.atr<=0:continue
        for side in (1,-1):
            s=states[side]
            if s and i-s["sweep_i"]>28:states[side]=None
        bull=bool(r.low<r.swing_lo and r.close>r.swing_lo);bear=bool(r.high>r.swing_hi and r.close<r.swing_hi)
        if bull:states[1]={"sweep_i":i,"extreme":float(r.low),"mss_ref":float(z.high.iloc[max(0,i-3):i].max()),"cisd_i":None,"mss_i":None,"disp_seen":False};c["sweep"]+=1
        if bear:states[-1]={"sweep_i":i,"extreme":float(r.high),"mss_ref":float(z.low.iloc[max(0,i-3):i].min()),"cisd_i":None,"mss_i":None,"disp_seen":False};c["sweep"]+=1
        for side in (1,-1):
            s=states[side]
            if not s:continue
            if s["cisd_i"] is None:
                if i>s["sweep_i"] and i-s["sweep_i"]<=8:
                    hit=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
                    if hit:s["cisd_i"]=i;c["cisd"]+=1
                elif i-s["sweep_i"]>8:states[side]=None
                continue
            if s["mss_i"] is None:
                if i-s["cisd_i"]>12:states[side]=None;continue
                if i<=s["cisd_i"]:continue
                hit=(r.close>s["mss_ref"]) if side==1 else (r.close<s["mss_ref"])
                if hit:s["mss_i"]=i;c["mss"]+=1
                else:continue
            if i<s["mss_i"] or i-s["mss_i"]>4:
                if i-s["mss_i"]>4:states[side]=None
                continue
            if float(r.disp)<.65:continue
            if not s["disp_seen"]:c["displacement"]+=1;s["disp_seen"]=True
            zone=poi_zone(z,i,side)
            if zone is None:continue
            c["mss_plus"]+=1
            down.append({"i":i,"side":side,"se":float(s["extreme"]),"de":float(r.close),"zone":zone,"stage":"volume","age":0})
            states[side]=None
    for s in down:
        for j in range(s["i"]+1,min(len(z),s["i"]+1+8+12+8+3)):
            b=z.iloc[j];s["age"]+=1
            if s["stage"]=="volume":
                vr=float(b.volume/b.vma) if np.isfinite(b.vma) and b.vma>0 else np.nan
                if np.isfinite(vr) and vr>=1.10:c["volume"]+=1;s["stage"]="equilibrium";s["age"]=0
                elif s["age"]>8:break
            elif s["stage"]=="equilibrium":
                zone=eqzone(s["side"],s["se"],s["de"])
                if zone is not None:c["equilibrium"]+=1;s["eq"]=zone;s["stage"]="pullback";s["age"]=0
                elif s["age"]>12:break
            elif s["stage"]=="pullback":
                lo,hi=sorted(s["eq"])
                if b.high>=lo and b.low<=hi:
                    c["pullback"]+=1
                    contexts.append({"ready_time":pd.Timestamp(b.datetime)+pd.Timedelta(minutes=5),"dir":int(s["side"]),
                                     "m5_stop":float(s["se"]),"m5_poi_type":str(s["zone"][2])})
                    break
                if s["age"]>8:break
    return pd.DataFrame(contexts),dict(c)

def m1_setups(z):
    """Causal six-pattern setup detector. Pivot is confirmed two bars after it occurs."""
    piv=[];seen=set();setups=[]
    for i in range(24,len(z)):
        idx=i-2;c0=float(z.close.iloc[idx])
        if c0<float(z.close.iloc[idx-2:idx].min()) and c0<=float(z.close.iloc[idx+1:idx+3].min()):
            piv.append({"type":"L","idx":idx,"price":c0})
        elif c0>float(z.close.iloc[idx-2:idx].max()) and c0>=float(z.close.iloc[idx+1:idx+3].max()):
            piv.append({"type":"H","idx":idx,"price":c0})
        piv=piv[-80:]
        r=z.iloc[i];a=float(r.atr)
        if not np.isfinite(a) or a<=0:continue
        c=float(r.close);pc=float(z.close.iloc[i-1])
        def add(name,di,poi,sl,key):
            if key in seen or not math.isfinite(poi+sl):return
            if (di>0 and not sl<poi) or (di<0 and not sl>poi):return
            seen.add(key);setups.append({"pattern":name,"dir":di,"poi":float(poi),"sl":float(sl),
                                         "atr":a,"signal_time":pd.Timestamp(r.datetime)})
        lows=[p for p in piv if p["type"]=="L" and 2<=i-p["idx"]<=8]
        if lows:
            p=lows[-1]
            if c>pc and c>float(z.close.iloc[i-2]):
                sl=float(z.low.iloc[max(0,p["idx"]-1):p["idx"]+2].min())-.10*a
                add("CLASSIC_V_BUY",1,p["price"],sl,f"CVB:{p['idx']}")
        highs=[p for p in piv if p["type"]=="H" and 2<=i-p["idx"]<=8]
        if highs:
            p=highs[-1]
            if c<pc and c<float(z.close.iloc[i-2]):
                sl=float(z.high.iloc[max(0,p["idx"]-1):p["idx"]+2].max())+.10*a
                add("CLASSIC_A_SELL",-1,p["price"],sl,f"CAS:{p['idx']}")
        if len(piv)>=3:
            x,y,w=piv[-3],piv[-2],piv[-1]
            if x["type"]=="L" and y["type"]=="H" and w["type"]=="L" and w["price"]<x["price"]-.05*a and c>y["price"]:
                sl=float(z.low.iloc[max(0,w["idx"]-1):w["idx"]+2].min())-.10*a
                add("QM_BUY",1,x["price"],sl,f"QMB:{x['idx']}:{w['idx']}")
            if x["type"]=="H" and y["type"]=="L" and w["type"]=="H" and w["price"]>x["price"]+.05*a and c<y["price"]:
                sl=float(z.high.iloc[max(0,w["idx"]-1):w["idx"]+2].max())+.10*a
                add("QM_SELL",-1,x["price"],sl,f"QMS:{x['idx']}:{w['idx']}")
        po=float(z.open.iloc[i-1]);ph=float(z.high.iloc[i-1]);pl=float(z.low.iloc[i-1]);pcl=float(z.close.iloc[i-1])
        avg=float((z.close.iloc[i-20:i]-z.open.iloc[i-20:i]).abs().mean());body=abs(c-float(r.open))
        if avg>0 and body>=avg:
            local_lo=float(z.low.iloc[i-5:i+1].min());local_hi=float(z.high.iloc[i-5:i+1].max())
            if pcl<po and c>float(r.open) and c>po and min(float(r.low),pl)<=local_lo+.15*a:
                add("OCL_BUY",1,max(po,pcl),min(local_lo,pl)-.10*a,f"OCLB:{i-1}")
            if pcl>po and c<float(r.open) and c<po and max(float(r.high),ph)>=local_hi-.15*a:
                add("OCL_SELL",-1,min(po,pcl),max(local_hi,ph)+.10*a,f"OCLS:{i-1}")
    return pd.DataFrame(setups)

def pair_entries(ctx,st,z1,raw,max_wait_min=60):
    if ctx.empty or st.empty:return pd.DataFrame(),Counter()
    stats=Counter();out=[]
    for _,cx in ctx.iterrows():
        t0=pd.Timestamp(cx.ready_time);t1=t0+pd.Timedelta(minutes=max_wait_min);di=int(cx.dir)
        cand=st[(st.dir==di)&(st.signal_time<=t1)&(st.signal_time>=t0-pd.Timedelta(minutes=30))].copy()
        found=None
        for _,s in cand.sort_values("signal_time").iterrows():
            start=max(t0,pd.Timestamp(s.signal_time))
            q=z1[(z1.datetime>=start)&(z1.datetime<=t1)]
            touched=False;touch_time=None
            for k in range(1,len(q)):
                b=q.iloc[k];prev=q.iloc[k-1];tol=.10*float(s.atr)
                if not touched:
                    touch=(float(b.low)<=float(s.poi)+tol) if di>0 else (float(b.high)>=float(s.poi)-tol)
                    invalid=(float(b.low)<=float(s.sl)) if di>0 else (float(b.high)>=float(s.sl))
                    if invalid:break
                    if touch:touched=True;touch_time=pd.Timestamp(b.datetime);stats["poi_touch"]+=1
                if touched:
                    confirm=(float(b.close)>float(s.poi) and float(b.close)>float(prev.close)) if di>0 else (float(b.close)<float(s.poi) and float(b.close)<float(prev.close))
                    if confirm:
                        confirm_end=pd.Timestamp(b.datetime)+pd.Timedelta(minutes=1)
                        rr=raw[raw.datetime>=confirm_end]
                        if rr.empty:break
                        first=rr.iloc[0]
                        found={"context_ready":str(t0),"pattern":s.pattern,"dir":di,"poi":float(s.poi),
                               "line_sl":float(s.sl),"touch_time":str(touch_time),"confirm_bar_time":str(b.datetime),
                               "entry_time":str(first.datetime),"entry_ns":int(pd.Timestamp(first.datetime).value),
                               "bid":float(first.bid),"ask":float(first.ask),"m5_stop":float(cx.m5_stop),
                               "m5_poi_type":cx.m5_poi_type}
                        stats["confirmed"]+=1;stats["pattern_"+str(s.pattern)]+=1;break
            if found:break
        if found:out.append(found)
        else:stats["no_m1_entry"]+=1
    return pd.DataFrame(out),stats

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
    bid=np.fromiter((fpx(x.bid_price) for x in ticks),dtype=float,count=len(ticks))
    ask=np.fromiter((fpx(x.ask_price) for x in ticks),dtype=float,count=len(ticks))
    raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"bid":bid,"ask":ask})
    z5=feature(bars(raw,5));z1=feature(bars(raw,1))
    ctx,counts=m5_context(z5);st=m1_setups(z1);entries,stats=pair_entries(ctx,st,z1,raw)
    ctx.to_csv(out/"m5_contexts.csv",index=False);st.to_csv(out/"m1_line_setups.csv",index=False);entries.to_csv(out/"entry_signals.csv",index=False)
    res={"version":"v1.14","architecture":"M5 corrected MSS+ ordered context -> M1 Close-line POI -> Raw BidAsk entry",
         "m15_amd":"separate upstream branch; not mixed into this normal-setup audit",
         "raw_ticks":len(ticks),"m5_counts":counts,"m5_context_ready":len(ctx),"m1_setups":len(st),
         "entry_signals":len(entries),"entry_stats":dict(stats),
         "entry_rule":"same-direction M1 V/A/QM/OCL setup; raw POI revisit after M5 context ready; favorable M1 Close reaction; first subsequent raw quote is executable entry",
         "patterns":["CLASSIC_V_BUY","CLASSIC_A_SELL","QM_BUY","QM_SELL","OCL_BUY","OCL_SELL"],
         "note":"This restores line chart as the entry locator. It is a signal audit; native Nautilus execution is the next stage."}
    (out/"result.json").write_text(json.dumps(res,indent=2),encoding="utf-8");print(json.dumps(res,indent=2))
if __name__=="__main__":main()
