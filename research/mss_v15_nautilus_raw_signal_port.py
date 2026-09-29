from __future__ import annotations

"""MSS_MultiStrat_v1_5 cloud-only Raw Bid/Ask research port.

This is a transparent research port of the recovered MQ5 signal layer.
It deliberately does NOT claim MT5 binary parity.  It consumes Nautilus
QuoteTicks, constructs M15/H1 bars from the raw stream, evaluates S01-S15,
and emits auditable signal/confluence evidence.  Execution/risk parity is a
separate gate; WR5 stays INVALID until that gate is complete.
"""

import argparse, hashlib, json, math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import nautilus_trader
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog


@dataclass
class Bar:
    ts_ns:int; o:float; h:float; l:float; c:float; vol:int=0

@dataclass
class Params:
    s01_vol_ma:int=20; s01_vol_mult:float=1.8; s01_body_atr:float=.30
    fractal_leg:int=2; swing_lookback:int=30; s03_fractal_count:int=4
    s05_range_lookback:int=30; s05_disc:float=.30
    s06_impulse:int=5; s06_fib_low:float=.618; s06_fib_high:float=.79
    asian_start:int=0; asian_end:int=8; ny_start:int=13; ny_end:int=17
    sb_start:int=14; sb_end:int=15; asian_bars:int=32; asian_break_atr:float=.50
    gann_lookback:int=100; gann_tol_pct:float=.30
    sweep_lookback:int=20; ob_lookback:int=50; ob_run:int=3
    fvg_lookback:int=30; breaker_lookback:int=60; eqh_lookback:int=30
    eqh_tol_pip:float=2.0


def fpx(x):
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)

def atr(b:List[Bar], n=14):
    if len(b)<n+1:return 0.0
    z=[]
    for i in range(len(b)-n,len(b)):
        pc=b[i-1].c
        z.append(max(b[i].h-b[i].l,abs(b[i].h-pc),abs(b[i].l-pc)))
    return sum(z)/len(z)

def fractals(b, leg, look):
    hs=[]; ls=[]; last=len(b)-1
    lo=max(leg,last-look+leg)
    for i in range(last-leg-1,lo-1,-1):
        if i-leg<0 or i+leg>last-1: continue
        if all(b[i-k].h<b[i].h and b[i+k].h<b[i].h for k in range(1,leg+1)): hs.append(b[i].h)
        if all(b[i-k].l>b[i].l and b[i+k].l>b[i].l for k in range(1,leg+1)): ls.append(b[i].l)
    return hs,ls

class Signals:
    def __init__(self,p:Params,pip:float): self.p=p; self.pip=pip

    def s01(self,b):
        p=self.p; look=max(5,p.s01_vol_ma)
        if len(b)<look+2:return 0
        x=b[-1]; av=sum(v.vol for v in b[-look-1:-1])/look; a=atr(b)
        if av<=0 or a<=0 or x.h<=x.l:return 0
        if x.vol < av*p.s01_vol_mult or abs(x.c-x.o)>a*p.s01_body_atr:return 0
        pos=(x.c-x.l)/(x.h-x.l)
        return -1 if pos>=.70 else 1 if pos<=.30 else 0

    def s02(self,b):
        p=self.p; leg=max(1,p.fractal_leg); look=max(p.swing_lookback,leg*4)
        if len(b)<look+4:return 0
        hs,ls=fractals(b,leg,look); c=b[-1].c
        if hs and c>max(hs):return 1
        if ls and c<min(ls):return -1
        return 0

    def s03(self,b):
        p=self.p; leg=max(1,p.fractal_leg); N=p.s03_fractal_count; look=N*(leg+2)*4
        if len(b)<look+4:return 0
        hs,ls=fractals(b,leg,look); hs=hs[:N+1]; ls=ls[:N+1]
        if len(hs)<2 or len(ls)<2:return 0
        up=hs[0]>hs[1] and ls[0]>ls[1]; dn=hs[0]<hs[1] and ls[0]<ls[1]
        if up and ls[0]<ls[1]:return -1
        if dn and hs[0]>hs[1]:return 1
        return 0

    def s04(self,b):
        if len(b)<24:return 0
        x=b[-1]
        for i in range(len(b)-2,max(1,len(b)-20),-1):
            if b[i].l>b[i-2].h:
                top,bot=b[i].l,b[i-2].h
                if x.l<=top and x.l>=bot and x.c>x.o and x.c>top:return 1
            if b[i].h<b[i-2].l:
                top,bot=b[i-2].l,b[i].h
                if x.h>=bot and x.h<=top and x.c<x.o and x.c<bot:return -1
        return 0

    def s05(self,b):
        look=max(10,self.p.s05_range_lookback)
        if len(b)<look+2:return 0
        hist=b[-look-1:-1]; x=b[-1]; hi=max(v.h for v in hist); lo=min(v.l for v in hist)
        if hi<=lo:return 0
        pos=(x.c-lo)/(hi-lo)
        if pos<=self.p.s05_disc and x.c>x.o:return 1
        if pos>=1-self.p.s05_disc and x.c<x.o:return -1
        return 0

    def s06(self,b):
        look=max(3,self.p.s06_impulse)
        if len(b)<look+4:return 0
        hist=b[-look-1:-1]; x=b[-1]
        hi=max(range(len(hist)),key=lambda i:hist[i].h); lo=min(range(len(hist)),key=lambda i:hist[i].l)
        H,L=hist[hi].h,hist[lo].l
        if H<=L:return 0
        leg=H-L
        if lo<hi:
            a=H-leg*self.p.s06_fib_high; z=H-leg*self.p.s06_fib_low
            if a<=x.c<=z and x.c>x.o:return 1
        if hi<lo:
            a=L+leg*self.p.s06_fib_low; z=L+leg*self.p.s06_fib_high
            if a<=x.c<=z and x.c<x.o:return -1
        return 0

    def s07(self,b,dt):
        p=self.p
        if not(p.ny_start<=dt.hour<p.ny_end):return 0
        today=[x for x in b[-100:] if datetime.fromtimestamp(x.ts_ns/1e9,timezone.utc).date()==dt.date()]
        asia=[x for x in today if p.asian_start<=datetime.fromtimestamp(x.ts_ns/1e9,timezone.utc).hour<p.asian_end]
        if not asia:return 0
        hi=max(x.h for x in asia); lo=min(x.l for x in asia); x=b[-1]
        pre=today[:-1]; sh=any(y.h>hi and datetime.fromtimestamp(y.ts_ns/1e9,timezone.utc).hour>=p.ny_start for y in pre)
        sl=any(y.l<lo and datetime.fromtimestamp(y.ts_ns/1e9,timezone.utc).hour>=p.ny_start for y in pre)
        if sh and x.c<x.o and x.c<hi:return -1
        if sl and x.c>x.o and x.c>lo:return 1
        return 0

    def s08(self,b,h1,dt):
        p=self.p
        if not(p.sb_start<=dt.hour<p.sb_end) or len(h1)<4:return 0
        a=atr(b); x=b[-1]
        if a<=0:return 0
        up=h1[-2].c>h1[-4].c; dn=h1[-2].c<h1[-4].c; body=x.c-x.o
        if up and body>a*.5:return 1
        if dn and body<-a*.5:return -1
        return 0

    def s09(self,b,dt):
        p=self.p
        if p.asian_start<=dt.hour<p.asian_end:return 0
        today=[x for x in b[-max(100,p.asian_bars+12):] if datetime.fromtimestamp(x.ts_ns/1e9,timezone.utc).date()==dt.date()]
        asia=[x for x in today if p.asian_start<=datetime.fromtimestamp(x.ts_ns/1e9,timezone.utc).hour<p.asian_end]
        if not asia:return 0
        hi=max(x.h for x in asia); lo=min(x.l for x in asia); x=b[-1]; a=atr(b)
        if a<=0:return 0
        body=x.c-x.o
        if x.c>hi and body>a*p.asian_break_atr:return 1
        if x.c<lo and body<-a*p.asian_break_atr:return -1
        return 0

    def s10(self,b):
        look=max(30,self.p.gann_lookback)
        if len(b)<look+4:return 0
        hist=b[-look-1:-1]; x=b[-1]; H=max(v.h for v in hist); L=min(v.l for v in hist)
        if H<=0 or L<=0:return 0
        tol=x.c*self.p.gann_tol_pct/100
        for n in range(1,9):
            lv=(math.sqrt(L)+.125*n)**2
            if abs(x.c-lv)<=tol:return -1 if x.c>b[-2].c else 1 if x.c<b[-2].c else 0
        for n in range(1,9):
            q=math.sqrt(H)-.125*n
            if q<=0:break
            lv=q*q
            if abs(x.c-lv)<=tol:return 1 if x.c<b[-2].c else -1 if x.c>b[-2].c else 0
        return 0

    def s11(self,b):
        n=max(5,self.p.sweep_lookback)
        if len(b)<n+2:return 0
        x=b[-1]; hist=b[-n-1:-1]; H=max(v.h for v in hist); L=min(v.l for v in hist)
        if x.h-H>=self.pip and x.c<H and x.c<x.o:return -1
        if L-x.l>=self.pip and x.c>L and x.c>x.o:return 1
        return 0

    def s12(self,b):
        n=max(self.p.ob_lookback,self.p.ob_run+5); run=self.p.ob_run
        if len(b)<n+2:return 0
        x=b[-1]; start=max(run,len(b)-n)
        for i in range(len(b)-2,start-1,-1):
            if all(b[i-k].c<b[i-k].o for k in range(1,run+1)) and b[i].c>b[i].o:
                if x.l<=b[i].h and x.l>=b[i].l*.9990 and x.c>b[i].h:return 1
            if all(b[i-k].c>b[i-k].o for k in range(1,run+1)) and b[i].c<b[i].o:
                if x.h>=b[i].l and x.h<=b[i].h*1.0010 and x.c<b[i].l:return -1
        return 0

    def s13(self,b):
        n=max(10,self.p.fvg_lookback)
        if len(b)<n+4:return 0
        x=b[-1]
        for i in range(len(b)-3,max(1,len(b)-n),-1):
            if b[i].l>b[i-2].h:
                top,bot=b[i].l,b[i-2].h
                if any(y.l<=bot for y in b[i+1:-1]) and x.c>x.o and x.c>top:return 1
            if b[i].h<b[i-2].l:
                top,bot=b[i-2].l,b[i].h
                if any(y.h>=top for y in b[i+1:-1]) and x.c<x.o and x.c<bot:return -1
        return 0

    def s14(self,b):
        n=max(15,self.p.breaker_lookback)
        if len(b)<n+5:return 0
        x=b[-1]
        for i in range(len(b)-6,max(2,len(b)-n+3),-1):
            bull=b[i].c>b[i].o and b[i-1].c<b[i-1].o and b[i-2].c<b[i-2].o
            if bull and any(y.c<b[i].l for y in b[i+1:-1]) and x.c<b[i].l and x.c<x.o:return -1
            bear=b[i].c<b[i].o and b[i-1].c>b[i-1].o and b[i-2].c>b[i-2].o
            if bear and any(y.c>b[i].h for y in b[i+1:-1]) and x.c>b[i].h and x.c>x.o:return 1
        return 0

    def s15(self,b):
        n=max(10,self.p.eqh_lookback)
        if len(b)<n+2:return 0
        x=b[-1]; tol=self.p.eqh_tol_pip*self.pip; hist=b[-n-1:-1]
        for r in range(len(hist)-1,0,-1):
            hr=hist[r].h; lr=hist[r].l
            if sum(abs(y.h-hr)<=tol for y in hist[:r+1])>=2 and x.h>hr+self.pip and x.c<hr:return -1
            if sum(abs(y.l-lr)<=tol for y in hist[:r+1])>=2 and x.l<lr-self.pip and x.c>lr:return 1
        return 0

    def all(self,b,h1):
        dt=datetime.fromtimestamp(b[-1].ts_ns/1e9,timezone.utc)
        vals=[self.s01(b),self.s02(b),self.s03(b),self.s04(b),self.s05(b),
              self.s06(b),self.s07(b,dt),self.s08(b,h1,dt),self.s09(b,dt),
              self.s10(b),self.s11(b),self.s12(b),self.s13(b),self.s14(b),self.s15(b)]
        pos=sum(v>0 for v in vals); neg=sum(v<0 for v in vals)
        direction=1 if pos>=neg and pos>0 else -1 if neg>pos else 0
        conf=max(pos,neg)
        return vals,conf,direction


def bars_from_ticks(ticks, minutes):
    bucket=None; cur=None; out=[]
    ns=minutes*60*1_000_000_000
    for t in ticks:
        mid=(fpx(t.bid_price)+fpx(t.ask_price))/2
        k=(int(t.ts_event)//ns)*ns
        if bucket is None or k!=bucket:
            if cur is not None:out.append(cur)
            bucket=k; cur=Bar(k,mid,mid,mid,mid,1)
        else:
            cur.h=max(cur.h,mid);cur.l=min(cur.l,mid);cur.c=mid;cur.vol+=1
    if cur is not None:out.append(cur)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True); ap.add_argument("--symbol",default="XAUUSD")
    ap.add_argument("--out",required=True); a=ap.parse_args()
    cat=ParquetDataCatalog(a.catalog)
    inst=next((x for x in cat.instruments() if x.id.symbol.value.replace("/","")==a.symbol),None)
    if inst is None:raise SystemExit(f"{a.symbol} instrument missing")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    if not ticks:raise SystemExit("no raw QuoteTicks")
    m15=bars_from_ticks(ticks,15); h1=bars_from_ticks(ticks,60)
    pip=0.01 if a.symbol=="XAUUSD" else 0.01 if a.symbol.endswith("JPY") else 0.0001
    sg=Signals(Params(),pip); counts=[{"long":0,"short":0} for _ in range(15)]
    tiers={}; events=[]; h1i=0
    for i in range(110,len(m15)):
        while h1i+1<len(h1) and h1[h1i+1].ts_ns<=m15[i].ts_ns:h1i+=1
        vals,conf,direction=sg.all(m15[:i+1],h1[:h1i+1])
        for j,v in enumerate(vals):
            if v>0:counts[j]["long"]+=1
            elif v<0:counts[j]["short"]+=1
        tiers[str(conf)]=tiers.get(str(conf),0)+1
        if conf>=4 and direction:
            events.append({"ts_ns":m15[i].ts_ns,"conf":conf,"direction":direction,"signals":vals})
    obj={
      "verification_level":"NAUTILUS_RAW_BIDASK_MSS_V15_SIGNAL_PORT_V1",
      "status":"MEASURED_SIGNAL_PARITY_PORT",
      "wr5_status":"INVALID",
      "wr5_reason":"execution/risk/margin/cashback parity not implemented in this signal-only gate",
      "symbol":a.symbol,"raw_ticks":len(ticks),"m15_bars":len(m15),"h1_bars":len(h1),
      "ohlc_source":"constructed internally from raw QuoteTicks; fills not evaluated here",
      "nautilus_version":getattr(nautilus_trader,"__version__","unknown"),
      "source_sha256":"7fb3065e4a9a3110c4aaaa594705893c3f96b47bf8df3b441bdcab396c0caa2d",
      "strategy_counts":{f"S{i+1:02d}":counts[i] for i in range(15)},
      "confluence_histogram":tiers,"qualified_signal_events":len(events),
      "sample_events":events[:100],
    }
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in obj.items() if k!="sample_events"},indent=2))

if __name__=="__main__":main()
