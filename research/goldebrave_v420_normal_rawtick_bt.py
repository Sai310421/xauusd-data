from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np, pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

# GoldeBrave v4.20 defaults, ordinary full-strategy BT.
TRADE_HOURS={11,15,16,17,18}; UNIT=0.10; FS=3.0; MIND_ADD=1.0
SL_ATR=1.2; TP_ATR=2.4; SL_MIN=4.0; TP_MIN=9.0; SL_MAX=12.0; TP_MAX=30.0
MIN_ENTRIES_DAY=3; BOOST_HOUR=9; BOOST_OFF=0.2; SPREAD_BREAK_UNITS=25
MAX_PENDING_SIDE=4; BE_TRIG=1.2; BE_LOCK=0.2; TR_TRIG=2.0; TR_OFF=3.0; TR_BAND=0.3
VOL_MIN=.6; VOL_MAX=2.5; ADX_TREND=25.; ADX_RANGE=18.; TREND_TP=1.25; RANGE_TP=.8; RANGE_BOOST=2.0
LOT=0.10; CONTRACT=100.0

@dataclass
class Pending:
    side:int; price:float; sl:float; tp:float; layer:str; created:str
@dataclass
class Pos:
    side:int; entry:float; sl:float; tp:float; layer:str; entry_time:str; mae:float=0.; mfe:float=0.
@dataclass
class Trade:
    side:int; layer:str; entry_time:str; exit_time:str; entry:float; exit:float; pnl_price:float; pnl_usd:float; result:str; mae:float; mfe:float

def fpx(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def load_ticks(path):
    cat=ParquetDataCatalog(path); inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ts=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    d=pd.DataFrame([(int(t.ts_event),fpx(t.bid_price),fpx(t.ask_price)) for t in ts],columns=['ns','bid','ask'])
    d=d.sort_values('ns').drop_duplicates('ns').reset_index(drop=True); d['time']=pd.to_datetime(d.ns,unit='ns',utc=True); d['mid']=(d.bid+d.ask)/2; return d

def mkbars(ticks,mins):
    x=ticks.set_index('time'); o=x.mid.resample(f'{mins}min',label='left',closed='left').ohlc(); return o.dropna().reset_index()
def atr(b,n):
    pc=b.close.shift(); tr=pd.concat([b.high-b.low,(b.high-pc).abs(),(b.low-pc).abs()],axis=1).max(axis=1); return tr.rolling(n).mean()
def adx(b,n=14):
    up=b.high.diff(); dn=-b.low.diff(); plus=np.where((up>dn)&(up>0),up,0.); minus=np.where((dn>up)&(dn>0),dn,0.)
    tr=pd.concat([b.high-b.low,(b.high-b.close.shift()).abs(),(b.low-b.close.shift()).abs()],axis=1).max(axis=1)
    atrn=tr.rolling(n).sum(); pdi=100*pd.Series(plus).rolling(n).sum()/(atrn+1e-12); mdi=100*pd.Series(minus).rolling(n).sum()/(atrn+1e-12)
    dx=100*(pdi-mdi).abs()/(pdi+mdi+1e-12); return dx.rolling(n).mean()
def zigzag(b,depth=12,backstep=3):
    # completed-bar deterministic MetaQuotes-style extremum approximation.
    pts=[]
    for i in range(depth-1,len(b)):
        w=b.iloc[i-depth+1:i+1]; hi=float(b.high.iloc[i]); lo=float(b.low.iloc[i])
        if hi>=float(w.high.max())-1e-12: pts.append((i,hi,1))
        if lo<=float(w.low.min())+1e-12: pts.append((i,lo,-1))
    pts.sort(); out=[]
    for p in pts:
        if not out or p[2]!=out[-1][2]: out.append(p)
        elif (p[2]>0 and p[1]>=out[-1][1]) or (p[2]<0 and p[1]<=out[-1][1]): out[-1]=p
    return out

def metrics(trades,raw_ticks,start,end):
    p=np.array([x.pnl_usd for x in trades],float); n=len(p); gp=p[p>0].sum() if n else 0.; gl=-p[p<0].sum() if n else 0.; pf=gp/gl if gl>0 else (999. if gp>0 else 0.)
    eq=np.cumsum(p) if n else np.array([0.]); peak=np.maximum.accumulate(np.r_[0.,eq]); curve=np.r_[0.,eq]; dd=peak-curve; mdd=float(dd.max())
    wins=int((p>0).sum()); days=max((pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/86400,1e-9)
    return {'raw_ticks':raw_ticks,'period_start':start,'period_end':end,'days':days,'N':n,'WR_pct':100*wins/n if n else 0.,'PF':float(pf),'net_usd':float(p.sum()),'max_closed_DD_usd':mdd,'RF':float(p.sum()/mdd) if mdd>0 else (999. if p.sum()>0 else 0.),'avg_usd':float(p.mean()) if n else 0.,'avg_win_usd':float(p[p>0].mean()) if (p>0).any() else 0.,'avg_loss_usd':float(p[p<0].mean()) if (p<0).any() else 0.,'trades_per_day':n/days}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    ticks=load_ticks(a.catalog); m1=mkbars(ticks,1); m15=mkbars(ticks,15); h1=mkbars(ticks,60)
    h1['atr14']=atr(h1,14); h1['atr480']=atr(h1,480); h1['adx']=adx(h1,14)
    zz15=zigzag(m15); zz60=zigzag(h1)
    tns=ticks.ns.to_numpy(); bid=ticks.bid.to_numpy(); ask=ticks.ask.to_numpy(); times=ticks.time.to_numpy()
    m1_by_ns={int(pd.Timestamp(t).value):i for i,t in enumerate(m1.time)}; m15_by_ns={int(pd.Timestamp(t).value):i for i,t in enumerate(m15.time)}; h1_by_ns={int(pd.Timestamp(t).value):i for i,t in enumerate(h1.time)}
    pend=[]; pos=[]; trades=[]; entries_day={}; placed_day=set(); brk=False; brk_t=0; last_min=None; last_m15=None; last_h1=None; current_day=None
    vol=1.; regime=0; av=4.; day_hi=-1e99; day_lo=1e99; prev_hour=None
    def sltp():
        tp_mult=TREND_TP if regime==1 else (RANGE_TP if regime==-1 else 1.0); return min(max(av*SL_ATR,SL_MIN),SL_MAX), min(max(av*TP_ATR,TP_MIN),TP_MAX)*tp_mult
    def count_side(s): return sum(1 for p in pend if p.side==s)
    def add_pending(s,px,layer,now):
        nonlocal pend
        if count_side(s)>=MAX_PENDING_SIDE: return
        md=(FS+MIND_ADD)*UNIT*vol
        if any(q.side==s and abs(q.price-px)<md for q in pend): return
        if any(abs(x-px)<md for x in placed_day): return
        sl,tp=sltp(); pend.append(Pending(s,px,px-s*sl,px+s*tp,layer,str(now))); placed_day.add(px)
    def rebuild(layer,bar_idx,pts,cap,now):
        nonlocal pend
        hib=day_hi+(FS+MIND_ADD)*UNIT*vol; lob=day_lo-(FS+MIND_ADD)*UNIT*vol; nb=nsell=0
        loidx=max(0,bar_idx-600)
        for pi,price,s in reversed(pts):
            if pi<loidx: break
            if pi>bar_idx-2: continue
            if s>0 and price>hib and nb<cap:
                nb+=1; add_pending(1,price-FS*UNIT*vol,layer,now); hib=price
            elif s<0 and price<lob and nsell<cap:
                nsell+=1; add_pending(-1,price+FS*UNIT*vol,layer,now); lob=price
    for k in range(len(ticks)):
        now=pd.Timestamp(times[k]); b=float(bid[k]); aask=float(ask[k]); mid=(b+aask)/2; spread=aask-b
        day=now.date(); hour=now.hour; minute=now.floor('min')
        if day!=current_day:
            current_day=day; day_hi=mid; day_lo=mid; entries_day[day]=0; placed_day=set(); pend=[]
        day_hi=max(day_hi,mid); day_lo=min(day_lo,mid)
        # Spread breaker and session cancellation
        if hour in TRADE_HOURS:
            if spread/UNIT>SPREAD_BREAK_UNITS:
                if not brk: pend=[]
                brk=True; brk_t=int(tns[k])
            elif brk and int(tns[k])-brk_t>120_000_000_000 and spread/UNIT<SPREAD_BREAK_UNITS*.5: brk=False
        if prev_hour!=hour:
            prev_hour=hour
            if hour not in TRADE_HOURS: pend=[]
        # pending fills on raw bid/ask
        if hour in TRADE_HOURS and not brk and pend:
            remain=[]
            for q in pend:
                hit=(q.side>0 and aask>=q.price) or (q.side<0 and b<=q.price)
                if hit:
                    ep=aask if q.side>0 else b; pos.append(Pos(q.side,ep,q.sl,q.tp,q.layer,str(now))); entries_day[day]=entries_day.get(day,0)+1
                else: remain.append(q)
            pend=remain
        # positions tick execution: TP/SL then BE continuously
        keep=[]
        for p in pos:
            px=b if p.side>0 else aask; ex=p.side*(px-p.entry); p.mfe=max(p.mfe,ex); p.mae=min(p.mae,ex)
            hit_sl=(px<=p.sl) if p.side>0 else (px>=p.sl); hit_tp=(px>=p.tp) if p.side>0 else (px<=p.tp)
            if hit_sl or hit_tp:
                pnlp=p.side*(px-p.entry); trades.append(Trade(p.side,p.layer,p.entry_time,str(now),p.entry,px,pnlp,pnlp*LOT*CONTRACT,'TP' if hit_tp else 'SL',p.mae,p.mfe)); continue
            trig=BE_TRIG*vol; lock=BE_LOCK*vol
            if ex>=trig:
                ns=p.entry+p.side*lock
                if (p.side>0 and ns>p.sl) or (p.side<0 and ns<p.sl): p.sl=ns
            keep.append(p)
        pos=keep
        # bar events
        if minute!=last_min:
            last_min=minute
            # prior M1-bar gated trail
            mi=m1_by_ns.get(int((minute-pd.Timedelta(minutes=1)).value))
            hi_i=h1_by_ns.get(int(now.floor('h').value))
            if mi is not None and hi_i is not None and pos:
                mh=float(m1.high.iloc[mi]); ml=float(m1.low.iloc[mi]); hh=float(h1.high.iloc[hi_i]); hl=float(h1.low.iloc[hi_i]); gate=(mh>=hh-TR_BAND*vol) or (ml<=hl+TR_BAND*vol)
                if gate:
                    for p in pos:
                        if p.side>0 and mh>p.entry+TR_TRIG*vol: p.sl=max(p.sl,mh-TR_OFF*vol)
                        if p.side<0 and ml<p.entry-TR_TRIG*vol: p.sl=min(p.sl,ml+TR_OFF*vol)
            # adaptation from completed H1
            hj=np.searchsorted(h1.time.to_numpy(),np.datetime64(now.floor('h')),side='left')-1
            if hj>=0:
                avv=h1.atr14.iloc[hj]; al=h1.atr480.iloc[hj]; ax=h1.adx.iloc[hj]
                if np.isfinite(avv): av=float(avv)
                if np.isfinite(avv) and np.isfinite(al) and al>0: vol=float(np.clip(avv/al,VOL_MIN,VOL_MAX))
                regime=1 if np.isfinite(ax) and ax>=ADX_TREND else (-1 if np.isfinite(ax) and ax<=ADX_RANGE else 0)
            if hour in TRADE_HOURS and not brk:
                # H1 / M15 scans when new bar begins
                hcur=np.searchsorted(h1.time.to_numpy(),np.datetime64(now.floor('h')),side='left')
                if now.minute==0 and now.floor('h')!=last_h1:
                    last_h1=now.floor('h'); rebuild('A',hcur,zz60,7 if regime==1 else 5,now)
                if now.minute%15==0 and now.floor('15min')!=last_m15:
                    last_m15=now.floor('15min'); j15=np.searchsorted(m15.time.to_numpy(),np.datetime64(now.floor('15min')),side='left'); rebuild('B',j15,zz15,3,now)
                    # Layer C
                    if hour>=BOOST_HOUR and entries_day.get(day,0)<MIN_ENTRIES_DAY:
                        bo=BOOST_OFF*vol*(RANGE_BOOST if regime==-1 else 1.0); add_pending(1,day_hi+bo,'C',now); add_pending(-1,day_lo-bo,'C',now)
    # force close any residual at final quote
    now=pd.Timestamp(times[-1]); b=float(bid[-1]); aask=float(ask[-1])
    for p in pos:
        px=b if p.side>0 else aask; pnlp=p.side*(px-p.entry); trades.append(Trade(p.side,p.layer,p.entry_time,str(now),p.entry,px,pnlp,pnlp*LOT*CONTRACT,'EOD',p.mae,p.mfe))
    met=metrics(trades,len(ticks),str(ticks.time.iloc[0]),str(ticks.time.iloc[-1])); met.update({'strategy':'GoldeBrave_v4.20_defaults','lot':LOT,'contract_size_assumed':CONTRACT,'entry_layers':dict(pd.Series([t.layer for t in trades]).value_counts()) if trades else {},'note':'Raw Bid/Ask execution; signal bars derived from raw ticks; standard ZigZag reproduced deterministically from completed bars.'})
    pd.DataFrame([asdict(x) for x in trades]).to_csv(out/'trades.csv',index=False); (out/'summary.json').write_text(json.dumps(met,indent=2),encoding='utf-8'); print(json.dumps(met,indent=2))
if __name__=='__main__': main()
