from __future__ import annotations
# Parallel BOT core screen. Each strategy keeps its own signal logic; common execution wrapper isolates edge quality.
import argparse, datetime as dt, itertools, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.bigplayer_synergy_supervisor_21d as core

OUT=Path('results/bot_parallel_portfolio_21d'); INITIAL=1000.; QTY=1.0

def rsi(c,n):
    d=c.diff(); g=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); l=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); rs=g/(l+1e-9); return 100-100/(1+rs)

def atr(b,n=14):
    p=b.close.shift(1); tr=pd.concat([(b.high-b.low),(b.high-p).abs(),(b.low-p).abs()],axis=1).max(axis=1); return tr.rolling(n,min_periods=n).mean()

def stoch(b,k=8,d=3,slow=3):
    ll=b.low.rolling(k).min(); hh=b.high.rolling(k).max(); raw=100*(b.close-ll)/(hh-ll).replace(0,np.nan); main=raw.rolling(slow).mean(); sig=main.rolling(d).mean(); return main,sig

def fast_signals(b):
    x=b.copy(); x['e200']=x.close.ewm(span=200,adjust=False).mean(); x['e50']=x.close.ewm(span=50,adjust=False).mean(); x['e20']=x.close.ewm(span=20,adjust=False).mean(); x['rsi']=rsi(x.close,7)
    prev=x.rsi.shift(1); buy=(x.close>x.e200)&(x.e20>x.e50)&(x.low<=x.e20)&(x.close>=x.e50)&(prev<=35)&(x.rsi>40)&(x.close>x.open); sell=(x.close<x.e200)&(x.e20<x.e50)&(x.high>=x.e20)&(x.close<=x.e50)&(prev>=65)&(x.rsi<60)&(x.close<x.open)
    s=pd.Series(0,index=x.index,dtype='int8'); s[buy]=1; s[sell]=-1; return s

def roy_signals(m1,m15):
    x=m1.copy(); x['ef']=x.close.ewm(span=20,adjust=False).mean(); x['es']=x.close.ewm(span=50,adjust=False).mean(); x['rsi']=rsi(x.close,14); x['st_m'],x['st_s']=stoch(x,8,3,3)
    h=m15[['datetime','close']].copy(); h['hef']=h.close.ewm(span=20,adjust=False).mean(); h['hes']=h.close.ewm(span=50,adjust=False).mean(); h=h[['datetime','hef','hes']]
    x=pd.merge_asof(x.sort_values('datetime'),h.sort_values('datetime'),on='datetime',direction='backward')
    pm=x.st_m.shift(1); ps=x.st_s.shift(1); hour=x.datetime.dt.hour; dow=x.datetime.dt.dayofweek
    buy=(x.hef>x.hes)&(x.ef>x.es)&(x.rsi>50)&(x.st_m>x.st_s)&(pm<=ps)&(x.st_m<80)&(x.volume>=50)&(hour>=8)&(hour<17)&(dow!=4)&(x.spread<=1.20)
    sell=(x.hef<x.hes)&(x.ef<x.es)&(x.rsi<50)&(x.st_m<x.st_s)&(pm>=ps)&(x.st_m>20)&(x.volume>=50)&(hour>=8)&(hour<17)&(dow!=4)&(x.spread<=1.20)
    s=pd.Series(0,index=x.index,dtype='int8'); s[buy]=1; s[sell]=-1; return x,s

def midas_liquidity_signals(b,lookback=50,strength=3):
    x=b.copy(); s=pd.Series(0,index=x.index,dtype='int8')
    highs=[]; lows=[]
    for i in range(strength,len(x)-strength):
        h=x.high.iat[i]; l=x.low.iat[i]
        if h>x.high.iloc[i-strength:i].max() and h>x.high.iloc[i+1:i+strength+1].max(): highs.append((i,h))
        if l<x.low.iloc[i-strength:i].min() and l<x.low.iloc[i+1:i+strength+1].min(): lows.append((i,l))
        if i<lookback: continue
        recent_h=[p for j,p in highs if j<i and j>=i-lookback][-5:]; recent_l=[p for j,p in lows if j<i and j>=i-lookback][-5:]
        if any(x.low.iat[i]<p and x.close.iat[i]>p for p in recent_l) and x.close.iat[i]>x.open.iat[i]: s.iat[i]=1
        elif any(x.high.iat[i]>p and x.close.iat[i]<p for p in recent_h) and x.close.iat[i]<x.open.iat[i]: s.iat[i]=-1
    return s

def bigplayer_tr(m1,m5):
    a=core.edges(m1); b=core.edges(m5); out=[]
    for tf,x,edge,sec in [('M1',a,'IMBALANCE',60),('M5',b,'ABSORPTION',300)]:
        for i in np.flatnonzero(x[edge].to_numpy()!=0):
            d=int(x[edge].iat[i]); ok=(int(x.trend_dir.iat[i])==d) and pd.notna(x.atr_med120.iat[i]) and x.atr14.iat[i]>=x.atr_med120.iat[i]
            if ok: out.append((x.datetime.iat[i]+pd.Timedelta(seconds=sec),d,tf))
    return out

def exec_first_passage(ticks,et,d,target_mult=2.0,max_min=30):
    a=ticks.datetime.searchsorted(et,'left'); z=ticks.datetime.searchsorted(et+pd.Timedelta(minutes=max_min),'left')
    if a>=len(ticks) or z>=len(ticks) or z<=a:return None
    entry=float(ticks.ask.iat[a] if d>0 else ticks.bid.iat[a]); spr=float(ticks.ask.iat[a]-ticks.bid.iat[a]); target=target_mult*spr
    ex=float(ticks.bid.iat[z] if d>0 else ticks.ask.iat[z]); xt=ticks.datetime.iat[z]; hit=False
    for j in range(a+1,z+1):
        px=float(ticks.bid.iat[j] if d>0 else ticks.ask.iat[j])
        if d*(px-entry)>=target: ex=px; xt=ticks.datetime.iat[j]; hit=True; break
    pnl=d*(ex-entry)*QTY; return entry,ex,xt,pnl,hit,spr

def metrics(t):
    if t.empty:return dict(N=0,N_per_day=0.,WR=0.,PF=0.,RF=0.,net_profit=0.,return_pct=0.,daily_return_pct=0.,max_dd_pct=0.,max_dd_usd=0.,final_balance=INITIAL,hit_rate=0.)
    t=t.sort_values('entry_time'); p=t.pnl; gp=p[p>0].sum(); gl=-p[p<0].sum(); net=p.sum(); eq=INITIAL+p.cumsum(); peak=np.maximum.accumulate(np.r_[INITIAL,eq.values])[1:]; dd=peak-eq.values; mdd=float(dd.max())
    return dict(N=len(t),N_per_day=len(t)/21.,WR=(p>0).mean()*100.,PF=gp/gl if gl>0 else math.inf,RF=net/mdd if mdd>0 else math.inf,net_profit=net,return_pct=net/INITIAL*100.,daily_return_pct=net/INITIAL*100./21.,max_dd_pct=float(np.max(np.where(peak>0,dd/peak*100.,0))),max_dd_usd=mdd,final_balance=INITIAL+net,hit_rate=t.hit.mean()*100.)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',default='2026-07-27'); ap.add_argument('--workers',type=int,default=48); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    days=core.business_days(dt.date.fromisoformat(a.start),21); ticks=core.load_ticks(days,a.workers); m1=core.bars(ticks,60); m5=core.bars(ticks,300); m15=core.bars(ticks,900)
    bots={}
    fs=fast_signals(m1); bots['FAST_SOURCE']=[(m1.datetime.iat[i]+pd.Timedelta(minutes=1),int(fs.iat[i])) for i in np.flatnonzero(fs.to_numpy()!=0)]
    rx,rs=roy_signals(m1,m15); bots['ROY_SOURCE']=[(rx.datetime.iat[i]+pd.Timedelta(minutes=1),int(rs.iat[i])) for i in np.flatnonzero(rs.to_numpy()!=0)]
    ms=midas_liquidity_signals(m1); bots['MIDAS_LIQ_SOURCE']=[(m1.datetime.iat[i]+pd.Timedelta(minutes=1),int(ms.iat[i])) for i in np.flatnonzero(ms.to_numpy()!=0)]
    bots['BIGPLAYER_TR']=[(et,d) for et,d,_ in bigplayer_tr(m1,m5)]
    frames={}; rows=[]
    for name,sigs in bots.items():
        tr=[]
        for et,d in sigs:
            q=exec_first_passage(ticks,pd.Timestamp(et),d,2.0,30)
            if q:
                en,ex,xt,pnl,hit,spr=q; tr.append(dict(bot=name,direction=d,entry_time=et,exit_time=xt,entry=en,exit=ex,pnl=pnl,hit=hit,entry_spread=spr))
        t=pd.DataFrame(tr); frames[name]=t; rows.append({'config':name,'kind':'SINGLE',**metrics(t)})
    names=list(frames)
    combos=[]
    for r in [2,3,4]: combos += list(itertools.combinations(names,r))
    for c in combos:
        parts=[frames[n] for n in c if not frames[n].empty]; t=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(); rows.append({'config':'+'.join(c),'kind':f'{len(c)}BOT',**metrics(t)})
    res=pd.DataFrame(rows).sort_values(['return_pct','PF'],ascending=False); res.to_csv(OUT/'summary_21d.csv',index=False)
    for n,t in frames.items(): t.to_csv(OUT/f'trades_{n}.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps({'period':'21 business days','execution':'raw bid/ask; common Economic BE first-passage target=2x entry spread; max horizon=30m','bar_policy':'direct raw QuoteTick aggregation; no OHLC resample','strategies':{'FAST_SOURCE':'EMA200/50/20 + RSI7 source rules','ROY_SOURCE':'M15/M1 EMA20/50 + RSI14 + Stoch8,3,3 + volume/session/friday/spread source rules','MIDAS_LIQ_SOURCE':'lookback50 strength3 liquidity sweep source rules','BIGPLAYER_TR':'current frozen BigPlayer positive edges + external Trend/Regime winner'},'portfolio':'equal 1x independent BOT aggregation; no netting'},indent=2)); print(res.to_string(index=False))
if __name__=='__main__':main()
