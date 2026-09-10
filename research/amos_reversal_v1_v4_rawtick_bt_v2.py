#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,os,random
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np,pandas as pd
TF_MIN={'M1':1,'M5':5,'M15':15,'H1':60}
@dataclass
class Trade:
    version:str; timeframe:str; mode:str; side:int; entry_time:str; exit_time:str; entry:float; stop:float; target:float; exit_price:float; r:float; result:str; leader_tf:str=''; structure_score:float=0.; context_score:float=0.; probability:float=0.
def cli():
    p=argparse.ArgumentParser(); p.add_argument('--version',required=True,choices=['V1','V2','V3','V4']); p.add_argument('--timeframe',required=True,choices=list(TF_MIN)); p.add_argument('--mode',required=True,choices=['BASE','AGGRESSIVE','STANDARD','CONSERVATIVE']); p.add_argument('--symbol',default='XAUUSD'); p.add_argument('--dataset',default='dukascopy_raw'); p.add_argument('--seed',type=int,default=42); p.add_argument('--rr',type=float,default=2.0); p.add_argument('--out',required=True); return p.parse_args()
def resolve_source(symbol,dataset):
    env=os.environ.get('AMOS_RAW_TICK_PATH','').strip()
    if env:
        p=Path(env)
        if p.exists(): return p
        raise FileNotFoundError(f'AMOS_RAW_TICK_PATH not found: {p}')
    if dataset!='dukascopy_raw': raise ValueError(f'unsupported dataset selector: {dataset}')
    p=Path('raw')/f'{symbol}_dukascopy_ticks.parquet'
    if not p.exists(): raise FileNotFoundError(f'raw tick source missing: {p}')
    return p
def load_ticks(path):
    d=pd.read_parquet(path) if path.suffix.lower()=='.parquet' else pd.read_csv(path); low={c.lower():c for c in d.columns}
    tc=next((low[k] for k in ['datetime','timestamp','time','ts'] if k in low),None); bc=next((low[k] for k in ['bid','bid_price'] if k in low),None); ac=next((low[k] for k in ['ask','ask_price'] if k in low),None)
    if not tc or not bc or not ac: raise ValueError('raw source requires datetime,bid,ask')
    out=pd.DataFrame({'time':pd.to_datetime(d[tc],utc=True,errors='coerce'),'bid':pd.to_numeric(d[bc],errors='coerce'),'ask':pd.to_numeric(d[ac],errors='coerce')})
    vc=next((low[k] for k in ['volume','tick_volume','size'] if k in low),None); out['vol']=pd.to_numeric(d[vc],errors='coerce').fillna(1.0) if vc else 1.0
    out=out.dropna().query('ask>=bid and bid>0').sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
    if out.empty: raise ValueError('empty raw tick source')
    return out
def bars(t,tf):
    x=t.set_index('time'); mid=(x.bid+x.ask)/2; rule=f'{TF_MIN[tf]}min'; o=mid.resample(rule,label='right',closed='right').ohlc(); o['volume']=x.vol.resample(rule,label='right',closed='right').sum(); return o.dropna().reset_index()
def atr(b,n=14):
    pc=b.close.shift(1); return pd.concat([b.high-b.low,(b.high-pc).abs(),(b.low-pc).abs()],axis=1).max(axis=1).rolling(n).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean(); rs=up/(dn+1e-12); return 100-100/(1+rs)
def feat(b):
    z=b.copy(); z['atr']=atr(z); z['rsi']=rsi(z.close); z['body']=(z.close-z.open).abs(); z['disp']=z.body/(z.atr+1e-12); z['swing_hi']=z.high.shift(1).rolling(5).max(); z['swing_lo']=z.low.shift(1).rolling(5).min(); z['bull_fvg']=z.low>z.high.shift(2); z['bear_fvg']=z.high<z.low.shift(2); z['vol_med']=z.volume.rolling(20).median(); tp=(z.high+z.low+z.close)/3; z['vwap']=(tp*z.volume).cumsum()/(z.volume.cumsum()+1e-12); z['regime']=(100*(z.close-z.close.shift(14)).abs()/(z.atr*14+1e-12)).clip(0,100); return z
def sigm(x): return 1/(1+math.exp(-max(-40,min(40,x))))
def logit(p):
    p=min(.999999,max(.000001,p)); return math.log(p/(1-p))
def signal(z,i,v,mode,diag=None):
    if i<30:return None
    r=z.iloc[i];p=z.iloc[i-1]
    if not np.isfinite(r.atr) or r.atr<=0:return None
    bs=r.low<r.swing_lo and r.close>r.swing_lo; ss=r.high>r.swing_hi and r.close<r.swing_hi; side=1 if bs else -1 if ss else 0
    if not side:return None
    if diag is not None: diag['sweep']+=1
    cisd=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
    mss=(r.close>z.high.shift(1).rolling(3).max().iloc[i]) if side==1 else (r.close<z.low.shift(1).rolling(3).min().iloc[i]); disp=r.disp>=0.8; fvg=bool(r.bull_fvg if side==1 else r.bear_fvg); opp=bool(z.bear_fvg.iloc[max(0,i-8):i].any()) if side==1 else bool(z.bull_fvg.iloc[max(0,i-8):i].any()); ifvg=opp and disp; bpr=fvg and opp
    st=1+1.0*cisd+1.2*mss+1.0*disp+0.7*fvg+0.5*ifvg+0.5*bpr
    if cisd and diag is not None: diag['cisd']+=1
    if mss and diag is not None: diag['mss']+=1
    if disp and diag is not None: diag['disp']+=1
    if v=='V1':
        ok=cisd and mss and disp and (fvg or ifvg or bpr)
        if ok and diag is not None: diag['accepted']+=1
        return (side,st,0.,.5,'') if ok else None
    th={'AGGRESSIVE':2.7,'STANDARD':3.5,'CONSERVATIVE':4.2}.get(mode,3.5)
    if v=='V2':
        ok=cisd and st>=th
        if ok and diag is not None: diag['accepted']+=1
        return (side,st,0.,sigm(st-3.5),'') if ok else None
    vel=min(1.5,abs(r.close-p.close)/(r.atr+1e-12)); adaptive=.55*st+.30*vel+.15
    base_pr=sigm(-2.1+.72*st+.35*vel)
    if v=='V3':
        gate={'AGGRESSIVE':.50,'STANDARD':.54,'CONSERVATIVE':.58}[mode]; ok=base_pr>=gate
        if ok and diag is not None: diag['accepted']+=1
        return (side,st,adaptive,base_pr,'SELF') if ok else None
    # V4: preserve V3 structural candidates. Context is a soft rank/boost, never a hard veto.
    mom=max(-1.5,min(1.5,((r.rsi-50)/25)*side)) if np.isfinite(r.rsi) else 0.0
    reg=max(-1,min(1,(r.regime-20)/20)) if np.isfinite(r.regime) else 0.0
    loc=max(-2,min(2,((r.close-r.vwap)/(r.atr+1e-12))*side)) if np.isfinite(r.vwap) else 0.0
    part=min(2,r.volume/(r.vol_med+1e-12))-1 if np.isfinite(r.vol_med) and r.vol_med>0 else 0.0
    ctx=.30*mom+.25*reg+.25*loc+.20*part
    v3_gate={'AGGRESSIVE':.50,'STANDARD':.54,'CONSERVATIVE':.58}[mode]
    if base_pr < v3_gate: return None
    # soft update: context can move confidence modestly but cannot delete a structurally valid V3 signal
    pr=sigm(logit(base_pr)+0.35*ctx)
    if diag is not None: diag['v3_candidate']+=1; diag['accepted']+=1
    return (side,st,ctx,pr,'ADAPTIVE')
def simulate(t,b,v,tf,mode,rr):
    z=feat(b); trades=[]; tv=t.time.astype('int64').to_numpy(); last_exit_ns=-1
    diag={'sweep':0,'cisd':0,'mss':0,'disp':0,'v3_candidate':0,'accepted':0}
    for i in range(30,len(z)-1):
        s=signal(z,i,v,mode,diag)
        if s is None:continue
        side,st,ctx,pr,leader=s; sig_t=z.time.iloc[i]; sig_ns=sig_t.value
        if sig_ns<=last_exit_ns:continue
        j=int(np.searchsorted(tv,sig_ns,side='right'))
        if j>=len(t):break
        q=t.iloc[j]; entry=float(q.ask if side==1 else q.bid); av=float(z.atr.iloc[i]); swing=float(z.swing_lo.iloc[i] if side==1 else z.swing_hi.iloc[i]); stop=min(swing,entry-.35*av) if side==1 else max(swing,entry+.35*av); risk=abs(entry-stop)
        if risk<=0 or not np.isfinite(risk):continue
        target=entry+side*rr*risk; horizon_ns=(sig_t+pd.Timedelta(minutes=TF_MIN[tf]*5)).value; end=int(np.searchsorted(tv,horizon_ns,side='right'))
        if end<=j+1:continue
        exit_px=None; exit_t=None; rval=None; result='TIME'
        for k in range(j+1,end):
            x=t.iloc[k]; px=float(x.bid if side==1 else x.ask)
            if (side==1 and px<=stop) or (side==-1 and px>=stop): exit_px=px; exit_t=x.time; rval=side*(px-entry)/risk; result='LOSS'; break
            if (side==1 and px>=target) or (side==-1 and px<=target): exit_px=px; exit_t=x.time; rval=side*(px-entry)/risk; result='WIN'; break
        if exit_px is None:
            x=t.iloc[end-1]; exit_px=float(x.bid if side==1 else x.ask); exit_t=x.time; rval=side*(exit_px-entry)/risk
        trades.append(Trade(v,tf,mode,side,str(sig_t),str(exit_t),entry,stop,target,exit_px,float(rval),result,leader,float(st),float(ctx),float(pr))); last_exit_ns=exit_t.value
    return trades,diag
def metrics(trades):
    rs=np.array([x.r for x in trades],float); n=len(rs); wins=int((rs>0).sum()); losses=int((rs<=0).sum()); gp=float(rs[rs>0].sum()) if n else 0.; gl=float(-rs[rs<0].sum()) if n else 0.; pf=gp/gl if gl>0 else (999. if gp>0 else 0.); curve=np.concatenate(([0.],np.cumsum(rs))) if n else np.array([0.]); peak=np.maximum.accumulate(curve); md=float((peak-curve).max()); net=float(rs.sum()) if n else 0.; rf=net/md if md>0 else (999. if net>0 else 0.); return {'N':n,'wins':wins,'losses':losses,'WR_pct':100*wins/n if n else 0.,'avg_R':float(rs.mean()) if n else 0.,'PF':pf,'EV_R_per_trade':float(rs.mean()) if n else 0.,'net_R':net,'max_DD_R':md,'RF':rf}
def main():
    a=cli(); random.seed(a.seed); np.random.seed(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); src=resolve_source(a.symbol,a.dataset); t=load_ticks(src); b=bars(t,a.timeframe); tr,diag=simulate(t,b,a.version,a.timeframe,a.mode,a.rr); m=metrics(tr); pd.DataFrame([asdict(x) for x in tr]).to_csv(out/'trades.csv',index=False); m.update({'version':a.version,'timeframe':a.timeframe,'mode':a.mode,'symbol':a.symbol,'dataset':a.dataset,'rr_target':a.rr,'period_start':str(t.time.iloc[0]),'period_end':str(t.time.iloc[-1]),'raw_tick_source':str(src),'verification_level':'RAW_DUKASCOPY_BIDASK','signal_diagnostics':diag}); (out/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8'); (out/'manifest.json').write_text(json.dumps({'args':vars(a),'rows':len(t),'bars':len(b),'raw_tick_source':str(src),'ohlc_execution_substitution':False},indent=2),encoding='utf-8'); print(json.dumps(m,indent=2))
if __name__=='__main__': main()
