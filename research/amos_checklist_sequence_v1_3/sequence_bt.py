#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from itertools import product

def hhmm(s):
    h,m=map(int,s.split(':')); return h*60+m

ASIA=(hhmm('00:00'),hhmm('06:00'))
LONDON=(hhmm('07:00'),hhmm('10:00'))
NY=(hhmm('13:30'),hhmm('16:00'))

def inwin(m,w): return w[0] <= m < w[1]

def atr(df,n=14):
    pc=df.close.shift(1)
    tr=pd.concat([(df.high-df.low).abs(),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def make_m15(df):
    x=df.set_index('datetime')
    h=pd.DataFrame({
        'open':x.open.resample('15min').first(),
        'high':x.high.resample('15min').max(),
        'low':x.low.resample('15min').min(),
        'close':x.close.resample('15min').last(),
        'volume':x.volume.resample('15min').sum(),
    }).dropna().reset_index()
    return h

def build_ref_arrays(df):
    ref_hi=np.full(len(df),np.nan); ref_lo=np.full(len(df),np.nan)
    daykey=df.datetime.dt.normalize()
    for _,g in df.groupby(daykey,sort=False):
        mins=g.datetime.dt.hour*60+g.datetime.dt.minute
        asia=g[(mins>=ASIA[0])&(mins<ASIA[1])]
        london=g[(mins>=LONDON[0])&(mins<LONDON[1])]
        lon_idx=g.index[(mins>=LONDON[0])&(mins<LONDON[1])]
        ny_idx=g.index[(mins>=NY[0])&(mins<NY[1])]
        if len(asia):
            ref_hi[lon_idx]=float(asia.high.max()); ref_lo[lon_idx]=float(asia.low.min())
        if len(london):
            ref_hi[ny_idx]=float(london.high.max()); ref_lo[ny_idx]=float(london.low.min())
    return ref_hi,ref_lo

def htf_context(h15,t,di,px,look=32):
    j=h15.datetime.searchsorted(t,side='right')-2
    if j<look:return False
    w=h15.iloc[j-look+1:j+1]
    hi=float(w.high.max()); lo=float(w.low.min()); eq=(hi+lo)/2
    return px>=eq if di<0 else px<=eq

def macro_context(t):
    m=t.hour*60+t.minute
    # narrower windows inside London/NY, treated only as optional context
    return inwin(m,(hhmm('07:30'),hhmm('09:30'))) or inwin(m,(hhmm('14:00'),hhmm('15:30')))

def volume_context(v,vma,mult=1.05):
    return bool(np.isfinite(vma) and vma>0 and v>=vma*mult)

def simulate_trade(df, event, rr):
    di=event['dir']; e=event['entry']; sl=event['stop']; R=abs(e-sl)
    tp=e+di*R*rr
    for j in range(event['entry_idx']+1,len(df)):
        lo=float(df.low.iat[j]); hi=float(df.high.iat[j])
        # conservative same-bar ambiguity: stop first
        if di>0:
            if lo<=sl:return -1.0,j
            if hi>=tp:return float(rr),j
        else:
            if hi>=sl:return -1.0,j
            if lo<=tp:return float(rr),j
    return 0.0,len(df)-1

def collect_events(df,h15,ref_hi,ref_lo,lookback,mss_body_atr,fib_lo,fib_hi,entry_mode):
    stages={'sweeps':0,'mss':0,'eq_touches':0,'eq_rebalances':0,'candidates':0}
    ev=[]
    state='IDLE'; sweep_i=None; di=0; sweep_ext=0.; protected=0.; age=0
    mss_i=None; disp_ext=0.; refH=refL=0.
    for i in range(max(30,lookback+5),len(df)):
        a=float(df.atr.iat[i])
        if not np.isfinite(a) or a<=0: continue
        b=df.iloc[i]; t=b.datetime
        H=ref_hi[i]; L=ref_lo[i]
        if state=='IDLE':
            if not (np.isfinite(H) and np.isfinite(L)): continue
            min_s=.02*a; max_s=1.2*a
            sh=(b.high>H+min_s and b.close<H and b.high-H<=max_s)
            sl=(b.low<L-min_s and b.close>L and L-b.low<=max_s)
            if not (sh or sl): continue
            stages['sweeps']+=1
            state='WAIT_MSS'; age=0; sweep_i=i; refH,refL=H,L
            if sh:
                di=-1; sweep_ext=float(b.high)
                protected=float(df.low.iloc[max(0,i-lookback):i].min())
            else:
                di=1; sweep_ext=float(b.low)
                protected=float(df.high.iloc[max(0,i-lookback):i].max())
            continue
        age+=1
        if state=='WAIT_MSS':
            if age>25:
                state='IDLE'; continue
            if di<0: sweep_ext=max(sweep_ext,float(b.high))
            else: sweep_ext=min(sweep_ext,float(b.low))
            body=abs(float(b.close-b.open))
            if di<0:
                ok=(b.close<protected and body>=mss_body_atr*a)
            else:
                ok=(b.close>protected and body>=mss_body_atr*a)
            if not ok: continue
            stages['mss']+=1
            mss_i=i; age=0; state='WAIT_EQ'
            disp_ext=float(b.low if di<0 else b.high)
            continue
        if state=='WAIT_EQ':
            if age>30:
                state='IDLE'; continue
            # extend impulse until retracement actually starts
            if di<0:
                disp_ext=min(disp_ext,float(b.low))
                rng=sweep_ext-disp_ext
                if rng<=0: continue
                zlo=disp_ext+fib_lo*rng; zhi=disp_ext+fib_hi*rng
                touch=(b.high>=zlo and b.low<=zhi and i>mss_i)
                proximal=zlo
                reject=(touch and b.close<proximal and b.close<b.open)
                stop=sweep_ext+.08*a
                opp=refL
            else:
                disp_ext=max(disp_ext,float(b.high))
                rng=disp_ext-sweep_ext
                if rng<=0: continue
                zlo=disp_ext-fib_hi*rng; zhi=disp_ext-fib_lo*rng
                touch=(b.low<=zhi and b.high>=zlo and i>mss_i)
                proximal=zhi
                reject=(touch and b.close>proximal and b.close>b.open)
                stop=sweep_ext-.08*a
                opp=refH
            if not touch: continue
            stages['eq_touches']+=1
            if entry_mode=='reject':
                if not reject: continue
                entry=float(b.close)
            else:
                entry=float(proximal)
            stages['eq_rebalances']+=1
            R=abs(entry-stop)
            if R<=0:
                state='IDLE';continue
            reward=(opp-entry)*di
            runway_rr=reward/R
            ctx_htf=htf_context(h15,t,di,entry)
            ctx_macro=macro_context(t)
            vma=float(df.vma.iat[i]) if np.isfinite(df.vma.iat[i]) else np.nan
            ctx_vol=volume_context(float(b.volume),vma)
            ev.append({
                'entry_idx':i,'time':str(t),'dir':di,'entry':entry,'stop':stop,'R':R,
                'runway_rr':float(runway_rr),'ctx_htf':int(ctx_htf),'ctx_macro':int(ctx_macro),
                'ctx_volume':int(ctx_vol),'ctx_count':int(ctx_htf)+int(ctx_macro)+int(ctx_vol),
                'lookback':lookback,'mss_body_atr':mss_body_atr,'fib_lo':fib_lo,'fib_hi':fib_hi,
                'entry_mode':entry_mode,
            })
            stages['candidates']+=1
            state='IDLE'
    return ev,stages

def cache_outcomes(df,events):
    for e in events:
        for rr in (2.0,3.0,4.0):
            r,ex=simulate_trade(df,e,rr)
            e[f'out_{int(rr)}R']=r
            e[f'exit_{int(rr)}R']=ex

def evaluate(df,events,rr,ctx_min,require_runway,risk=.35):
    eq=100.; peak=100.; maxdd=0.; w=l=be=0; gw=gl=0.; sumr=0.; end_idx=-1
    accepted=0
    rk=int(rr)
    for e in events:
        if e['entry_idx']<=end_idx: continue
        if e['ctx_count']<ctx_min: continue
        if require_runway and e['runway_rr']<rr: continue
        accepted+=1
        r,ex=e[f'out_{rk}R'],e[f'exit_{rk}R']; end_idx=ex
        if r>0:w+=1;gw+=r
        elif r<0:l+=1;gl+=1
        else:be+=1
        sumr+=r
        eq*=max(.0001,1+(risk/100)*r)
        peak=max(peak,eq); maxdd=max(maxdd,100*(peak-eq)/peak)
    n=w+l+be
    wr=100*w/n if n else 0.
    pf=gw/gl if gl>0 else (np.inf if gw>0 else 0.)
    return dict(N=n,wins=w,losses=l,BE=be,WR_pct=wr,PF_R=pf,sum_R=sumr,
                Return_pct_risk_compound=eq-100,MaxDD_pct=maxdd,accepted=accepted)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.data); df.columns=[c.lower() for c in df.columns]; df['datetime']=pd.to_datetime(df.datetime)
    for c in ['open','high','low','close','volume']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna().sort_values('datetime').reset_index(drop=True)
    df['atr']=atr(df);df['vma']=df.volume.shift(1).rolling(20).mean()
    h15=make_m15(df)
    ref_hi,ref_lo=build_ref_arrays(df)
    core_grid=list(product([4,8],[.35,.55],[(.50,.62),(.62,.79)],['touch','reject']))
    all_events=[]; stage_rows=[]; result_rows=[]
    for lb,mb,fibs,emode in core_grid:
        ev,st=collect_events(df,h15,ref_hi,ref_lo,lb,mb,fibs[0],fibs[1],emode)
        cache_outcomes(df,ev)
        all_events.extend(ev)
        stage_rows.append(dict(lookback=lb,mss_body_atr=mb,fib_lo=fibs[0],fib_hi=fibs[1],entry_mode=emode,**st))
        for rr,ctx,runway in product([2.0,3.0,4.0],[0,1,2],[False,True]):
            z=evaluate(df,ev,rr,ctx,runway)
            row=dict(model='MSS_EQ_REBALANCE',lookback=lb,mss_body_atr=mb,fib_lo=fibs[0],fib_hi=fibs[1],
                     entry_mode=emode,target_R=rr,context_min=ctx,clear_target_required=int(runway),**z)
            # Rank for robust positive expectancy; do not reward tiny N
            row['objective']=(min(row['PF_R'],8)*math.sqrt(max(row['N'],1))/(1+row['MaxDD_pct'])) if row['N']>=20 else -1
            result_rows.append(row)
    res=pd.DataFrame(result_rows)
    res.replace([np.inf,-np.inf],np.nan,inplace=True)
    res.to_csv(out/'sequence_matrix.csv',index=False)
    pd.DataFrame(stage_rows).to_csv(out/'stage_counts.csv',index=False)
    pd.DataFrame(all_events).to_csv(out/'candidate_events.csv',index=False)
    top=res[(res.N>=20)&res.PF_R.notna()].sort_values(['objective','PF_R','N'],ascending=False)
    top.head(50).to_csv(out/'top50.csv',index=False)
    # Explicit parity views: core-only and clear-target variants
    views=res[(res.context_min==0)&(res.target_R==3.0)].sort_values(['PF_R','N'],ascending=False)
    views.to_csv(out/'core_3R_views.csv',index=False)
    meta={
      'verification_level':'M1_OHLC_ORDERED_STATE_MACHINE_SCREEN',
      'model':'MSS + EQ Rebalance',
      'ordered_core':['Liquidity Sweep','Displacement-driven MSS','Equilibrium/OTE calculated from sweep-to-displacement leg','Pullback into EQ/OTE','EQ Rebalance execution','Entry'],
      'optional_context':['HTF PDA Delivery','Macro Window','Volume Influx'],
      'clear_target_tested_as':'optional hard gate using opposing session liquidity >= chosen fixed-R target',
      'rows':len(df),'start':str(df.datetime.iloc[0]),'end':str(df.datetime.iloc[-1]),
      'limitations':['mid-quote OHLCV','no spread/slippage/swap','same-bar ambiguity resolved stop-first','entry/stop/target are reconstruction hypotheses','Raw Bid/Ask Tick required before promotion']
    }
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('ORDERED CORE:', ' -> '.join(meta['ordered_core']))
    print('DATA',meta['start'],'->',meta['end'],'rows',meta['rows'])
    print('TOP20')
    cols=['lookback','mss_body_atr','fib_lo','fib_hi','entry_mode','target_R','context_min','clear_target_required','N','WR_pct','PF_R','Return_pct_risk_compound','MaxDD_pct','objective']
    print(top[cols].head(20).to_string(index=False))
    print('STAGES')
    print(pd.DataFrame(stage_rows).to_string(index=False))

if __name__=='__main__': main()
