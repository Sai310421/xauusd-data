import json, math
from pathlib import Path
import pandas as pd
import numpy as np

OUT=Path('bt_results/c0_multisymbol_m1m5_v1'); OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-02-25 00:00:00'); END=pd.Timestamp('2026-05-26 23:59:59')
PROFILE={'B':(.62,.78),'C':(.55,.78),'D':(.78,.90)}
PIVOT_LR={'M1':2,'M5':2}; WAIT={'M1':12,'M5':12}; MAX_HOLD={'M1':180,'M5':120}
INITIAL=1000.0; RISK_FRAC=0.0025

def normalize(df):
    df=df.copy(); df.columns=[str(c).lower() for c in df.columns]
    if 'datetime' not in df.columns:
        for c in ['time','timestamp','date']:
            if c in df.columns: df=df.rename(columns={c:'datetime'}); break
    df['datetime']=pd.to_datetime(df['datetime']).dt.tz_localize(None)
    need=['datetime','open','high','low','close']
    return df.loc[(df.datetime>=START)&(df.datetime<=END),need].dropna().sort_values('datetime').reset_index(drop=True)

def atr14(df):
    pc=df.close.shift(1)
    tr=pd.concat([(df.high-df.low),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()

def confirmed_pivots(df,lr):
    H=df.high.to_numpy(); L=df.low.to_numpy(); events=[]
    for i in range(2*lr,len(df)):
        j=i-lr
        if H[j]>=np.max(H[j-lr:j+lr+1]): events.append((i,j,'H',float(H[j])))
        if L[j]<=np.min(L[j-lr:j+lr+1]): events.append((i,j,'L',float(L[j])))
    events.sort(key=lambda z:(z[0],z[1],z[2])); alt=[]
    for e in events:
        if not alt or alt[-1][2]!=e[2]: alt.append(e)
        elif (e[2]=='H' and e[3]>=alt[-1][3]) or (e[2]=='L' and e[3]<=alt[-1][3]): alt[-1]=e
    return alt

def candidates(df,tf):
    piv=confirmed_pivots(df,PIVOT_LR[tf]); out=[]
    for k in range(4,len(piv)):
        five=piv[k-4:k+1]; typ=''.join(x[2] for x in five)
        if typ not in ('LHLHL','HLHLH'): continue
        X,A,B,C,D=[x[3] for x in five]; xa=abs(A-X); ab=abs(A-B)
        if xa<=0 or ab<=0: continue
        rb=abs(A-B)/xa; rc=abs(C-B)/ab; rd=abs(A-D)/xa
        if not(PROFILE['B'][0]<=rb<=PROFILE['B'][1] and PROFILE['C'][0]<=rc<=PROFILE['C'][1] and PROFILE['D'][0]<=rd<=PROFILE['D'][1]): continue
        bull=(typ=='LHLHL'); qm=(D<B and C<A) if bull else (D>B and C>A)
        if not qm: continue
        conf_i=five[-1][0]; d_i=five[-1][1]; p0=max(0,d_i-3)
        choch=float(df.high.iloc[p0:d_i].max()) if bull else float(df.low.iloc[p0:d_i].min())
        if np.isfinite(choch): out.append(dict(conf_i=conf_i,d_i=d_i,dir=1 if bull else -1,A=A,D=D,rb=rb,rc=rc,rd=rd,choch=choch))
    return out

def simulate(df,entry_i,c,atr,equity,tf):
    entry=float(df.open.iloc[entry_i]); av=float(atr.iloc[entry_i]) if np.isfinite(atr.iloc[entry_i]) else abs(entry)*0.0005
    # Instrument-neutral buffer: ATR normalized, no fixed XAU dollar buffer.
    buf=max(av*0.15,abs(entry)*1e-7)
    if c['dir']==1: sl=c['D']-buf; R=entry-sl
    else: sl=c['D']+buf; R=sl-entry
    if R<=0: return None
    t1=entry+R*c['dir']; t2=entry+2*R*c['dir']; t3=(max(c['A'],entry+3*R) if c['dir']==1 else min(c['A'],entry-3*R))
    targets=[t1,t2,t3]; parts=[.5,.3,.2]; alive=[1,1,1]
    risk_dollars=equity*RISK_FRAC; pnl_r=0.0; mae_r=0.0; end_i=entry_i
    for i in range(entry_i,min(len(df),entry_i+MAX_HOLD[tf]+1)):
        hi=float(df.high.iloc[i]); lo=float(df.low.iloc[i]); end_i=i
        adverse=((lo-entry)/R if c['dir']==1 else (entry-hi)/R)
        mae_r=min(mae_r,adverse*sum(parts[j] for j,a in enumerate(alive) if a))
        if (lo<=sl if c['dir']==1 else hi>=sl):
            for j,a in enumerate(alive):
                if a: pnl_r-=parts[j]; alive[j]=0
            break
        for j,t in enumerate(targets):
            if alive[j] and (hi>=t if c['dir']==1 else lo<=t):
                pnl_r += parts[j]*abs(t-entry)/R; alive[j]=0
        if not any(alive): break
    if any(alive):
        ex=float(df.close.iloc[end_i]); rr=((ex-entry)/R if c['dir']==1 else (entry-ex)/R)
        for j,a in enumerate(alive):
            if a: pnl_r += parts[j]*rr
    return {'pnl':pnl_r*risk_dollars,'mae':mae_r*risk_dollars,'end_i':end_i,'r_mult':pnl_r}

def run_one(path):
    name=path.stem; bits=name.split('_'); symbol=bits[0]; tf=bits[1]
    df=normalize(pd.read_parquet(path)); df['atr']=atr14(df); cs=candidates(df,tf)
    trades=[]; blocked=-1; equity=INITIAL; peak=INITIAL; max_float_dd=0.0
    for c in cs:
        if c['conf_i']<=blocked: continue
        sig=None
        for i in range(c['conf_i'],min(len(df)-1,c['conf_i']+WAIT[tf]+1)):
            if (c['dir']==1 and float(df.close.iloc[i])>c['choch']) or (c['dir']==-1 and float(df.close.iloc[i])<c['choch']): sig=i; break
        if sig is None or sig+1>=len(df): continue
        r=simulate(df,sig+1,c,df.atr,equity,tf)
        if r is None: continue
        max_float_dd=max(max_float_dd,100*(peak-(equity+r['mae']))/peak)
        equity+=r['pnl']; peak=max(peak,equity); blocked=r['end_i']; trades.append(r)
    if not trades:
        return dict(Symbol=symbol,TF=tf,rows=len(df),N=0,WR=0,PF=0,Net=0,Final=INITIAL,MaxDD_closed=0,MaxDD_float=0,Month21_pct=0)
    t=pd.DataFrame(trades); gp=t.loc[t.pnl>0,'pnl'].sum(); gl=-t.loc[t.pnl<0,'pnl'].sum(); eq=INITIAL+t.pnl.cumsum(); pk=eq.cummax(); dd=((pk-eq)/pk*100).max()
    biz=len({x.date() for x in df.datetime if x.weekday()<5}); final=float(eq.iloc[-1]); mo=((final/INITIAL)**(21/max(1,biz))-1)*100 if final>0 else -100
    return dict(Symbol=symbol,TF=tf,rows=len(df),N=len(t),WR=(t.pnl>0).mean()*100,PF=gp/gl if gl>0 else 9999,Net=final-INITIAL,Final=final,MaxDD_closed=dd,MaxDD_float=max_float_dd,Month21_pct=mo,AvgR=t.r_mult.mean())

files=sorted(Path('parquet').glob('*_M[15]_90d_20260526.parquet'))
rows=[]
for p in files:
    try: rows.append(run_one(p))
    except Exception as e: rows.append({'Symbol':p.stem.split('_')[0],'TF':p.stem.split('_')[1],'status':'ERROR','error':repr(e)})
out=pd.DataFrame(rows); out['status']=out.get('status',pd.Series(index=out.index,dtype=object)).fillna('OK'); out.to_csv(OUT/'per_symbol_tf.csv',index=False)
ok=out[out.status=='OK'].copy()
by_symbol=ok.groupby('Symbol').agg(N=('N','sum'),Net=('Net','sum'),MeanPF=('PF','mean'),BestMonth21=('Month21_pct','max'),WorstDD=('MaxDD_float','max')).reset_index().sort_values(['MeanPF','N'],ascending=[False,False]); by_symbol.to_csv(OUT/'by_symbol.csv',index=False)
summary={'verification':'C0_RECONSTRUCTED_PROXY_NATIVE_M1_M5','profile':'balanced','period':[str(START),str(END)],'risk_per_trade_pct':RISK_FRAC*100,'g75':False,'costs':'none','resample':False,'files':[p.name for p in files],'results':ok.to_dict('records')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
print(out.to_string(index=False)); print('\nBY SYMBOL\n',by_symbol.to_string(index=False))
