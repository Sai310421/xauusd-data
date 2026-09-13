import runpy, json, math
from pathlib import Path
import pandas as pd, numpy as np

# Reuse the validated C0 multisymbol causal detector/normalizer.
ns=runpy.run_path('scripts/c0_multisymbol_m1m5_bt_v1.py')
normalize=ns['normalize']; atr14=ns['atr14']; candidates=ns['candidates']; WAIT=ns['WAIT']; INITIAL=1000.0; RISK_FRAC=0.0025
OUT=Path('bt_results/c0_g75_top2_m1_opt_v1'); OUT.mkdir(parents=True,exist_ok=True)
SYMS=['EURUSD','XAUUSD']; TF='M1'; MAX_HOLD=180
TRIGGERS=[0.15,0.25,0.35]; ADD_STEPS=[0.10,0.15]; REVERSALS=[0.25,0.35,0.50]; MAX_ADDS=[3,5,10]; ADD_FACTORS=[0.25,0.50,1.00]

def make_setups(df):
    df=df.copy(); df['atr']=atr14(df); out=[]; blocked=-1
    for c in candidates(df,TF):
        if c['conf_i']<=blocked: continue
        sig=None
        for i in range(c['conf_i'],min(len(df)-1,c['conf_i']+WAIT[TF]+1)):
            if (c['dir']==1 and float(df.close.iloc[i])>c['choch']) or (c['dir']==-1 and float(df.close.iloc[i])<c['choch']): sig=i; break
        if sig is None or sig+1>=len(df): continue
        entry_i=sig+1; entry=float(df.open.iloc[entry_i]); av=float(df.atr.iloc[entry_i]) if np.isfinite(df.atr.iloc[entry_i]) else abs(entry)*0.0005
        buf=max(av*.15,abs(entry)*1e-7); sl=c['D']-buf if c['dir']==1 else c['D']+buf; R=(entry-sl) if c['dir']==1 else (sl-entry)
        if R<=0: continue
        end=min(len(df)-1,entry_i+MAX_HOLD)
        out.append(dict(c=c,entry_i=entry_i,entry=entry,atr=av,sl=sl,R=R,end=end))
        blocked=end
    return out

def sim(df,s,par,equity):
    c=s['c']; d=c['dir']; entry=s['entry']; R=s['R']; sl=s['sl']; av=s['atr']; end=s['end']
    risk=equity*RISK_FRAC; base_v=risk/R
    parts=[.5,.3,.2]; alive=[1,1,1]; targets=[entry+d*R,entry+d*2*R,(max(c['A'],entry+3*R) if d==1 else min(c['A'],entry-3*R))]
    base_pnl=0.; adds=[]; g75_pnl=0.; activated=False; last_ref=entry; peak=entry; trough=entry; add_count=0; mae=0.; exit_i=end
    for i in range(s['entry_i'],end+1):
        hi=float(df.high.iloc[i]); lo=float(df.low.iloc[i]); cl=float(df.close.iloc[i]);
        peak=max(peak,hi); trough=min(trough,lo)
        # portfolio floating adverse estimate from base + active G75 legs
        base_open=sum(parts[j] for j,a in enumerate(alive) if a)*base_v
        g_open=sum(x['v'] for x in adds if x['alive'])
        adverse=((lo-entry) if d==1 else (entry-hi))*base_open
        for x in adds:
            if x['alive']: adverse += ((lo-x['px']) if d==1 else (x['px']-hi))*x['v']
        mae=min(mae,adverse)
        # Base SL first (conservative); also closes G75.
        if (lo<=sl if d==1 else hi>=sl):
            for j,a in enumerate(alive):
                if a: base_pnl += -parts[j]*risk; alive[j]=0
            for x in adds:
                if x['alive']: g75_pnl += d*(sl-x['px'])*x['v']; x['alive']=0
            exit_i=i; break
        # Base targets.
        for j,t in enumerate(targets):
            if alive[j] and (hi>=t if d==1 else lo<=t):
                base_pnl += parts[j]*risk*abs(t-entry)/R; alive[j]=0
        # G75 causal activation/add decisions from close; fills next bar open.
        fav=d*(cl-entry)
        if not activated and fav>=par['tr']*av and i+1<=end:
            activated=True; last_ref=entry
        if activated and add_count<par['ma'] and d*(cl-last_ref)>=par['ad']*av and i+1<=end:
            px=float(df.open.iloc[i+1]); v=base_v*par['af']; adds.append({'px':px,'v':v,'alive':1}); add_count+=1; last_ref=px
        # Reversal exit for G75 legs only.
        if activated and adds:
            rev=((peak-cl) if d==1 else (cl-trough))
            if rev>=par['rv']*av:
                for x in adds:
                    if x['alive']: g75_pnl += d*(cl-x['px'])*x['v']; x['alive']=0
                activated=False; last_ref=cl; peak=cl; trough=cl
        if not any(alive) and not any(x['alive'] for x in adds): exit_i=i; break
        exit_i=i
    ex=float(df.close.iloc[exit_i])
    for j,a in enumerate(alive):
        if a: base_pnl += d*(ex-entry)*base_v*parts[j]
    for x in adds:
        if x['alive']: g75_pnl += d*(ex-x['px'])*x['v']
    return base_pnl+g75_pnl, mae, add_count, base_pnl, g75_pnl

def run_symbol(sym,par):
    path=Path(f'parquet/{sym}_M1_90d_20260526.parquet'); df=normalize(pd.read_parquet(path)); setups=make_setups(df)
    eq=INITIAL; peak=INITIAL; maxdd=0.; mfdd=0.; pnls=[]; adds=0; gp=gl=0.; bp=gg=0.
    for s in setups:
        pnl,mae,nadd,bpnl,gpnl=sim(df,s,par,eq); mfdd=max(mfdd,100*(peak-(eq+mae))/peak); eq+=pnl; peak=max(peak,eq); maxdd=max(maxdd,100*(peak-eq)/peak); pnls.append(pnl); adds+=nadd; bp+=bpnl; gg+=gpnl
    if pnls:
        a=np.array(pnls); gp=a[a>0].sum(); gl=-a[a<0].sum(); wr=(a>0).mean()*100; pf=gp/gl if gl>0 else 9999
    else: wr=pf=0
    biz=len({x.date() for x in df.datetime if x.weekday()<5}); mo=((eq/INITIAL)**(21/max(1,biz))-1)*100 if eq>0 else -100
    return dict(Symbol=sym,N=len(pnls),WR=wr,PF=pf,Net=eq-INITIAL,Final=eq,MaxDD_closed=maxdd,MaxDD_float=mfdd,Month21_pct=mo,G75Adds=adds,BasePnL=bp,G75PnL=gg,**par)

rows=[]
for tr in TRIGGERS:
 for ad in ADD_STEPS:
  for rv in REVERSALS:
   for ma in MAX_ADDS:
    for af in ADD_FACTORS:
     par={'tr':tr,'ad':ad,'rv':rv,'ma':ma,'af':af}
     sub=[run_symbol(s,par) for s in SYMS]; rows.extend(sub)
     # shared-equity reference: sequential symbol ledgers combined only approximately via summed Net; DD=max constituent.
     rows.append(dict(Symbol='COMBINED_REF',N=sum(x['N'] for x in sub),WR=np.average([x['WR'] for x in sub],weights=[max(1,x['N']) for x in sub]),PF=np.average([x['PF'] for x in sub]),Net=sum(x['Net'] for x in sub),Final=INITIAL+sum(x['Net'] for x in sub),MaxDD_closed=max(x['MaxDD_closed'] for x in sub),MaxDD_float=max(x['MaxDD_float'] for x in sub),Month21_pct=sum(x['Month21_pct'] for x in sub),G75Adds=sum(x['G75Adds'] for x in sub),BasePnL=sum(x['BasePnL'] for x in sub),G75PnL=sum(x['G75PnL'] for x in sub),**par))
out=pd.DataFrame(rows); out.to_csv(OUT/'all_candidates.csv',index=False)
# rank combined under DD<=5 first, then per symbol
comb=out[out.Symbol=='COMBINED_REF'].copy(); safe=comb[comb.MaxDD_float<=5].sort_values(['Month21_pct','PF'],ascending=False); safe.head(30).to_csv(OUT/'top_combined_dd5.csv',index=False)
for s in SYMS: out[out.Symbol==s].sort_values(['Month21_pct','PF'],ascending=False).head(30).to_csv(OUT/f'top_{s}.csv',index=False)
summary={'verification':'C0_G75_RECONSTRUCTED_PROXY_NATIVE_M1_CAUSAL','symbols':SYMS,'risk_per_base_trade_pct':RISK_FRAC*100,'g75':'ATR-normalized same-direction profit chase; next-bar-open adds','costs':'none','resample':False,'best_dd5':safe.head(10).to_dict('records')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); print(safe.head(15).to_string(index=False))
