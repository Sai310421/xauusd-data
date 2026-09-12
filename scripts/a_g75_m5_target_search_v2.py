import pandas as pd, numpy as np, math
from pathlib import Path

DATA = Path('csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv')
OUT = Path('bt_results/a_g75_m5_target_search_v2')
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df['datetime'] = pd.to_datetime(df['datetime'])
for c in ['open','high','low','close']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna().reset_index(drop=True)
close=df.close
ema9=close.ewm(span=9,adjust=False).mean(); ema21=close.ewm(span=21,adjust=False).mean()

win=2; sup=np.nan; res=np.nan
last_sup=np.full(len(df),np.nan); last_res=np.full(len(df),np.nan)
for i in range(len(df)):
    j=i-win
    if j>=win and i>=2*win:
        sl=df.low.iloc[j-win:j+win+1]; sh=df.high.iloc[j-win:j+win+1]
        if df.low.iloc[j]==sl.min(): sup=df.low.iloc[j]
        if df.high.iloc[j]==sh.max(): res=df.high.iloc[j]
    last_sup[i]=sup; last_res[i]=res

pip=0.1
k_dist=np.array([80,100,110,130,150,160,170,190,210,220,230,250,270],float)
k_val=np.array([.07,.08,.09,.10,.11,.12,.13,.14,.15,.16,.17,.18,.20],float)
def kval(d): return float(k_val[np.argmin(np.abs(k_dist-d))])

def entries(regime=False,pending=12):
    out=[]; seen=set()
    for i in range(30,len(df)-2):
        s=last_sup[i]; r=last_res[i]
        if not np.isfinite(s) or not np.isfinite(r) or r<=s: continue
        k=kval((r-s)/pip)
        levels=[]
        if (not regime) or ema9.iloc[i]>ema21.iloc[i]: levels.append(('BUY',(math.sqrt(r)-k)**2))
        if (not regime) or ema9.iloc[i]<ema21.iloc[i]: levels.append(('SELL',(math.sqrt(s)+k)**2))
        for side,px in levels:
            for j in range(i+1,min(len(df),i+pending+1)):
                if df.low.iloc[j] <= px <= df.high.iloc[j]:
                    key=(j,side,round(px,2))
                    if key not in seen:
                        seen.add(key); out.append((j,side,px))
                    break
    return out

def sim(es,lot=.001,trig=.12,step=.025,maxadds=10,maxhold=96):
    eq=1000.; peak=eq; mddf=0.; mddc=0.; rec=[]
    for idx,side,e0 in es:
        if idx>=len(df)-2: continue
        fills=[(e0,lot)]; adds=0; last=e0; exi=min(len(df)-1,idx+maxhold); exp=df.close.iloc[exi]
        eq0=eq; worst=0.
        for j in range(idx+1,min(len(df),idx+maxhold+1)):
            while adds<maxadds:
                gap=trig if adds==0 else step
                nxt=last+gap if side=='BUY' else last-gap
                touched=df.high.iloc[j]>=nxt if side=='BUY' else df.low.iloc[j]<=nxt
                if not touched: break
                fills.append((nxt,lot)); last=nxt; adds+=1
            mark=df.low.iloc[j] if side=='BUY' else df.high.iloc[j]
            fp=sum(((mark-p) if side=='BUY' else (p-mark))*l*100 for p,l in fills)
            worst=min(worst,fp)
            e9=ema9.iloc[j]
            if (side=='BUY' and df.high.iloc[j]>=e9) or (side=='SELL' and df.low.iloc[j]<=e9):
                exi=j; exp=e9; break
        pnl=sum(((exp-p) if side=='BUY' else (p-exp))*l*100 for p,l in fills)
        eq += pnl; peak=max(peak,eq)
        mddc=max(mddc,(peak-eq)/peak*100); mddf=max(mddf,-worst/max(eq0,1e-9)*100)
        rec.append((pnl,adds))
    if not rec: return None
    p=np.array([x[0] for x in rec]); gp=p[p>0].sum(); gl=-p[p<0].sum()
    bdays=max(1,np.busday_count(df.datetime.iloc[0].date(),df.datetime.iloc[-1].date())+1)
    month=((eq/1000.)**(21./bdays)-1)*100 if eq>0 else -100
    return dict(N=len(p),WR=(p>0).mean()*100,PF=gp/gl if gl>0 else np.inf,Net=eq-1000,Final=eq,MaxDD_closed=mddc,MaxDD_float=mddf,Month21_pct=month,G75Adds=sum(x[1] for x in rec),AvgAdds=np.mean([x[1] for x in rec]))

rows=[]
for regime in [False,True]:
  for pending in [6,12,24]:
    es=entries(regime,pending)
    for lot in [0.0010,0.0011,0.0012,0.00125,0.0013,0.00135,0.0014,0.00145,0.0015]:
      for trig in [0.08,0.10,0.12,0.15,0.20]:
        for step in [0.020,0.025,0.030,0.040,0.050]:
          for ma in [4,6,8,10]:
            r=sim(es,lot,trig,step,ma)
            if r:
              r.update(regime_filter=regime,pending_bars=pending,lot=lot,trigger=trig,step=step,max_adds=ma)
              rows.append(r)
res=pd.DataFrame(rows)
res.to_csv(OUT/'all_results.csv',index=False)
valid=res[(res.MaxDD_float<=5)&(res.Month21_pct>=50)].sort_values(['Month21_pct','PF'],ascending=[False,False])
valid.to_csv(OUT/'target_hits.csv',index=False)
best=res[res.MaxDD_float<=5].sort_values(['Month21_pct','PF'],ascending=[False,False]).head(100)
best.to_csv(OUT/'best_dd5.csv',index=False)
print('rows',len(res),'hits',len(valid))
print(best.head(20).to_string(index=False))
print('TARGETS')
print(valid.head(20).to_string(index=False) if len(valid) else 'NONE')