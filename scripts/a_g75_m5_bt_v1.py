import json, math
from pathlib import Path
import pandas as pd
import numpy as np

DATA=Path('csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv')
OUT=Path('bt_results/a_g75_m5_v1')
OUT.mkdir(parents=True,exist_ok=True)
PIP=0.10
K_PIPS=np.array([80,100,110,130,150,160,170,190,210,220,230,250,270],dtype=float)
K_VALS=np.array([.07,.08,.09,.10,.11,.12,.13,.14,.15,.16,.17,.18,.20],dtype=float)

def nearest_k(pips):
    return float(K_VALS[np.argmin(np.abs(K_PIPS-pips))])

def add_sr(df,left=3,right=3):
    n=len(df); H=df.high.to_numpy(); L=df.low.to_numpy()
    sh1=np.full(n,np.nan); sh2=np.full(n,np.nan); sl1=np.full(n,np.nan); sl2=np.full(n,np.nan)
    highs=[]; lows=[]
    for i in range(n):
        j=i-right
        if j>=left and j+right < n:
            if H[j] > np.max(H[j-left:j]) and H[j] >= np.max(H[j+1:j+right+1]):
                highs.append(float(H[j])); highs=highs[-2:]
            if L[j] < np.min(L[j-left:j]) and L[j] <= np.min(L[j+1:j+right+1]):
                lows.append(float(L[j])); lows=lows[-2:]
        if len(highs)>=1: sh1[i]=highs[-1]
        if len(highs)>=2: sh2[i]=highs[-2]
        if len(lows)>=1: sl1[i]=lows[-1]
        if len(lows)>=2: sl2[i]=lows[-2]
    df=df.copy(); df['sh1']=sh1;df['sh2']=sh2;df['sl1']=sl1;df['sl2']=sl2
    return df

def orders(row,gap2=45,gap3=130,regime_filter=False):
    if not all(np.isfinite([row.sh1,row.sh2,row.sl1,row.sl2])): return [],0
    R=float(row.sh1); S=float(row.sl1)
    if R<=S: return [],0
    regime=0
    if row.sh1>row.sh2 and row.sl1>row.sl2: regime=1
    elif row.sh1<row.sh2 and row.sl1<row.sl2: regime=-1
    k=nearest_k((R-S)/PIP)
    b1=(math.sqrt(R)-k)**2; s1=(math.sqrt(S)+k)**2
    buys=[b1,b1-gap2*PIP,b1-(gap2+gap3)*PIP]
    sells=[s1,s1+gap2*PIP,s1+(gap2+gap3)*PIP]
    out=[]
    if not regime_filter or regime>=0:
        out += [('buy',j+1,p) for j,p in enumerate(buys)]
    if not regime_filter or regime<=0:
        out += [('sell',j+1,p) for j,p in enumerate(sells)]
    return out,regime

def run(df,base_lot=.001,gap2=45,gap3=130,regime_filter=False,g75=True,trigger=.12,add=.025,reversal=.20,max_adds=10,tp_pips=60,rr=2.0):
    trades=[]; pos=None; equity=1000.; peak_eq=1000.; max_float_dd=0.; base_entries=0; adds_total=0
    sl_dist=(tp_pips/rr)*PIP; tp_dist=tp_pips*PIP
    for i in range(10,len(df)):
        r=df.iloc[i]; prev=df.iloc[i-1]
        # mark-to-market at adverse bar extreme for floating-DD estimate
        if pos:
            side=pos['side']
            adverse=float(r.low if side=='buy' else r.high)
            mtm=sum((((adverse-e['price']) if side=='buy' else (e['price']-adverse))*e['lot']*100) for e in pos['entries'])
            eq=equity+mtm; peak_eq=max(peak_eq,equity)
            max_float_dd=max(max_float_dd,100*(peak_eq-eq)/peak_eq)

        # manage existing basket using CLOSED-BAR G75 ordering
        if pos:
            side=pos['side']; close=float(r.close); high=float(r.high); low=float(r.low)
            total=sum(e['lot'] for e in pos['entries']); avg=sum(e['price']*e['lot'] for e in pos['entries'])/total
            exit_px=None; reason=None
            if not pos['activated']:
                # conservative fixed-SL/TP while G75 has not activated; if both hit, SL wins
                sl=avg-sl_dist if side=='buy' else avg+sl_dist
                tp=avg+tp_dist if side=='buy' else avg-tp_dist
                slhit=(low<=sl) if side=='buy' else (high>=sl)
                tphit=(high>=tp) if side=='buy' else (low<=tp)
                if slhit: exit_px=sl;reason='pre_g75_sl'
                elif tphit: exit_px=tp;reason='pre_g75_tp'
                elif g75:
                    activated=(close>=avg+trigger) if side=='buy' else (close<=avg-trigger)
                    if activated:
                        pos['activated']=True;pos['peak_close']=close;pos['last_add']=avg+trigger if side=='buy' else avg-trigger
            if exit_px is None and pos['activated']:
                if side=='buy':
                    pos['peak_close']=max(pos['peak_close'],close)
                    while pos['adds']<max_adds and close>=pos['last_add']+add:
                        ap=pos['last_add']+add;pos['entries'].append({'price':ap,'lot':base_lot,'kind':'g75'});pos['adds']+=1;adds_total+=1;pos['last_add']=ap
                    if close<=pos['peak_close']-reversal: exit_px=close;reason='g75_reversal'
                else:
                    pos['peak_close']=min(pos['peak_close'],close)
                    while pos['adds']<max_adds and close<=pos['last_add']-add:
                        ap=pos['last_add']-add;pos['entries'].append({'price':ap,'lot':base_lot,'kind':'g75'});pos['adds']+=1;adds_total+=1;pos['last_add']=ap
                    if close>=pos['peak_close']+reversal: exit_px=close;reason='g75_reversal'
            if exit_px is not None:
                pnl=sum((((exit_px-e['price']) if side=='buy' else (e['price']-exit_px))*e['lot']*100) for e in pos['entries'])
                equity+=pnl;peak_eq=max(peak_eq,equity)
                trades.append({'entry_i':pos['start_i'],'exit_i':i,'side':side,'pnl':pnl,'adds':pos['adds'],'base_layers':pos['base_layers'],'reason':reason})
                pos=None
                continue

        if pos is None:
            ods,_=orders(prev,gap2,gap3,regime_filter)  # causal: only info known before current bar
            valid=[]
            for side,layer,p in ods:
                # MT5 limit validity at previous close plus fill in current bar
                if side=='buy' and p<float(prev.close) and float(r.low)<=p<=float(r.high): valid.append((side,layer,p))
                if side=='sell' and p>float(prev.close) and float(r.low)<=p<=float(r.high): valid.append((side,layer,p))
            sides=set(x[0] for x in valid)
            if len(sides)==1:
                side=next(iter(sides)); fills=sorted([x for x in valid if x[0]==side],key=lambda x:x[1])
                entries=[{'price':p,'lot':base_lot,'kind':f'L{layer}'} for _,layer,p in fills]
                base_entries+=len(entries)
                pos={'side':side,'entries':entries,'base_layers':len(entries),'start_i':i,'activated':False,'adds':0,'last_add':sum(e['price'] for e in entries)/len(entries),'peak_close':float(r.close)}
    t=pd.DataFrame(trades)
    if len(t)==0: return {}
    gp=t.loc[t.pnl>0,'pnl'].sum();gl=-t.loc[t.pnl<0,'pnl'].sum();eq=1000+t.pnl.cumsum();pk=eq.cummax();dd=(pk-eq)/pk*100
    days=pd.Series(df.datetime.dt.date).nunique()
    biz=len(sorted(set(d for d in df.datetime.dt.date if pd.Timestamp(d).weekday()<5)))
    final=1000+t.pnl.sum();month=((final/1000)**(21/max(1,biz))-1)*100 if final>0 else -100
    return {'N':len(t),'WR':(t.pnl>0).mean()*100,'PF':gp/gl if gl>0 else 9999,'Net':t.pnl.sum(),'Final':final,'MaxDD_closed':dd.max(),'MaxDD_float':max_float_dd,'BaseEntries':base_entries,'G75Adds':adds_total,'AvgAdds':t['adds'].mean(),'Month21_pct':month,'BusinessDays':biz}

def main():
    df=pd.read_csv(DATA);df.columns=[c.lower() for c in df.columns];df['datetime']=pd.to_datetime(df['datetime']);df=df.sort_values('datetime').reset_index(drop=True);df=add_sr(df)
    rows=[]
    configs=[]
    for rf in [False,True]:
      for gaps in [(45,130),(30,100),(60,160)]:
       for lot in [.0002,.0003,.0004,.0005,.00075,.001]:
        configs.append((rf,gaps[0],gaps[1],lot))
    for rf,g2,g3,lot in configs:
        base=run(df,base_lot=lot,gap2=g2,gap3=g3,regime_filter=rf,g75=False)
        g=run(df,base_lot=lot,gap2=g2,gap3=g3,regime_filter=rf,g75=True)
        rows.append({'mode':'BASE','regime_filter':rf,'gap2':g2,'gap3':g3,'lot':lot,**base})
        rows.append({'mode':'G75','regime_filter':rf,'gap2':g2,'gap3':g3,'lot':lot,**g})
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'all_results.csv',index=False)
    feasible=out[(out['mode']=='G75')&(out['MaxDD_float']<=5.0)].sort_values(['Month21_pct','PF'],ascending=False)
    feasible.to_csv(OUT/'feasible_dd5.csv',index=False)
    summary={'data_rows':len(df),'start':str(df.datetime.min()),'end':str(df.datetime.max()),'top_dd5':feasible.head(15).to_dict(orient='records')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
