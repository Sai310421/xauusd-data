import json, math
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv")
OUT = Path("bt_results/xau_hybrid_v031")
OUT.mkdir(parents=True, exist_ok=True)

P = {
    "RSI_Period":14,"RSI_Low":10.0,"RSI_High":90.0,"RSI_Mid":50.0,"RSI_ExtremeLookback":12,
    "StochK":5,"StochD":9,"StochSlowing":3,"StochLow":15.0,"StochHigh":85.0,"StochMid":50.0,
    "MACD_Fast":12,"MACD_Slow":26,"RequireMACDSlope":True,"RequireMACDAcceleration":False,
    "ADX_Period":14,"ADX_Range":20.0,"ADX_Trend":23.0,
    "ATR_Fast":14,"ATR_Slow":100,"ATR_RangeRatio":0.80,
    "BB_Period":20,"BB_Dev":2.0,
    "RangeLookback":24,"MinRangeATR":1.50,"SweepATRBuffer":0.15,
    "CRTThreshold":75,"CRTTimeoutMinutes":30,"TrendERMin":0.40,
    "BaseTrendLot":0.01,"MaxLegs":5,
    "CBLot":0.01,"CBMinMultiplier":0.20,"CBMaxNetLots":0.20,"CBMaxTotalLots":1.00,
    "CBWeightMeanRev":1.0,"CBWeightMomentum":1.0,"CBWeightInventory":0.75,"CBWeightCRTMax":3.0,
    "MaxAccountDDPercent":10.0,"MaxBasketLossUSD":500.0,
}

def atr(df,n):
    pc=df.close.shift(1)
    tr=pd.concat([(df.high-df.low).abs(),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def rsi(s,n):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=up/dn.replace(0,np.nan)
    out=100-100/(1+rs)
    return out.fillna(50)

def adx(df,n):
    up=df.high.diff(); dn=-df.low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0)
    minus=np.where((dn>up)&(dn>0),dn,0.0)
    a=atr(df,n)
    pdi=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()/a
    mdi=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean().fillna(0)

def resample_ohlc(m1, rule, minutes):
    x=m1.set_index("datetime")[["open","high","low","close","volume"]]
    y=x.resample(rule, label="left", closed="left").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    y.index=y.index+pd.Timedelta(minutes=minutes)
    return y

m1=pd.read_csv(DATA,parse_dates=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
# M1 timestamps are bar-open times; move to close times.
m1["datetime"]=m1["datetime"]+pd.Timedelta(minutes=1)
m1=m1.set_index("datetime")
m5=resample_ohlc(m1.reset_index().assign(datetime=lambda x:x.datetime-pd.Timedelta(minutes=1)),"5min",5)
m15=resample_ohlc(m1.reset_index().assign(datetime=lambda x:x.datetime-pd.Timedelta(minutes=1)),"15min",15)

# Indicators
m15["rsi"]=rsi(m15.close,P["RSI_Period"])
ema_fast=m15.close.ewm(span=P["MACD_Fast"],adjust=False).mean()
ema_slow=m15.close.ewm(span=P["MACD_Slow"],adjust=False).mean()
m15["macd"]=ema_fast-ema_slow
m15["macd_slope"]=m15.macd-m15.macd.shift(1)
m15["macd_accel"]=m15.macd_slope-m15.macd_slope.shift(1)
m15["rsi_low_seen"]=m15.rsi.shift(1).rolling(P["RSI_ExtremeLookback"]).min()<=P["RSI_Low"]
m15["rsi_high_seen"]=m15.rsi.shift(1).rolling(P["RSI_ExtremeLookback"]).max()>=P["RSI_High"]
m15["rsi_reclaim_up"]=(m15.rsi.shift(1)<=P["RSI_Low"])&(m15.rsi>P["RSI_Low"])
m15["rsi_reclaim_dn"]=(m15.rsi.shift(1)>=P["RSI_High"])&(m15.rsi<P["RSI_High"])

m5["atr14"]=atr(m5,P["ATR_Fast"]); m5["atr100"]=atr(m5,P["ATR_Slow"])
m5["atr_ratio"]=m5.atr14/m5.atr100
m5["adx"]=adx(m5,P["ADX_Period"])
mid=m5.close.rolling(P["BB_Period"]).mean()
sd=m5.close.rolling(P["BB_Period"]).std(ddof=0)
m5["bb_width"]=(2*P["BB_Dev"]*sd)/mid.abs()
m5["bb_pct"]=m5.bb_width.rolling(50).apply(lambda a: np.mean(a[:-1] < a[-1]) if len(a)==50 else np.nan,raw=True)
m5["adx_falling"]=(m5.adx<m5.adx.shift(1))&(m5.adx.shift(1)<m5.adx.shift(2))
m5["adx_expanding"]=m5.adx>m5.adx.shift(1)
m5["range_state"]=(m5.bb_pct<=0.20)&(m5.adx<P["ADX_Range"])&m5.adx_falling&(m5.atr_ratio<P["ATR_RangeRatio"])
m5["range_hi"]=m5.high.rolling(P["RangeLookback"]).max()
m5["range_lo"]=m5.low.rolling(P["RangeLookback"]).min()
m5["range_eq"]=(m5.range_hi+m5.range_lo)/2
m5["range_width_ok"]=(m5.range_hi-m5.range_lo)>=P["MinRangeATR"]*m5.atr14
# ER
direction=(m5.close-m5.close.shift(20)).abs()
noise=m5.close.diff().abs().rolling(20).sum()
m5["er20"]=(direction/noise.replace(0,np.nan)).fillna(0)

# Stochastic main = raw %K smoothed by Slowing
ll=m1.low.rolling(P["StochK"]).min(); hh=m1.high.rolling(P["StochK"]).max()
rawk=100*(m1.close-ll)/(hh-ll).replace(0,np.nan)
m1["stoch"]=rawk.rolling(P["StochSlowing"]).mean()
m1["stoch_reclaim_up"]=(m1.stoch.shift(1)<=P["StochLow"])&(m1.stoch>P["StochLow"])
m1["stoch_reclaim_dn"]=(m1.stoch.shift(1)>=P["StochHigh"])&(m1.stoch<P["StochHigh"])
m1["stoch50_up"]=(m1.stoch.shift(1)<P["StochMid"])&(m1.stoch>=P["StochMid"])
m1["stoch50_dn"]=(m1.stoch.shift(1)>P["StochMid"])&(m1.stoch<=P["StochMid"])
m1["m1_break_up"]=m1.close>m1.high.shift(1)
m1["m1_break_dn"]=m1.close<m1.low.shift(1)

# Align latest completed M5/M15 values to each M1 close
m5a=m5.reindex(m1.index,method="ffill")
m15a=m15.reindex(m1.index,method="ffill")

def m5_pos(t):
    try: return m5.index.get_loc(m5a.loc[t].name if False else m5.index[m5.index<=t][-1])
    except: return None

def macd_aligned(row15, d):
    if pd.isna(row15.macd): return False
    if d>0 and row15.macd<=0: return False
    if d<0 and row15.macd>=0: return False
    if P["RequireMACDSlope"]:
        if d>0 and row15.macd_slope<=0: return False
        if d<0 and row15.macd_slope>=0: return False
    if P["RequireMACDAcceleration"]:
        if d>0 and row15.macd_accel<=0: return False
        if d<0 and row15.macd_accel>=0: return False
    return True

def detect_m5_features(i,d):
    if i is None or i<25: return dict(dis=False,cisd=False,br=False,fvg=False,ifvg=False,bpr=False)
    r=m5.iloc
    cur=r[i]
    avg=np.mean(np.abs(m5.close.iloc[i-20:i]-m5.open.iloc[i-20:i]))
    body=abs(cur.close-cur.open); brange=cur.high-cur.low
    dis=((d>0 and cur.close>cur.open) or (d<0 and cur.close<cur.open)) and body>1.8*avg and brange>1.2*cur.atr14
    cisd=False
    for k in range(i-1,max(-1,i-9),-1):
        bar=r[k]
        if d>0 and bar.close<bar.open and cur.close>bar.open: cisd=True; break
        if d<0 and bar.close>bar.open and cur.close<bar.open: cisd=True; break
    prev=m5.iloc[i-3:i]
    breaker=(cur.close>prev.high.max()) if d>0 else (cur.close<prev.low.min())
    fvg=(cur.low>m5.iloc[i-2].high) if d>0 else (cur.high<m5.iloc[i-2].low)
    ifvg=False
    for lag in range(3,9):
        a=i-lag; b=a-2
        if b<0: continue
        if d>0 and m5.iloc[a].high<m5.iloc[b].low and cur.close>m5.iloc[b].low: ifvg=True; break
        if d<0 and m5.iloc[a].low>m5.iloc[b].high and cur.close<m5.iloc[b].high: ifvg=True; break
    bulls=[]; bears=[]
    for k in range(max(2,i-11),i+1):
        if m5.iloc[k].low>m5.iloc[k-2].high: bulls.append((m5.iloc[k-2].high,m5.iloc[k].low))
        if m5.iloc[k].high<m5.iloc[k-2].low: bears.append((m5.iloc[k].high,m5.iloc[k-2].low))
    bpr=any(max(a[0],b[0])<min(a[1],b[1]) for a in bulls for b in bears)
    return dict(dis=dis,cisd=cisd,br=breaker,fvg=fvg,ifvg=ifvg,bpr=bpr)

def run(name, cb_mode="off", initial_balance=1000.0):
    state="SEARCH"; range_locked=False; rh=rl=req=np.nan
    crt_dir=0; crt_score=0; crt_start=None; cycle_dir=0; legs=[False]*5
    pos=[]; balance=initial_balance; peak=initial_balance; maxdd=0; halted=False
    trades_opened=0; cb_count=0; leg_count=[0]*5; state_counts={s:0 for s in ["SEARCH","RANGE","TRANSITION","EXPANSION","TREND"]}
    risk_reason=""; equity_curve=[]; last_m5_time=None

    def floating(price):
        return sum((price-p["entry"])*p["dir"]*100.0*p["lot"] for p in pos)
    def inv():
        L=sum(p["lot"] for p in pos if p["dir"]>0); S=sum(p["lot"] for p in pos if p["dir"]<0)
        return L,S,L-S
    def openpos(d,lot,kind):
        nonlocal trades_opened,cb_count
        if halted or lot<=0:return False
        pos.append({"dir":d,"lot":lot,"entry":price,"kind":kind})
        trades_opened+=1
        if kind=="CB":cb_count+=1
        return True
    def openleg(d,k):
        if k<0 or k>=5 or k>=P["MaxLegs"] or legs[k]: return False
        ok=openpos(d,P["BaseTrendLot"],f"Leg{k+1}")
        if ok: legs[k]=True; leg_count[k]+=1
        return ok

    for t,row1 in m1.iterrows():
        price=float(row1.close)
        r5=m5a.loc[t]; r15=m15a.loc[t]
        if pd.isna(r5.adx) or pd.isna(r15.rsi): continue
        state_counts[state]+=1
        eq=balance+floating(price); peak=max(peak,eq)
        dd=100*(peak-eq)/peak if peak>0 else 0; maxdd=max(maxdd,dd)
        equity_curve.append((t,eq,dd,len(pos)))
        if dd>=P["MaxAccountDDPercent"] or floating(price)<=-P["MaxBasketLossUSD"]:
            balance += floating(price)
            pos.clear(); halted=True
            risk_reason=f"DD {dd:.2f}%" if dd>=P["MaxAccountDDPercent"] else f"Basket {floating(price):.2f}"
            break

        dhtf=0
        if r15.rsi>P["RSI_Mid"] and macd_aligned(r15,1): dhtf=1
        elif r15.rsi<P["RSI_Mid"] and macd_aligned(r15,-1): dhtf=-1

        expansion=lambda d: d!=0 and r5.adx>P["ADX_Trend"] and r5.adx>m5.adx.shift(1).reindex([r5.name]).iloc[0] and r5.atr_ratio>1.0
        trend_end = r5.adx<P["ADX_Range"] and r5.er20<0.25 and r5.atr_ratio<P["ATR_RangeRatio"]

        # New M5 bar marker for pattern calculations
        mt = m5.index[m5.index<=t][-1] if len(m5.index[m5.index<=t]) else None
        mi = m5.index.get_loc(mt) if mt is not None else None

        if state=="SEARCH":
            if bool(r5.range_state) and bool(r5.range_width_ok):
                rh=float(r5.range_hi); rl=float(r5.range_lo); req=float(r5.range_eq); range_locked=True; state="RANGE"
                continue
            if dhtf and expansion(dhtf) and r5.er20>=P["TrendERMin"]:
                cycle_dir=dhtf; crt_dir=dhtf; legs=[False]*5; state="TREND"
        elif state=="RANGE":
            sweep=0
            if range_locked:
                b=float(r5.atr14)*P["SweepATRBuffer"]
                if r5.low<rl-b and r5.close>rl: sweep=1
                elif r5.high>rh+b and r5.close<rh: sweep=-1
            if sweep:
                crt_dir=sweep; crt_start=t; crt_score=25; state="TRANSITION"
        elif state=="TRANSITION":
            if crt_start is None or (t-crt_start).total_seconds()>P["CRTTimeoutMinutes"]*60:
                crt_dir=0; crt_score=0; range_locked=False; state="SEARCH"; continue
            f=detect_m5_features(mi,crt_dir)
            crt_score=25+20*f["dis"]+15*f["cisd"]+10*f["br"]+10*f["ifvg"]+5*f["bpr"]+5*f["fvg"]+5*bool(r5.adx_expanding)+5*(r5.atr_ratio>1.0)
            if (crt_dir>0 and bool(r15.rsi_reclaim_up)) or (crt_dir<0 and bool(r15.rsi_reclaim_dn)): crt_score+=5
            eqbreak=(r5.close>req) if crt_dir>0 else (r5.close<req)
            if crt_score>=P["CRTThreshold"] and eqbreak:
                cycle_dir=crt_dir; legs=[False]*5; state="EXPANSION"
        elif state=="EXPANSION":
            if crt_dir==0: continue
            if not legs[0]:
                initial=((crt_dir>0 and (bool(r15.rsi_reclaim_up) or bool(row1.stoch_reclaim_up))) or
                         (crt_dir<0 and (bool(r15.rsi_reclaim_dn) or bool(row1.stoch_reclaim_dn))) or
                         (macd_aligned(r15,crt_dir) and ((bool(row1.m1_break_up) if crt_dir>0 else bool(row1.m1_break_dn)) or expansion(crt_dir))))
                if initial or crt_score>=P["CRTThreshold"]: openleg(crt_dir,0)
                if not legs[0]: continue
            if expansion(crt_dir): state="TREND"
        elif state=="TREND":
            d=cycle_dir
            if d:
                initial=((d>0 and (bool(r15.rsi_reclaim_up) or bool(row1.stoch_reclaim_up))) or
                         (d<0 and (bool(r15.rsi_reclaim_dn) or bool(row1.stoch_reclaim_dn))) or
                         (macd_aligned(r15,d) and ((bool(row1.m1_break_up) if d>0 else bool(row1.m1_break_dn)) or expansion(d))))
                if not legs[0] and initial: openleg(d,0)
                if not legs[1] and ((d>0 and bool(row1.stoch_reclaim_up)) or (d<0 and bool(row1.stoch_reclaim_dn))): openleg(d,1)
                if not legs[2] and ((d>0 and bool(row1.stoch50_up)) or (d<0 and bool(row1.stoch50_dn))): openleg(d,2)
                if not legs[3] and ((d>0 and bool(row1.m1_break_up)) or (d<0 and bool(row1.m1_break_dn))): openleg(d,3)
                if not legs[4] and mi is not None and mi>=3:
                    m5br=(r5.close>m5.iloc[mi-2:mi].high.max()) if d>0 else (r5.close<m5.iloc[mi-2:mi].low.min())
                    if m5br and expansion(d): openleg(d,4)
            if trend_end:
                crt_dir=0; crt_score=0; range_locked=False; cycle_dir=0; legs=[False]*5; state="SEARCH"

        # Conservative CB approximation: max one entry per M1 bar, vs EA target every 5 sec.
        if cb_mode=="1min" and state in ("RANGE","TRANSITION"):
            L,S,N=inv()
            total=L+S
            if total+P["CBLot"]<=P["CBMaxTotalLots"]:
                mom=(1 if row1.close>m1.close.shift(1).loc[t] else -1)+(1 if row1.close>m1.close.shift(3).loc[t] else -1)
                if row1.close>m1.high.shift(1).loc[t]: mom+=0.5
                if row1.close<m1.low.shift(1).loc[t]: mom-=0.5
                mr=0
                if range_locked and rh>rl: mr=-((price-req)/((rh-rl)/2))
                invs=0 if abs(N)<1e-12 else -max(-1,min(1,N/P["CBMaxNetLots"]))
                crtb=0 if crt_dir==0 or crt_score<=0 else crt_dir*max(0,min(1,crt_score/P["CRTThreshold"]))
                score=mom*P["CBWeightMomentum"]+mr*P["CBWeightMeanRev"]+invs*P["CBWeightInventory"]+crtb*P["CBWeightCRTMax"]
                pref=1 if score>=0 else -1
                lot=P["CBLot"]
                if crt_dir and crt_score>0:
                    prog=max(0,min(1,crt_score/P["CRTThreshold"]))
                    lot=P["CBLot"]*max(P["CBMinMultiplier"],1-prog)
                def can(d):
                    fut=N+lot*(1 if d>0 else -1)
                    return total+lot<=P["CBMaxTotalLots"] and abs(fut)<=P["CBMaxNetLots"]
                dd= pref if can(pref) else (-pref if can(-pref) else 0)
                if dd: openpos(dd,lot,"CB")

    # forced mark-to-market close for reporting only
    if len(m1):
        last=float(m1.close.iloc[min(len(equity_curve)-1,len(m1)-1)]) if equity_curve else float(m1.close.iloc[-1])
    else: last=0.0
    if pos:
        balance += floating(last)
        pos.clear()
    ret=100*(balance/initial_balance-1)
    gross_pos=0; gross_neg=0
    # This EA has no normal close events, so trade-level PF/WR are not meaningful.
    out={
        "scenario":name,"initial_balance":initial_balance,"final_balance_forced_close":round(balance,2),
        "return_pct_forced_close":round(ret,3),"max_equity_dd_pct":round(maxdd,3),
        "risk_halted":halted,"risk_reason":risk_reason,"orders_opened":trades_opened,
        "cb_orders":cb_count,"leg_orders":sum(leg_count),"leg_counts":leg_count,
        "state_minutes":state_counts,
        "pf":"N/A - no normal exit logic","win_rate":"N/A - no normal exit logic",
        "note":"Final balance uses forced end-of-test close; EA v0.31 itself only closes positions on hard risk stop."
    }
    ec=pd.DataFrame(equity_curve,columns=["time","equity","dd_pct","open_positions"])
    ec.to_csv(OUT/f"equity_{name}.csv",index=False)
    return out

results=[
    run("core_cb_off_1000",cb_mode="off",initial_balance=1000.0),
    run("cb_1min_lowerbound_1000",cb_mode="1min",initial_balance=1000.0),
    run("core_cb_off_300",cb_mode="off",initial_balance=300.0),
    run("cb_1min_lowerbound_300",cb_mode="1min",initial_balance=300.0),
]
(OUT/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame(results).to_csv(OUT/"summary.csv",index=False)
md=["# XAUUSD Hybrid EA v0.31 - bar-level diagnostic BT","",
    "Data: repo M1 Q1/Q2 2026, M5/M15 derived from M1.","Execution: M1-close approximation. CB 5-second behavior cannot be reproduced from M1 bars.",
    "The 1-minute CB scenario is intentionally a conservative lower-bound on order frequency.",""]
for x in results:
    md += [f"## {x['scenario']}",f"- Return (forced close): {x['return_pct_forced_close']}%",
           f"- Max equity DD: {x['max_equity_dd_pct']}%",f"- Orders: {x['orders_opened']} (CB {x['cb_orders']}, legs {x['leg_orders']})",
           f"- Risk halted: {x['risk_halted']} {x['risk_reason']}",f"- PF/WR: {x['pf']} / {x['win_rate']}",""]
md += ["## Structural finding","v0.31 has no normal take-profit, stop-loss, basket-profit close, or trend-end position close. TrendEnded resets state only. Therefore PF/WR are undefined until an exit engine is added. Risk-stop is the only programmed closing path."]
(OUT/"REPORT.md").write_text("\n".join(md),encoding="utf-8")
print(json.dumps(results,indent=2))
