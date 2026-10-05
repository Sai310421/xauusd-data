import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(os.environ.get("XAU_HYBRID_DATA", "csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv"))
OUT = Path("bt_results/xau_hybrid_v033_core_sweep")
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
    "EnableBasketTP":True,"BasketTakeProfitUSD":7.0,
    "EnableRecoveryExit":True,"RecoveryArmLossUSD":15.0,
    "RecoveryExitProfitUSD":0.50,"RecoveryMinHoldSeconds":30,
    "CloseOnTrendEnd":True,
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
    recovery_armed=False; recovery_armed_time=None; basket_
def run_cfg(name, **overrides):
    old={k:P[k] for k in overrides}
    P.update(overrides)
    try:
        out=run(name,cb_mode="off",initial_balance=1000.0)
        out["params"]={k:P[k] for k in overrides}
        return out
    finally:
        P.update(old)

results=[]
idx=0
for adx_trend in [23.0, 28.0]:
    for er_min in [0.40, 0.50]:
        for crt_th in [75, 85]:
            for accel in [False, True]:
                for max_legs in [2, 3]:
                    idx += 1
                    name=f"s{idx:02d}_adx{int(adx_trend)}_er{int(er_min*100)}_crt{crt_th}_acc{int(accel)}_legs{max_legs}"
                    results.append(run_cfg(
                        name,
                        ADX_Trend=adx_trend,
                        TrendERMin=er_min,
                        CRTThreshold=crt_th,
                        RequireMACDAcceleration=accel,
                        MaxLegs=max_legs,
                    ))

def score(x):
    pf=x["basket_profit_factor"]
    pfv=999.0 if pf=="inf" else float(pf)
    passed=(x["return_pct"]>=0.0 and pfv>=1.2 and x["max_equity_dd_pct"]<=10.0)
    return (1 if passed else 0, x["return_pct"], pfv, -x["max_equity_dd_pct"])

ranked=sorted(results,key=score,reverse=True)
top=ranked[:10]
for i,x in enumerate(top,1):
    x["rank"]=i
    pf=x["basket_profit_factor"]
    pfv=999.0 if pf=="inf" else float(pf)
    x["pass_core_gate"]=bool(x["return_pct"]>=0.0 and pfv>=1.2 and x["max_equity_dd_pct"]<=10.0)

(OUT/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame(results).to_csv(OUT/"summary.csv",index=False)
(OUT/"top10.json").write_text(json.dumps(top,ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame(top).to_csv(OUT/"top10.csv",index=False)

md=["# XAUUSD Hybrid EA v0.33 - CB OFF core sweep","",
    "Source: canonical Nautilus Raw XAUUSD Bid/Ask QuoteTicks -> causal M1/M5/M15 diagnostic bars.",
    "Execution remains M1-close approximation; this sweep intentionally disables cashback.",
    "Core gate: Return >= 0%, Basket PF >= 1.2, MaxDD <= 10%.","",
    "## Top 10"]
for x in top:
    md += [
        f"### #{x['rank']} {x['scenario']}",
        f"- Return: {x['return_pct']}%",
        f"- MaxDD: {x['max_equity_dd_pct']}%",
        f"- Basket PF / WR: {x['basket_profit_factor']} / {x['basket_win_rate_pct']}%",
        f"- Orders: {x['orders_opened']} | Basket closes: {x['basket_closes']}",
        f"- Gate: {'PASS' if x['pass_core_gate'] else 'FAIL'}",
        f"- Params: {x['params']}",
        ""
    ]
passes=[x for x in ranked if (x["return_pct"]>=0 and (999.0 if x["basket_profit_factor"]=="inf" else float(x["basket_profit_factor"]))>=1.2 and x["max_equity_dd_pct"]<=10.0)]
md += ["## Gate result",f"- Passing configurations: {len(passes)} / {len(results)}"]
if passes:
    b=passes[0]
    md += [f"- Best pass: {b['scenario']} | Return {b['return_pct']}% | DD {b['max_equity_dd_pct']}% | PF {b['basket_profit_factor']} | WR {b['basket_win_rate_pct']}%"]
else:
    b=ranked[0]
    md += [f"- No full pass. Best candidate: {b['scenario']} | Return {b['return_pct']}% | DD {b['max_equity_dd_pct']}% | PF {b['basket_profit_factor']} | WR {b['basket_win_rate_pct']}%"]
(OUT/"REPORT.md").write_text("\n".join(md),encoding="utf-8")
print(json.dumps({"tested":len(results),"passes":len(passes),"best":ranked[0]},indent=2))
