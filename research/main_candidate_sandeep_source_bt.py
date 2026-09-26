#!/usr/bin/env python3
"""Source-faithful BAR pre-BT for Sandyyy123/xauusd-scalper-research.
Research only. XAUUSD M5 BAR data; NOT Raw Tick/Nautilus certification.
Upstream remains READ ONLY. This file independently reproduces the documented Python strategy.
"""
from pathlib import Path
import csv, json, math
from datetime import datetime, time

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv"
OUT=ROOT/"results/main-candidate-sandeep"
OUT.mkdir(parents=True,exist_ok=True)

def load():
    rows=[]
    with SRC.open() as f:
        for r in csv.DictReader(f):
            ts=r["datetime"]
            rows.append({"ts":ts,"open":float(r["open"]),"high":float(r["high"]),"low":float(r["low"]),"close":float(r["close"])})
    return rows

def hour_of(ts):
    s=ts.replace("Z","")
    try: return datetime.fromisoformat(s).time()
    except Exception:
        return datetime.strptime(s[:19],"%Y-%m-%d %H:%M:%S").time()

def in_session(ts):
    t=hour_of(ts)
    return time(7,0)<=t<=time(11,0) or time(13,0)<=t<=time(17,0)

def ema(vals,n):
    a=2/(n+1); out=[]; x=None
    for v in vals:
        x=v if x is None else a*v+(1-a)*x
        out.append(x)
    return out

def roll_mean(vals,n):
    out=[None]*len(vals); s=0.0; q=[]
    for i,v in enumerate(vals):
        if v is None: q.append(None)
        else:
            q.append(v); s+=v
        if len(q)>n:
            z=q.pop(0)
            if z is not None:s-=z
        if len(q)==n and all(z is not None for z in q): out[i]=s/n
    return out

def rsi(cl,n=7):
    gains=[None]*len(cl); losses=[None]*len(cl)
    for i in range(1,len(cl)):
        d=cl[i]-cl[i-1]; gains[i]=max(d,0); losses[i]=max(-d,0)
    ag=roll_mean(gains,n); al=roll_mean(losses,n); out=[None]*len(cl)
    for i in range(len(cl)):
        if ag[i] is None or al[i] is None: continue
        if al[i]==0: out[i]=100.0
        else:
            rs=ag[i]/al[i]; out[i]=100-100/(1+rs)
    return out

def atr(hi,lo,cl,n=14):
    tr=[]
    for i in range(len(cl)):
        if i==0: tr.append(hi[i]-lo[i])
        else: tr.append(max(hi[i]-lo[i],abs(hi[i]-cl[i-1]),abs(lo[i]-cl[i-1])))
    return roll_mean(tr,n)

def stochastic(hi,lo,cl,kp=5,dp=3):
    k=[None]*len(cl)
    for i in range(kp-1,len(cl)):
        ll=min(lo[i-kp+1:i+1]); hh=max(hi[i-kp+1:i+1])
        k[i]=100*(cl[i]-ll)/(hh-ll+1e-10)
    d=roll_mean(k,dp)
    return k,d

def indicators(rows):
    cl=[r["close"] for r in rows]; hi=[r["high"] for r in rows]; lo=[r["low"] for r in rows]
    e20=ema(cl,20); e50=ema(cl,50); rr=rsi(cl,7); kk,dd=stochastic(hi,lo,cl,5,3)
    aa=atr(hi,lo,cl,14); aa50=roll_mean(aa,50)
    for i,r in enumerate(rows):
        vol_ok=aa[i] is not None and aa50[i] is not None and aa[i] > aa50[i]*0.8
        valid=rr[i] is not None and kk[i] is not None and dd[i] is not None and vol_ok and in_session(r["ts"])
        r["atr"]=aa[i]; r["long"]=bool(valid and e20[i]>e50[i] and rr[i]<30 and kk[i]<20 and kk[i]>dd[i])
        r["short"]=bool(valid and e20[i]<e50[i] and rr[i]>70 and kk[i]>80 and kk[i]<dd[i])
    return rows

def simulate(rows,start,end,sl_mult=1.0,tp_mult=1.5,commission=0.0003,slippage=0.0001):
    trades=[]
    # source-like independent forward trade simulation, cap 60 future bars
    for i in range(start,min(end,len(rows)-1)):
        r=rows[i]
        direction="long" if r["long"] else ("short" if r["short"] else None)
        if direction is None or r["atr"] is None: continue
        entry=r["close"]*(1+slippage if direction=="long" else 1-slippage)
        sl=r["close"]-sl_mult*r["atr"] if direction=="long" else r["close"]+sl_mult*r["atr"]
        tp=r["close"]+tp_mult*r["atr"] if direction=="long" else r["close"]-tp_mult*r["atr"]
        out="open"; exitp=None; exit_i=None
        for j in range(i+1,min(i+61,end,len(rows))):
            b=rows[j]
            # upstream checks SL before TP: conservative same-bar ordering
            if direction=="long":
                if b["low"]<=sl: out="sl"; exitp=sl; exit_i=j; break
                if b["high"]>=tp: out="tp"; exitp=tp; exit_i=j; break
            else:
                if b["high"]>=sl: out="sl"; exitp=sl; exit_i=j; break
                if b["low"]<=tp: out="tp"; exitp=tp; exit_i=j; break
        if exitp is None:
            exit_i=min(i+60,end-1,len(rows)-1); exitp=rows[exit_i]["close"]
        sign=1 if direction=="long" else -1
        pnl=sign*(exitp-entry)/entry-commission
        trades.append({"entry_i":i,"exit_i":exit_i,"entry_time":r["ts"],"exit_time":rows[exit_i]["ts"],"direction":direction,"outcome":out,"pnl_pct":pnl})
    return trades

def metrics(trades,days):
    n=len(trades); wins=[t["pnl_pct"] for t in trades if t["pnl_pct"]>0]; losses=[t["pnl_pct"] for t in trades if t["pnl_pct"]<0]
    pf=sum(wins)/abs(sum(losses)) if losses else (999.0 if wins else 0.0)
    eq=1.; peak=1.; mdd=0.
    # event-order realized PnL approximation
    for t in sorted(trades,key=lambda x:x["exit_i"]):
        eq*=1+t["pnl_pct"]; peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
    return {"N":n,"N_per_day":n/max(days,1),"WR":len(wins)/n if n else 0.0,"PF":pf,"compound_return":eq-1,"max_dd_realized_order":mdd,
            "avg_trade":sum(t["pnl_pct"] for t in trades)/n if n else 0.0,
            "long_N":sum(t["direction"]=="long" for t in trades),"short_N":sum(t["direction"]=="short" for t in trades)}

rows=indicators(load())
cut=int(len(rows)*0.60)
# approximate calendar duration from timestamps
def days_between(a,b):
    def p(s):
        s=s.replace("Z","")
        try:return datetime.fromisoformat(s)
        except:return datetime.strptime(s[:19],"%Y-%m-%d %H:%M:%S")
    return max((p(rows[b]["ts"])-p(rows[a]["ts"])).total_seconds()/86400,1)

# Use fixed upstream default parameters; no OOS tuning.
is_tr=simulate(rows,0,cut)
oos_tr=simulate(rows,cut,len(rows))
summary={
 "provenance":{"upstream":"Sandyyy123/xauusd-scalper-research","mode":"Python strategy source-faithful BAR reproduction","raw_tick_certified":False},
 "parameters":{"EMA_fast":20,"EMA_slow":50,"RSI":7,"RSI_OS":30,"RSI_OB":70,"Stoch_K":5,"Stoch_D":3,"ATR":14,"SL_ATR":1.0,"TP_ATR":1.5,"commission":0.0003,"slippage":0.0001},
 "rows":len(rows),"split_index":cut,
 "IS":metrics(is_tr,days_between(0,cut-1)),
 "OOS":metrics(oos_tr,days_between(cut,len(rows)-1)),
 "warnings":[
  "BAR DATA ONLY; not Raw Tick/Nautilus certification.",
  "Session hours are interpreted directly from dataset timestamps; timezone provenance must be verified before promotion.",
  "The upstream Python backtester uses forward-bar simulation and its concurrency accounting is not a true event-driven portfolio engine.",
  "MQL5 upstream implementation is not exact Python parity: the Python rolling ATR volatility threshold is absent in the EA."
 ]
}
with (OUT/"summary.json").open("w") as f: json.dump(summary,f,indent=2)
with (OUT/"trades_oos.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["entry_time","exit_time","direction","outcome","pnl_pct"])
    w.writeheader()
    for t in oos_tr:w.writerow({k:t[k] for k in w.fieldnames})
print(json.dumps(summary,indent=2))
