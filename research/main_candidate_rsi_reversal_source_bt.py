#!/usr/bin/env python3
"""BAR pre-BT reproducing olivertwigg/XAU-RSI-Reversal-50-EMA-Bot.
Research only. Historical news filter OFF, matching upstream behavior when no FMP key/CSV is supplied.
NOT Raw Tick/Nautilus certification.
"""
from pathlib import Path
import csv,json,math
from datetime import datetime,timedelta

ROOT=Path(__file__).resolve().parents[1]
M5=ROOT/"csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv"
D1=ROOT/"csv/XAUUSD/XAUUSD_D1_2026Q1Q2.csv"
OUT=ROOT/"results/main-candidate-rsi-reversal"; OUT.mkdir(parents=True,exist_ok=True)

RSI_PERIOD=14; RSI_SHORT=63.; RSI_LONG=37.; SL_DOLLARS=10.
COOLDOWN_MIN=40; MAX_LOSSES_PER_DAY=2
BLOCKED={1,2,3,4,5,21,22,23}; EMA_D=50

def parse_ts(s):
 s=s.replace("Z","")
 try:return datetime.fromisoformat(s)
 except:return datetime.strptime(s[:19],"%Y-%m-%d %H:%M:%S")

def load(path):
 rows=[]
 with path.open() as f:
  for r in csv.DictReader(f):
   rows.append({"ts":parse_ts(r["datetime"]),"open":float(r["open"]),"high":float(r["high"]),"low":float(r["low"]),"close":float(r["close"])})
 return rows

def wilder_rsi(cl,n=14):
 out=[None]*len(cl); ag=None; al=None
 gains=[0.0]*len(cl); losses=[0.0]*len(cl)
 for i in range(1,len(cl)):
  d=cl[i]-cl[i-1]; gains[i]=max(d,0); losses[i]=max(-d,0)
 for i in range(1,len(cl)):
  if i<n: continue
  if i==n:
   ag=sum(gains[1:n+1])/n; al=sum(losses[1:n+1])/n
  else:
   ag=(ag*(n-1)+gains[i])/n; al=(al*(n-1)+losses[i])/n
  if al==0: out[i]=100.0
  else:
   rs=ag/al; out[i]=100-100/(1+rs)
 return out

def daily_bias_map(d1,m5):
 # upstream: only fully closed prior daily bars, EMA50, current prior close > prior EMA => long else short
 closes=[]; ema=None; alpha=2/(EMA_D+1); by_date={}
 d_sorted=sorted(d1,key=lambda r:r["ts"])
 for r in d_sorted:
  ema=r["close"] if ema is None else alpha*r["close"]+(1-alpha)*ema
  closes.append((r["ts"].date(),r["close"],ema))
 dates=sorted({r["ts"].date() for r in m5})
 for d in dates:
  prior=[x for x in closes if x[0]<d]
  if len(prior)<EMA_D: by_date[d]=None
  else:
   _,c,e=prior[-1]; by_date[d]="long" if c>e else "short"
 return by_date

def is_fomc_window(ts):
 if ts.weekday()!=2:return False
 for h in (19,20):
  dec=ts.replace(hour=h,minute=0,second=0,microsecond=0)
  if dec-timedelta(minutes=120)<=ts<=dec+timedelta(minutes=30):return True
 return False

def run(rows,bias,start,end):
 cl=[r["close"] for r in rows]; rr=wilder_rsi(cl)
 trades=[]; inpos=None; cooldown=None; loss_day=None; losses_today=0
 for i in range(max(start,RSI_PERIOD+2),min(end,len(rows))):
  r=rows[i]; ts=r["ts"]
  if loss_day!=ts.date():
   loss_day=ts.date(); losses_today=0
  cur=rr[i]; prev=rr[i-1]
  if cur is None or prev is None: continue

  if inpos:
   sig=inpos; outcome=None; exitp=None
   if sig["dir"]=="long":
    if r["low"]<=sig["sl"]:outcome="sl"; exitp=sig["sl"]
    elif cur>=RSI_SHORT:outcome="rsi"; exitp=r["close"]
   else:
    if r["high"]>=sig["sl"]:outcome="sl"; exitp=sig["sl"]
    elif cur<=RSI_LONG:outcome="rsi"; exitp=r["close"]
   if outcome:
    pnl=(exitp-sig["entry"]) if sig["dir"]=="long" else (sig["entry"]-exitp)
    won=pnl>=0
    trades.append({"entry_time":sig["entry_time"].isoformat(),"exit_time":ts.isoformat(),"direction":sig["dir"],"entry":sig["entry"],"exit":exitp,"exit_type":outcome,"pnl_usd_per_oz":pnl})
    inpos=None
    if not won:
     losses_today+=1; cooldown=ts+timedelta(minutes=COOLDOWN_MIN)

  if inpos is not None: continue
  if losses_today>=MAX_LOSSES_PER_DAY: continue
  if cooldown and ts<cooldown: continue
  if ts.hour in BLOCKED: continue
  if is_fomc_window(ts): continue
  # historical news filter intentionally OFF: upstream does this when no FMP key / news CSV
  b=bias.get(ts.date())
  if prev<=RSI_SHORT<cur:
   if b=="long": continue
   inpos={"dir":"short","entry_time":ts,"entry":r["close"],"sl":r["close"]+SL_DOLLARS}
  elif prev>=RSI_LONG>cur:
   if b=="short": continue
   inpos={"dir":"long","entry_time":ts,"entry":r["close"],"sl":r["close"]-SL_DOLLARS}
 return trades

def metrics(trades,days):
 n=len(trades); pnl=[t["pnl_usd_per_oz"] for t in trades]
 wins=[x for x in pnl if x>=0]; losses=[x for x in pnl if x<0]
 pf=sum(wins)/abs(sum(losses)) if losses else (999. if wins else 0.)
 eq=0.; peak=0.; dd=0.; streak=0; maxst=0
 for x in pnl:
  eq+=x; peak=max(peak,eq); dd=min(dd,eq-peak)
  streak=streak+1 if x<0 else 0; maxst=max(maxst,streak)
 return {"N":n,"N_per_day":n/max(days,1),"WR":len(wins)/n if n else 0.,"PF":pf,"net_usd_per_oz":sum(pnl),"max_dd_usd_per_oz":dd,
         "avg_trade_usd_per_oz":sum(pnl)/n if n else 0.,"max_loss_streak":maxst,
         "long_N":sum(t["direction"]=="long" for t in trades),"short_N":sum(t["direction"]=="short" for t in trades)}

rows=load(M5); d1=load(D1); bias=daily_bias_map(d1,rows)
cut=int(len(rows)*.60)
def days(a,b): return max((rows[b]["ts"]-rows[a]["ts"]).total_seconds()/86400,1)
is_tr=run(rows,bias,0,cut); oos_tr=run(rows,bias,cut,len(rows))
summary={
 "provenance":{"upstream":"olivertwigg/XAU-RSI-Reversal-50-EMA-Bot","mode":"BAR reproduction","raw_tick_certified":False,"historical_news_filter":"OFF (upstream no-key behavior)"},
 "parameters":{"RSI":14,"short_cross":63,"long_cross":37,"SL_USD":10,"cooldown_min":40,"max_losses_day":2,"daily_ema_filter":50,"blocked_utc_hours":sorted(BLOCKED)},
 "rows":len(rows),"split_index":cut,
 "IS":metrics(is_tr,days(0,cut-1)),"OOS":metrics(oos_tr,days(cut,len(rows)-1)),
 "warnings":["BAR DATA ONLY; not Raw Tick/Nautilus.","No historical high-impact-news CSV/API key was supplied; news filter is OFF, which matches upstream fallback behavior.","Dataset timestamp timezone must be verified before promotion.","PnL is USD/oz price movement, not account return or lot-sized PnL."]
}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
with (OUT/"trades_oos.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=["entry_time","exit_time","direction","entry","exit","exit_type","pnl_usd_per_oz"]);w.writeheader();w.writerows(oos_tr)
print(json.dumps(summary,indent=2))
