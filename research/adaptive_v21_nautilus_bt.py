from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median

RESULT_DIR = Path(os.getenv("RESULT_DIR", "results/adaptive-v21"))
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OTE = {"front": 0.669, "core": 0.708, "deep": 0.786}

@dataclass
class Trade:
    side: str
    entry: float
    exit: float
    pnl: float
    mae: float
    mfe: float
    slices: int
    exit_reason: str


def load_quotes() -> list[dict]:
    for p in [Path("data/xauusd_quotes.jsonl"), Path("data/xauusd_ticks.jsonl"), Path("books/xauusd_quotes.jsonl"), Path("books/xauusd_ticks.jsonl")]:
        if p.exists():
            out=[]
            with p.open("r",encoding="utf-8") as f:
                for line in f:
                    if line.strip(): out.append(json.loads(line))
            return out
    raise FileNotFoundError("Raw Bid/Ask quote JSONL not found")


def bid(q): return float(q.get("bid",q.get("b",0)))
def ask(q): return float(q.get("ask",q.get("a",0)))

def robust_step(xs: list[float]) -> float:
    d=[abs(xs[i]-xs[i-1]) for i in range(1,len(xs)) if xs[i]!=xs[i-1]]
    return max(median(d) if d else 0.001, 0.001)

def fib_price(a: float,b: float,r: float,bull: bool) -> float:
    return b-r*(b-a) if bull else b+r*(a-b)


def detect_a0(hist: list[dict]):
    """Causal tick-native A0: liquidity -> sweep/reclaim -> MSS.
    No candles, OHLC, midpoint or future data are used.
    Returns (bullish, anchor_a, anchor_b, invalidation) or None.
    """
    if len(hist)<360: return None
    bids=[bid(q) for q in hist]; asks=[ask(q) for q in hist]
    base_n=240
    recent_start=base_n

    # Bullish candidate: sweep prior BID liquidity low, reclaim, then BID MSS above pre-sweep swing high.
    base_low=min(bids[:base_n]); eps=max(robust_step(bids[:base_n])*2.0,0.01)
    sweep_i=None
    for j in range(recent_start,len(hist)-20):
        if bids[j] < base_low-eps:
            sweep_i=j; break
    if sweep_i is not None:
        reclaim_i=next((j for j in range(sweep_i+1,len(hist)-10) if bids[j] > base_low),None)
        if reclaim_i is not None:
            pre_hi=max(bids[max(0,sweep_i-80):sweep_i])
            mss_i=next((j for j in range(reclaim_i+1,len(hist)) if bids[j] > pre_hi+eps),None)
            if mss_i is not None:
                a=min(bids[sweep_i:mss_i+1]); b=max(bids[sweep_i:mss_i+1])
                if b-a >= 8*eps:
                    return True,a,b,a-eps

    # Bearish candidate: sweep prior ASK liquidity high, reclaim, then ASK MSS below pre-sweep swing low.
    base_high=max(asks[:base_n]); eps=max(robust_step(asks[:base_n])*2.0,0.01)
    sweep_i=None
    for j in range(recent_start,len(hist)-20):
        if asks[j] > base_high+eps:
            sweep_i=j; break
    if sweep_i is not None:
        reclaim_i=next((j for j in range(sweep_i+1,len(hist)-10) if asks[j] < base_high),None)
        if reclaim_i is not None:
            pre_lo=min(asks[max(0,sweep_i-80):sweep_i])
            mss_i=next((j for j in range(reclaim_i+1,len(hist)) if asks[j] < pre_lo-eps),None)
            if mss_i is not None:
                a=max(asks[sweep_i:mss_i+1]); b=min(asks[sweep_i:mss_i+1])
                if a-b >= 8*eps:
                    return False,a,b,a+eps
    return None


def run():
    quotes=load_quotes()
    if len(quotes)<10000: raise RuntimeError(f"Insufficient raw quotes: {len(quotes)}")

    trades=[]
    lookback=360
    horizon=360
    stride=120
    setups=0

    for i in range(lookback,len(quotes)-horizon,stride):
        setup=detect_a0(quotes[i-lookback:i])
        if setup is None: continue
        setups+=1
        bullish,a,b,invalid=setup
        levels=[fib_price(a,b,OTE[k],bullish) for k in ("front","core","deep")]
        path=quotes[i:i+horizon]

        fills=[]
        fill_indices=[]
        for level in levels:
            # Executable-side touch only: long BUY requires ASK <= level; short SELL requires BID >= level.
            j=next((j for j,q in enumerate(path) if (ask(q)<=level if bullish else bid(q)>=level)),None)
            if j is not None:
                px=ask(path[j]) if bullish else bid(path[j])
                fills.append(px); fill_indices.append(j)
        if not fills: continue

        avg_entry=sum(fills)/len(fills)
        risk=avg_entry-invalid if bullish else invalid-avg_entry
        if risk<=0: continue
        target=avg_entry+1.5*risk if bullish else avg_entry-1.5*risk
        start=max(fill_indices)
        exit_px=None; reason="TIME"
        marks=[]
        for q in path[start:]:
            mark=bid(q) if bullish else ask(q)  # executable exit side
            marks.append(mark)
            if bullish:
                if mark<=invalid: exit_px=mark; reason="INVALIDATION"; break
                if mark>=target: exit_px=mark; reason="TP_1.5R"; break
            else:
                if mark>=invalid: exit_px=mark; reason="INVALIDATION"; break
                if mark<=target: exit_px=mark; reason="TP_1.5R"; break
        if exit_px is None:
            q=path[-1]; exit_px=bid(q) if bullish else ask(q)
        direction=1.0 if bullish else -1.0
        pnl=direction*(exit_px-avg_entry)
        excursions=[direction*(m-avg_entry) for m in marks] or [pnl]
        trades.append(Trade("LONG" if bullish else "SHORT",avg_entry,exit_px,pnl,min(excursions),max(excursions),len(fills),reason))

    wins=[t.pnl for t in trades if t.pnl>0]; losses=[-t.pnl for t in trades if t.pnl<0]
    gross_win=sum(wins); gross_loss=sum(losses)
    pf=gross_win/gross_loss if gross_loss>0 else (math.inf if gross_win>0 else 0.0)
    wr=len(wins)/len(trades) if trades else 0.0
    ev=sum(t.pnl for t in trades)/len(trades) if trades else 0.0
    eq=peak=max_dd=0.0
    for t in trades:
        eq+=t.pnl; peak=max(peak,eq); max_dd=max(max_dd,peak-eq)

    summary={
        "verification_level":"A0_REAL_RAW_BID_ASK_SWEEP_MSS_OTE",
        "quotes":len(quotes),"setups":setups,"trades":len(trades),"wr":wr,"pf":pf,
        "ev_price_units":ev,"net_price_units":sum(t.pnl for t in trades),"max_dd_price_units":max_dd,
        "avg_mae":mean([t.mae for t in trades]) if trades else 0.0,
        "avg_mfe":mean([t.mfe for t in trades]) if trades else 0.0,
        "slice_fill_counts":{"1":sum(t.slices==1 for t in trades),"2":sum(t.slices==2 for t in trades),"3":sum(t.slices==3 for t in trades)},
        "exit_counts":{"TP_1.5R":sum(t.exit_reason=="TP_1.5R" for t in trades),"INVALIDATION":sum(t.exit_reason=="INVALIDATION" for t in trades),"TIME":sum(t.exit_reason=="TIME" for t in trades)},
        "ohlc_resample_used":False,"mid_price_used":False,
        "notes":["Tick-native A0: Liquidity -> Sweep/Reclaim -> MSS -> locked OTE 0.669/0.708/0.786.","Long fill Ask / exit Bid; Short fill Bid / exit Ask.","RR target 1.5R; common thesis invalidation beyond sweep."]
    }
    (RESULT_DIR/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (RESULT_DIR/"trades.json").write_text(json.dumps([asdict(t) for t in trades],indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": run()
