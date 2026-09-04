from __future__ import annotations

"""Hybrid Rule Raw Bid/Ask execution driver for AMOS G-3.

Signals are calculated causally from M1 bars built directly from ordered Dukascopy
quote ticks. Orders are never filled from OHLC bars: entries/reversals/terminal
liquidation use the first subsequent executable raw Ask/Bid quote. This is a new,
documented Rule-only harness based on the recorded EMA9/EMA21, EMA slope, RSI14,
3-bar momentum, body/ATR14 feature set and min_rule_score=0.68; it is not a claim
that a missing historical implementation has been recovered.
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import lzma
import math
import struct
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

REC = struct.Struct(">3i2f")
HOSTS = ("https://datafeed.dukascopy.com/datafeed", "https://www.dukascopy.com/datafeed")
HEADERS = {"User-Agent": "amos-hybrid-rule-rawtick/1.0", "Accept": "*/*"}
DEFAULT_END = dt.datetime(2026, 8, 20, 0, 0, tzinfo=dt.timezone.utc)
WINDOW_SECONDS = {"1h": 3600, "1d": 86400, "5d": 5 * 86400, "30d": 30 * 86400, "90d": 90 * 86400}
EMA_FAST, EMA_SLOW, RSI_PERIOD, ATR_PERIOD, MOMENTUM_LOOKBACK = 9, 21, 14, 14, 3
MIN_RULE_SCORE = 0.68
W_TREND, W_MOMENTUM, W_BODY = 0.40, 0.35, 0.25

@dataclass
class Trade:
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl: float
    entry_score: float
    exit_reason: str


def _hour_jobs(start: dt.datetime, end: dt.datetime):
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        yield cur
        cur += dt.timedelta(hours=1)


def fetch_hour(origin: dt.datetime):
    day = origin.date()
    rel = f"XAUUSD/{day.year}/{day.month-1:02d}/{day.day:02d}/{origin.hour:02d}h_ticks.bi5"
    for host in HOSTS:
        try:
            req = urllib.request.Request(f"{host}/{rel}", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                raw = r.read()
            dec = lzma.decompress(raw)
            rows = []
            for i in range(0, len(dec)-REC.size+1, REC.size):
                ms, ask_i, bid_i, ask_v, bid_v = REC.unpack_from(dec, i)
                ask, bid = ask_i/1000.0, bid_i/1000.0
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue
                rows.append((origin+dt.timedelta(milliseconds=ms), bid, ask, float(bid_v), float(ask_v)))
            return rows
        except Exception:
            continue
    return []


def load_ticks(start: dt.datetime, end: dt.datetime, workers: int) -> pd.DataFrame:
    rows = []
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for part in ex.map(fetch_hour, list(_hour_jobs(start, end))):
            rows.extend(part)
    rows = [r for r in rows if start <= r[0] < end]
    if not rows:
        raise SystemExit("BLOCKED_NO_RAW_QUOTES")
    rows.sort(key=lambda x: x[0])
    df = pd.DataFrame(rows, columns=["datetime","bid","ask","bid_size","ask_size"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates("datetime", keep="last").reset_index(drop=True)
    finite = df[["bid","ask","bid_size","ask_size"]].applymap(lambda x: math.isfinite(float(x))).all(axis=1)
    if not finite.all() or (df.bid <= 0).any() or (df.ask <= 0).any() or (df.ask < df.bid).any() or not df.datetime.is_monotonic_increasing:
        raise SystemExit("FAIL_RAW_QUOTE_QUALITY")
    df["mid"] = (df.bid+df.ask)/2.0
    return df


def raw_ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    x = ticks[["datetime","mid"]].copy()
    x["minute"] = x.datetime.dt.floor("min")
    g = x.groupby("minute", sort=True)
    return g.agg(open=("mid","first"), high=("mid","max"), low=("mid","min"), close=("mid","last"), tick_count=("mid","size")).reset_index().rename(columns={"minute":"datetime"})


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain/loss.replace(0, float("nan"))
    out = 100.0-100.0/(1.0+rs)
    return out.where(loss != 0, 100.0)


def make_signals(bars: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    prev = x.close.shift(1)
    tr = pd.concat([(x.high-x.low).abs(), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    x["ema_fast"] = x.close.ewm(span=EMA_FAST, adjust=False).mean()
    x["ema_slow"] = x.close.ewm(span=EMA_SLOW, adjust=False).mean()
    x["ema_slope"] = x.ema_fast.diff()
    x["rsi"] = _rsi(x.close, RSI_PERIOD)
    x["momentum"] = x.close-x.close.shift(MOMENTUM_LOOKBACK)
    x["body"] = (x.close-x.open).abs()
    x["direction"], x["score"] = "", 0.0
    for i, r in x.iterrows():
        vals = [r.atr,r.ema_fast,r.ema_slow,r.ema_slope,r.rsi,r.momentum,r.body]
        if not all(pd.notna(v) and math.isfinite(float(v)) for v in vals) or r.atr <= 0:
            continue
        up = r.ema_fast > r.ema_slow and r.ema_slope > 0 and r.momentum > 0 and r.rsi > 50
        dn = r.ema_fast < r.ema_slow and r.ema_slope < 0 and r.momentum < 0 and r.rsi < 50
        if not (up or dn):
            continue
        score = max(0.0, min(1.0, W_TREND*min(abs(r.ema_slope)/r.atr,1.0) + W_MOMENTUM*min(abs(r.rsi-50.0)/50.0,1.0) + W_BODY*min(r.body/r.atr,1.0)))
        if score >= MIN_RULE_SCORE:
            x.at[i,"direction"] = "BUY" if up else "SELL"
            x.at[i,"score"] = score
    return x


def run_execution(ticks: pd.DataFrame, signals: pd.DataFrame) -> list[Trade]:
    sig = signals[signals.direction != ""][["datetime","direction","score"]].copy()
    if sig.empty:
        return []
    sig["execute_after"] = sig.datetime+pd.Timedelta(minutes=1)
    events = list(sig.itertuples(index=False))
    trades, pos, eidx = [], None, 0
    for q in ticks.itertuples(index=False):
        while eidx < len(events) and q.datetime >= events[eidx].execute_after:
            ev = events[eidx]; eidx += 1
            side = ev.direction
            if pos is not None and pos["side"] == side:
                continue
            if pos is not None:
                exit_px = float(q.bid if pos["side"] == "BUY" else q.ask)
                pnl = exit_px-pos["entry_price"] if pos["side"] == "BUY" else pos["entry_price"]-exit_px
                trades.append(Trade(pos["side"],pos["entry_time"].isoformat(),q.datetime.isoformat(),pos["entry_price"],exit_px,pnl,pos["score"],"REVERSE"))
            pos = {"side":side,"entry_time":q.datetime,"entry_price":float(q.ask if side == "BUY" else q.bid),"score":float(ev.score)}
    if pos is not None:
        q = ticks.iloc[-1]
        exit_px = float(q.bid if pos["side"] == "BUY" else q.ask)
        pnl = exit_px-pos["entry_price"] if pos["side"] == "BUY" else pos["entry_price"]-exit_px
        trades.append(Trade(pos["side"],pos["entry_time"].isoformat(),q.datetime.isoformat(),pos["entry_price"],exit_px,pnl,pos["score"],"END_OF_WINDOW"))
    return trades


def metrics(trades: list[Trade]) -> dict:
    pnl = [float(t.pnl) for t in trades]
    wins, losses = [x for x in pnl if x > 0], [x for x in pnl if x < 0]
    if losses and sum(losses) != 0:
        pf = sum(wins)/abs(sum(losses))
    elif wins:
        pf = None  # mathematically infinite; keep JSON strict and explicit
    else:
        pf = 0.0
    wr = 100.0*len(wins)/len(pnl) if pnl else 0.0
    eq = peak = maxdd = 0.0
    for x in pnl:
        eq += x; peak = max(peak, eq); maxdd = max(maxdd, peak-eq)
    return {"N":len(pnl),"WR_pct":wr,"PF":pf,"PF_infinite":bool(wins and not losses),"NetProfit_price_units":sum(pnl),"MaxClosedDD_price_units":maxdd}


def main():
    import nautilus_trader
    ap = argparse.ArgumentParser(); ap.add_argument("--window", choices=WINDOW_SECONDS, required=True); ap.add_argument("--end", default=DEFAULT_END.isoformat()); ap.add_argument("--workers", type=int, default=32); args = ap.parse_args()
    end = dt.datetime.fromisoformat(args.end.replace("Z","+00:00"))
    if end.tzinfo is None: end = end.replace(tzinfo=dt.timezone.utc)
    end = end.astimezone(dt.timezone.utc); start = end-dt.timedelta(seconds=WINDOW_SECONDS[args.window])
    ticks = load_ticks(start,end,args.workers); bars = raw_ticks_to_m1(ticks); signals = make_signals(bars); trades = run_execution(ticks,signals)
    out = Path("results/hybrid-rule")/args.window; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([asdict(t) for t in trades]).to_csv(out/"trades.csv",index=False)
    report = {"status":"PASS","verification_level":"RAW_BIDASK_EXECUTION_HARNESS","provenance":"dukascopy_raw_bidask","nautilus_version":getattr(nautilus_trader,"__version__","unknown"),"window":args.window,"start":start.isoformat(),"end":end.isoformat(),"raw_quotes":int(len(ticks)),"m1_signal_bars_from_raw_ticks":int(len(bars)),"ohlc_source_input_used":False,"fills_use_raw_bidask":True,"rule_spec":{"ema_fast":EMA_FAST,"ema_slow":EMA_SLOW,"rsi":RSI_PERIOD,"atr":ATR_PERIOD,"momentum":MOMENTUM_LOOKBACK,"min_rule_score":MIN_RULE_SCORE,"weights":[W_TREND,W_MOMENTUM,W_BODY]},"metrics":metrics(trades),"limitations":["new documented Rule-only implementation, not recovered historical source","native spread included through executable Bid/Ask; commission/slippage/latency not yet modeled","M1 signal bars are formed causally from raw quotes only; fills remain subsequent raw quote events"]}
    text = json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)
    (out/"summary.json").write_text(text,encoding="utf-8"); print(text)

if __name__ == "__main__": main()
