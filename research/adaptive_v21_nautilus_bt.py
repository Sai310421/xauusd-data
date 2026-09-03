from __future__ import annotations

import bisect
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

RESULT_DIR = Path(os.getenv("RESULT_DIR", "results/adaptive-v21"))
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OTE = {"front": 0.669, "core": 0.708, "deep": 0.786}
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900}

@dataclass
class Trade:
    tf: str
    side: str
    entry: float
    exit: float
    pnl: float
    mae: float
    mfe: float
    slices: int
    exit_reason: str


def load_quotes():
    for p in [
        Path("data/xauusd_quotes.jsonl"),
        Path("data/xauusd_ticks.jsonl"),
        Path("books/xauusd_quotes.jsonl"),
        Path("books/xauusd_ticks.jsonl"),
    ]:
        if p.exists():
            out = []
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    q = json.loads(line)
                    ts = q["ts"]
                    q["_t"] = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                    out.append(q)
            return out
    raise FileNotFoundError("Raw Bid/Ask quote JSONL not found")


def bid(q): return float(q.get("bid", q.get("b", 0)))
def ask(q): return float(q.get("ask", q.get("a", 0)))


def robust_step(xs):
    d = [abs(xs[i] - xs[i - 1]) for i in range(1, len(xs)) if xs[i] != xs[i - 1]]
    return max(median(d) if d else 0.001, 0.001)


def fib_retrace(impulse_start, impulse_end, r, bull):
    return impulse_end - r * (impulse_end - impulse_start) if bull else impulse_end + r * (impulse_start - impulse_end)


def detect_crt_delivery_time(hist, tf_sec):
    """Raw-tick CRT using real elapsed time, not a fixed tick-count window.

    Context = 8 x TF duration. First 5 x TF establishes CRT range; last 3 x TF is
    sweep/reclaim/delivery search. Bull and bear candidates are both evaluated and
    the most recently confirmed MSS/delivery wins. No OHLC, candles or midpoint.
    """
    if len(hist) < 50:
        return None
    t0, t1 = hist[0]["_t"], hist[-1]["_t"]
    if t1 - t0 < 7.5 * tf_sec:
        return None
    range_end_t = t1 - 3 * tf_sec
    range_end = bisect.bisect_right([q["_t"] for q in hist], range_end_t)
    if range_end < 20 or len(hist) - range_end < 20:
        return None

    B = [bid(q) for q in hist]
    A = [ask(q) for q in hist]
    base_b, base_a = B[:range_end], A[:range_end]
    lo, hi = min(base_b), max(base_a)
    eps = max(robust_step(base_b + base_a) * 3.0, 0.01)
    candidates = []

    # Bullish: sell-side sweep -> reclaim -> opposite-side delivery/MSS.
    for s in range(range_end, len(hist) - 5):
        if B[s] >= lo - eps:
            continue
        reclaim_deadline = hist[s]["_t"] + tf_sec
        reclaim = next((j for j in range(s + 1, len(hist)) if hist[j]["_t"] <= reclaim_deadline and B[j] > lo), None)
        if reclaim is None:
            continue
        pre_start_t = hist[s]["_t"] - 2 * tf_sec
        pre_start = bisect.bisect_left([q["_t"] for q in hist], pre_start_t, 0, s)
        if s - pre_start < 5:
            continue
        internal_hi = max(B[pre_start:s])
        mss = next((j for j in range(reclaim + 1, len(hist)) if B[j] > internal_hi + eps), None)
        if mss is None:
            continue
        sweep_low = min(B[s:mss + 1])
        impulse_end = max(B[reclaim:mss + 1])
        if impulse_end - sweep_low >= 10 * eps:
            candidates.append((mss, True, sweep_low, impulse_end, sweep_low, eps))

    # Bearish: buy-side sweep -> reclaim -> opposite-side delivery/MSS.
    for s in range(range_end, len(hist) - 5):
        if A[s] <= hi + eps:
            continue
        reclaim_deadline = hist[s]["_t"] + tf_sec
        reclaim = next((j for j in range(s + 1, len(hist)) if hist[j]["_t"] <= reclaim_deadline and A[j] < hi), None)
        if reclaim is None:
            continue
        pre_start_t = hist[s]["_t"] - 2 * tf_sec
        pre_start = bisect.bisect_left([q["_t"] for q in hist], pre_start_t, 0, s)
        if s - pre_start < 5:
            continue
        internal_lo = min(A[pre_start:s])
        mss = next((j for j in range(reclaim + 1, len(hist)) if A[j] < internal_lo - eps), None)
        if mss is None:
            continue
        sweep_high = max(A[s:mss + 1])
        impulse_end = min(A[reclaim:mss + 1])
        if sweep_high - impulse_end >= 10 * eps:
            candidates.append((mss, False, sweep_high, impulse_end, sweep_high, eps))

    if not candidates:
        return None
    # No bull-first bias: use the latest confirmed delivery/MSS in elapsed time.
    _, bull, a, b, sweep, eps = max(candidates, key=lambda x: x[0])
    return bull, a, b, sweep, eps


def simulate_setup(path, tf, bull, a, b, sweep, eps):
    levels = [fib_retrace(a, b, OTE[k], bull) for k in ("front", "core", "deep")]
    invalid = sweep - eps if bull else sweep + eps
    filled = [False, False, False]
    fills = []
    marks = []
    exitpx = None
    reason = "TIME"

    for q in path:
        # Causal slice fills. Long BUY can fill only at Ask; short SELL only at Bid.
        for n, level in enumerate(levels):
            if filled[n]:
                continue
            touched = ask(q) <= level if bull else bid(q) >= level
            if touched:
                filled[n] = True
                fills.append(ask(q) if bull else bid(q))

        if not fills:
            continue

        entry = sum(fills) / len(fills)
        risk = entry - invalid if bull else invalid - entry
        if risk <= 0:
            return None
        target = entry + 1.5 * risk if bull else entry - 1.5 * risk
        mark = bid(q) if bull else ask(q)
        marks.append(mark)

        if bull and mark <= invalid:
            exitpx, reason = mark, "INVALIDATION"
            break
        if (not bull) and mark >= invalid:
            exitpx, reason = mark, "INVALIDATION"
            break
        if bull and mark >= target:
            exitpx, reason = mark, "TP_1.5R"
            break
        if (not bull) and mark <= target:
            exitpx, reason = mark, "TP_1.5R"
            break

    if not fills:
        return None
    entry = sum(fills) / len(fills)
    if exitpx is None:
        exitpx = bid(path[-1]) if bull else ask(path[-1])
    d = 1.0 if bull else -1.0
    pnl = d * (exitpx - entry)
    ex = [d * (x - entry) for x in marks] or [pnl]
    return Trade(tf, "LONG" if bull else "SHORT", entry, exitpx, pnl, min(ex), max(ex), len(fills), reason)


def run():
    Q = load_quotes()
    if len(Q) < 10000:
        raise RuntimeError("Insufficient raw quotes")
    times = [q["_t"] for q in Q]
    trades = []
    setups_by_tf = {tf: 0 for tf in TF_SECONDS}

    for tf, tf_sec in TF_SECONDS.items():
        context_sec = 8 * tf_sec
        horizon_sec = 6 * tf_sec
        scan_sec = tf_sec  # one independent state decision per native TF duration
        next_scan = times[0] + context_sec
        last_end = times[-1] - horizon_sec

        while next_scan <= last_end:
            i = bisect.bisect_left(times, next_scan)
            h0 = bisect.bisect_left(times, next_scan - context_sec, 0, i)
            hist = Q[h0:i]
            z = detect_crt_delivery_time(hist, tf_sec)
            if z is not None:
                bull, a, b, sweep, eps = z
                setups_by_tf[tf] += 1
                j = bisect.bisect_right(times, next_scan + horizon_sec, i)
                if j > i:
                    t = simulate_setup(Q[i:j], tf, bull, a, b, sweep, eps)
                    if t is not None:
                        trades.append(t)
            next_scan += scan_sec

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [-t.pnl for t in trades if t.pnl < 0]
    gw, gl = sum(wins), sum(losses)
    pf = gw / gl if gl else (math.inf if gw else 0.0)
    wr = len(wins) / len(trades) if trades else 0.0
    ev = sum(t.pnl for t in trades) / len(trades) if trades else 0.0
    eq = peak = dd = 0.0
    for t in trades:
        eq += t.pnl
        peak = max(peak, eq)
        dd = max(dd, peak - eq)

    per_tf = {}
    for tf in TF_SECONDS:
        tt = [t for t in trades if t.tf == tf]
        ww = [t.pnl for t in tt if t.pnl > 0]
        ll = [-t.pnl for t in tt if t.pnl < 0]
        per_tf[tf] = {
            "setups": setups_by_tf[tf],
            "trades": len(tt),
            "wr": len(ww) / len(tt) if tt else 0.0,
            "pf": sum(ww) / sum(ll) if ll else (math.inf if ww else 0.0),
            "net_price_units": sum(t.pnl for t in tt),
        }

    S = {
        "verification_level": "CRT_RAW_BID_ASK_TIME_AXIS_M1_M5_M15",
        "quotes": len(Q),
        "time_axis": "elapsed-time raw quote states; no OHLC/resample",
        "tf_seconds": TF_SECONDS,
        "setups": sum(setups_by_tf.values()),
        "setups_by_tf": setups_by_tf,
        "trades": len(trades),
        "wr": wr,
        "pf": pf,
        "ev_price_units": ev,
        "net_price_units": sum(t.pnl for t in trades),
        "max_dd_price_units": dd,
        "avg_mae": mean([t.mae for t in trades]) if trades else 0.0,
        "avg_mfe": mean([t.mfe for t in trades]) if trades else 0.0,
        "per_tf": per_tf,
        "slice_fill_counts": {str(n): sum(t.slices == n for t in trades) for n in (1, 2, 3)},
        "exit_counts": {r: sum(t.exit_reason == r for t in trades) for r in ("TP_1.5R", "INVALIDATION", "TIME")},
        "ohlc_resample_used": False,
        "mid_price_used": False,
        "future_fill_management_bias_removed": True,
        "notes": [
            "M1/M5/M15 are independent elapsed-time raw-tick states, not candles.",
            "CRT range -> sweep -> reclaim -> opposite delivery/MSS -> OTE 0.669/0.708/0.786.",
            "Latest MSS wins if bull and bear candidates coexist; no bull-first bias.",
            "Slice fill and exit are processed sequentially; no waiting for future slices before managing risk.",
            "Long Ask entry/Bid exit; Short Bid entry/Ask exit.",
        ],
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    (RESULT_DIR / "trades.json").write_text(json.dumps([asdict(t) for t in trades], indent=2), encoding="utf-8")
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    run()
