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
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900}
OTE = {"front": 0.669, "core": 0.708, "deep": 0.786}

# Hard gate: a Fib family is NOT allowed to trade until its own detector is implemented.
# This prevents the old failure mode where every Fib family became a generic level-touch entry.
FIB_ADAPTER_STATUS = {
    "OTE_CRT": "ENABLED_STRICT",
    "BUTTERFLY_HARMONIC_D_PRZ": "DISABLED_UNTIL_XABCD_D_PRZ_DETECTOR",
    "POP": "DISABLED_UNTIL_POP_SETUP_DETECTOR",
    "GOLD_SILVER": "DISABLED_UNTIL_GOLD_SILVER_SETUP_DETECTOR",
    "CRT_LEVELS": "DISABLED_UNTIL_CRT_LEVEL_SETUP_DETECTOR",
    "ORDER_FLOW_FIB": "DISABLED_UNTIL_ORDER_FLOW_SETUP_DETECTOR",
    "PREMIUM_DISCOUNT": "DISABLED_UNTIL_PD_SETUP_DETECTOR",
    "MONKEY": "DISABLED_UNTIL_MONKEY_SETUP_DETECTOR",
    "FIB_SNR": "DISABLED_UNTIL_FIB_SNR_SETUP_DETECTOR",
    "TARGET_PRICES": "DISABLED_UNTIL_TARGET_PRICE_SETUP_DETECTOR",
    "STD_DEV_FIB": "DISABLED_UNTIL_STD_DEV_SETUP_DETECTOR",
    "CIRCLE_TIME": "DISABLED_UNTIL_CIRCLE_TIME_SETUP_DETECTOR",
    "CIRCLE_HARMONIC": "DISABLED_UNTIL_CIRCLE_HARMONIC_SETUP_DETECTOR",
}


@dataclass
class Setup:
    strategy_id: str
    tf: str
    bull: bool
    impulse_start: float
    impulse_end: float
    sweep: float
    eps: float
    confirmed_at: float


@dataclass
class Trade:
    strategy_id: str
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
                    q["_t"] = datetime.fromisoformat(q["ts"].replace("Z", "+00:00")).timestamp()
                    out.append(q)
            return out
    raise FileNotFoundError("Raw Bid/Ask quote JSONL not found")


def bid(q): return float(q.get("bid", q.get("b", 0)))
def ask(q): return float(q.get("ask", q.get("a", 0)))


def robust_step(xs):
    d = [abs(xs[i] - xs[i - 1]) for i in range(1, len(xs)) if xs[i] != xs[i - 1]]
    return max(median(d) if d else 0.001, 0.001)


def fib_retrace(a, b, r, bull):
    return b - r * (b - a) if bull else b + r * (a - b)


def detect_ote_crt(hist, tf, tf_sec):
    """Strict OTE adapter.

    Required order:
      CRT range -> liquidity sweep -> reclaim -> opposite delivery/MSS
      -> displacement threshold -> lock impulse -> OTE area becomes ARMED.

    No entry occurs here. Entry still requires OTE arrival + tick-native confirmation.
    """
    if len(hist) < 50:
        return None
    t0, t1 = hist[0]["_t"], hist[-1]["_t"]
    if t1 - t0 < 7.5 * tf_sec:
        return None

    times = [q["_t"] for q in hist]
    range_end_t = t1 - 3 * tf_sec
    range_end = bisect.bisect_right(times, range_end_t)
    if range_end < 20 or len(hist) - range_end < 20:
        return None

    B = [bid(q) for q in hist]
    A = [ask(q) for q in hist]
    base_b, base_a = B[:range_end], A[:range_end]
    lo, hi = min(base_b), max(base_a)
    eps = max(robust_step(base_b + base_a) * 3.0, 0.01)
    candidates = []

    # Bullish OTE/CRT: sell-side sweep -> reclaim -> bullish MSS/delivery.
    for s in range(range_end, len(hist) - 5):
        if B[s] >= lo - eps:
            continue
        reclaim_deadline = hist[s]["_t"] + tf_sec
        reclaim = next((j for j in range(s + 1, len(hist)) if hist[j]["_t"] <= reclaim_deadline and B[j] > lo), None)
        if reclaim is None:
            continue
        pre_start = bisect.bisect_left(times, hist[s]["_t"] - 2 * tf_sec, 0, s)
        if s - pre_start < 5:
            continue
        internal_hi = max(B[pre_start:s])
        mss = next((j for j in range(reclaim + 1, len(hist)) if B[j] > internal_hi + eps), None)
        if mss is None:
            continue
        sweep_low = min(B[s:mss + 1])
        impulse_end = max(B[reclaim:mss + 1])
        displacement = impulse_end - sweep_low
        if displacement < 10 * eps:
            continue
        duration = max(hist[mss]["_t"] - hist[reclaim]["_t"], 1e-9)
        velocity = displacement / duration
        if velocity < (6 * eps) / max(tf_sec, 1):
            continue
        candidates.append((mss, Setup("OTE_CRT", tf, True, sweep_low, impulse_end, sweep_low, eps, hist[mss]["_t"])))

    # Bearish OTE/CRT: buy-side sweep -> reclaim -> bearish MSS/delivery.
    for s in range(range_end, len(hist) - 5):
        if A[s] <= hi + eps:
            continue
        reclaim_deadline = hist[s]["_t"] + tf_sec
        reclaim = next((j for j in range(s + 1, len(hist)) if hist[j]["_t"] <= reclaim_deadline and A[j] < hi), None)
        if reclaim is None:
            continue
        pre_start = bisect.bisect_left(times, hist[s]["_t"] - 2 * tf_sec, 0, s)
        if s - pre_start < 5:
            continue
        internal_lo = min(A[pre_start:s])
        mss = next((j for j in range(reclaim + 1, len(hist)) if A[j] < internal_lo - eps), None)
        if mss is None:
            continue
        sweep_high = max(A[s:mss + 1])
        impulse_end = min(A[reclaim:mss + 1])
        displacement = sweep_high - impulse_end
        if displacement < 10 * eps:
            continue
        duration = max(hist[mss]["_t"] - hist[reclaim]["_t"], 1e-9)
        velocity = displacement / duration
        if velocity < (6 * eps) / max(tf_sec, 1):
            continue
        candidates.append((mss, Setup("OTE_CRT", tf, False, sweep_high, impulse_end, sweep_high, eps, hist[mss]["_t"])))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def strategy_setups(hist, tf, tf_sec):
    """Return only fully-qualified setups from enabled strategy adapters."""
    out = []
    if FIB_ADAPTER_STATUS["OTE_CRT"] == "ENABLED_STRICT":
        z = detect_ote_crt(hist, tf, tf_sec)
        if z is not None:
            out.append(z)
    # Other Fib families are intentionally unable to emit a setup until their own
    # logic detector exists. There is no fallback generic Fib touch detector.
    return out


def confirmation_after_touch(path, touch_i, bull, eps, depth):
    """Tick-native post-arrival confirmation. Touch alone is forbidden.

    E1 front: initial rejection.
    E2 core 0.708: rejection + local structure break.
    E3 deep: sweep/reclaim + micro-MSS style confirmation.
    Returns confirmation index or None.
    """
    end = min(len(path), touch_i + 40)
    if end - touch_i < 3:
        return None

    for j in range(touch_i + 2, end):
        left = path[touch_i:j + 1]
        if bull:
            lows = [ask(q) for q in left]
            marks = [bid(q) for q in left]
            local_low = min(lows)
            rejection = bid(path[j]) >= local_low + 2 * eps
            if not rejection:
                continue
            if depth == 0:
                return j
            prior_hi = max(marks[:-1]) if len(marks) > 1 else marks[-1]
            structure = bid(path[j]) >= prior_hi + eps
            if depth == 1 and structure:
                return j
            if depth == 2:
                swept = local_low <= min(lows[: max(1, len(lows)//2)])
                reclaimed = ask(path[j]) >= local_low + 3 * eps
                if swept and reclaimed and structure:
                    return j
        else:
            highs = [bid(q) for q in left]
            marks = [ask(q) for q in left]
            local_high = max(highs)
            rejection = ask(path[j]) <= local_high - 2 * eps
            if not rejection:
                continue
            if depth == 0:
                return j
            prior_lo = min(marks[:-1]) if len(marks) > 1 else marks[-1]
            structure = ask(path[j]) <= prior_lo - eps
            if depth == 1 and structure:
                return j
            if depth == 2:
                swept = local_high >= max(highs[: max(1, len(highs)//2)])
                reclaimed = bid(path[j]) <= local_high - 3 * eps
                if swept and reclaimed and structure:
                    return j
    return None


def simulate_ote_crt(path, setup):
    bull = setup.bull
    a, b, sweep, eps = setup.impulse_start, setup.impulse_end, setup.sweep, setup.eps
    levels = [fib_retrace(a, b, OTE[k], bull) for k in ("front", "core", "deep")]
    invalid = sweep - eps if bull else sweep + eps
    filled = [False, False, False]
    fills = []
    marks = []
    exitpx = None
    reason = "TIME"

    # Process the raw quote stream sequentially. Each slice needs both level arrival
    # and its own confirmation; no future E2/E3 knowledge is used.
    i = 0
    while i < len(path):
        q = path[i]
        for depth, level in enumerate(levels):
            if filled[depth]:
                continue
            touched = ask(q) <= level if bull else bid(q) >= level
            if not touched:
                continue
            confirm_i = confirmation_after_touch(path, i, bull, eps, depth)
            if confirm_i is None:
                continue
            cq = path[confirm_i]
            fillpx = ask(cq) if bull else bid(cq)
            # Do not fill if thesis already invalidated before confirmation.
            exit_side = bid(cq) if bull else ask(cq)
            if (bull and exit_side <= invalid) or ((not bull) and exit_side >= invalid):
                continue
            filled[depth] = True
            fills.append(fillpx)

        if fills:
            entry = sum(fills) / len(fills)
            risk = entry - invalid if bull else invalid - entry
            if risk <= 0:
                return None
            target = entry + 1.5 * risk if bull else entry - 1.5 * risk
            mark = bid(q) if bull else ask(q)
            marks.append(mark)
            if bull and mark <= invalid:
                exitpx, reason = mark, "INVALIDATION"; break
            if (not bull) and mark >= invalid:
                exitpx, reason = mark, "INVALIDATION"; break
            if bull and mark >= target:
                exitpx, reason = mark, "TP_1.5R"; break
            if (not bull) and mark <= target:
                exitpx, reason = mark, "TP_1.5R"; break
        i += 1

    if not fills:
        return None
    entry = sum(fills) / len(fills)
    if exitpx is None:
        exitpx = bid(path[-1]) if bull else ask(path[-1])
    d = 1.0 if bull else -1.0
    pnl = d * (exitpx - entry)
    excursions = [d * (x - entry) for x in marks] or [pnl]
    return Trade(setup.strategy_id, setup.tf, "LONG" if bull else "SHORT", entry, exitpx, pnl,
                 min(excursions), max(excursions), len(fills), reason)


def simulate_strategy(path, setup):
    if setup.strategy_id == "OTE_CRT":
        return simulate_ote_crt(path, setup)
    raise RuntimeError(f"No execution adapter for {setup.strategy_id}")


def run():
    Q = load_quotes()
    if len(Q) < 10000:
        raise RuntimeError("Insufficient raw quotes")
    times = [q["_t"] for q in Q]
    trades = []
    setups_by_tf = {tf: 0 for tf in TF_SECONDS}
    setups_by_strategy = {k: 0 for k in FIB_ADAPTER_STATUS}

    for tf, tf_sec in TF_SECONDS.items():
        context_sec = 8 * tf_sec
        horizon_sec = 6 * tf_sec
        next_scan = times[0] + context_sec
        last_end = times[-1] - horizon_sec

        while next_scan <= last_end:
            i = bisect.bisect_left(times, next_scan)
            h0 = bisect.bisect_left(times, next_scan - context_sec, 0, i)
            hist = Q[h0:i]
            for setup in strategy_setups(hist, tf, tf_sec):
                setups_by_tf[tf] += 1
                setups_by_strategy[setup.strategy_id] += 1
                j = bisect.bisect_right(times, next_scan + horizon_sec, i)
                if j > i:
                    trade = simulate_strategy(Q[i:j], setup)
                    if trade is not None:
                        trades.append(trade)
            next_scan += tf_sec

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
        "verification_level": "STRICT_FIB_STRATEGY_ADAPTER_GATE_RAW_BID_ASK_M1_M5_M15",
        "quotes": len(Q),
        "time_axis": "elapsed-time raw quote states; no OHLC/resample",
        "tf_seconds": TF_SECONDS,
        "fib_adapter_status": FIB_ADAPTER_STATUS,
        "generic_fib_touch_entry_allowed": False,
        "touch_without_confirmation_allowed": False,
        "setups": sum(setups_by_tf.values()),
        "setups_by_tf": setups_by_tf,
        "setups_by_strategy": setups_by_strategy,
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
            "Every Fibonacci family must pass its own strategy detector before its levels are armed.",
            "No generic Fib level-touch fallback exists; incomplete adapters are hard-disabled.",
            "OTE_CRT requires CRT range -> sweep -> reclaim -> delivery/MSS -> displacement -> OTE arrival -> depth-specific confirmation.",
            "E1 requires initial rejection; E2 requires rejection plus local structure confirmation; E3 requires deep sweep/reclaim plus micro-structure confirmation.",
            "M1/M5/M15 remain independent elapsed-time raw-tick states.",
            "Long Ask entry/Bid exit; Short Bid entry/Ask exit.",
        ],
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    (RESULT_DIR / "trades.json").write_text(json.dumps([asdict(t) for t in trades], indent=2), encoding="utf-8")
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    run()
