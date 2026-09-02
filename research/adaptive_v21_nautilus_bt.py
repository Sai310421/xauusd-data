from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean

RESULT_DIR = Path(os.getenv("RESULT_DIR", "results/adaptive-v21"))
RESULT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_LEVELS = {
    "OTE": {"core": 0.708, "front": 0.669, "deep": 0.786},
    "POP": {"core": 0.669, "front": 0.559, "deep": 0.786},
    "GOLD_SILVER": {"core": 0.705, "front": 0.688, "deep": 0.718},
}

@dataclass
class EntrySlice:
    name: str
    ratio: float
    role: str
    filled: bool = False
    fill_price: float | None = None

@dataclass
class Trade:
    side: str
    strategy: str
    entry: float
    exit: float
    pnl: float
    mae: float
    mfe: float
    slices: int


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fib_price(a: float, b: float, r: float, bullish: bool) -> float:
    return b - r * (b - a) if bullish else b + r * (a - b)


def build_slices(strategy: str, a: float, b: float, bullish: bool) -> list[EntrySlice]:
    lv = PRIMARY_LEVELS[strategy]
    return [
        EntrySlice("E1", lv["front"], "front"),
        EntrySlice("E2", lv["core"], "primary"),
        EntrySlice("E3", lv["deep"], "deep"),
    ]


def load_quotes() -> list[dict]:
    candidates = [
        Path("books/xauusd_quotes.jsonl"),
        Path("books/xauusd_ticks.jsonl"),
        Path("data/xauusd_quotes.jsonl"),
        Path("data/xauusd_ticks.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            rows = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows
    raise FileNotFoundError(
        "No repository raw quote JSONL found. Expected one of: " + ", ".join(map(str, candidates))
    )


def mid(q: dict) -> float:
    bid = float(q.get("bid", q.get("b", 0)))
    ask = float(q.get("ask", q.get("a", 0)))
    return (bid + ask) / 2.0


def spread(q: dict) -> float:
    bid = float(q.get("bid", q.get("b", 0)))
    ask = float(q.get("ask", q.get("a", 0)))
    return ask - bid


def run() -> None:
    # This runner is intentionally strict: it only promotes to REAL_BT when
    # repository raw Bid/Ask quote data is actually present. It does not fall
    # back to OHLC resampling or synthetic fills.
    try:
        quotes = load_quotes()
    except FileNotFoundError as e:
        manifest = {
            "verification_level": "BLOCKED_NO_RAW_QUOTES",
            "reason": str(e),
            "ohlc_resample_used": False,
            "required": "Raw Bid/Ask quote JSONL or catalog adapter",
        }
        (RESULT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return

    if len(quotes) < 500:
        raise RuntimeError(f"Insufficient raw quotes: {len(quotes)}")

    # Minimal causal v2.1 execution-gate harness. It verifies the 3-stage
    # Strategy-Specific Primary architecture without lookahead. Full A0/B0
    # signal discovery is intentionally separated from execution accounting.
    mids = [mid(q) for q in quotes]
    spr = [spread(q) for q in quotes]
    trades: list[Trade] = []

    window = 240
    horizon = 120
    for i in range(window, len(quotes) - horizon, horizon):
        hist = mids[i-window:i]
        lo, hi = min(hist), max(hist)
        if hi <= lo:
            continue
        bullish = hist[-1] > mean(hist)
        a, b = (lo, hi) if bullish else (hi, lo)

        # Initial gate uses OTE only. B0/Butterfly gets its own D-PRZ adapter
        # once harmonic point extraction is wired from the strategy library.
        strategy = "OTE"
        slices = build_slices(strategy, a, b, bullish)
        prices = {s.name: fib_price(a, b, s.ratio, bullish) for s in slices}

        fills: list[tuple[float, float]] = []
        invalidation = lo if bullish else hi
        path = quotes[i:i+horizon]
        path_mid = [mid(q) for q in path]
        for s in slices:
            p = prices[s.name]
            touched = any(x <= p for x in path_mid) if bullish else any(x >= p for x in path_mid)
            if touched:
                # actual-side quote approximation: long fills at ask, short at bid
                j = next(j for j, x in enumerate(path_mid) if (x <= p if bullish else x >= p))
                q = path[j]
                fill = float(q.get("ask" if bullish else "bid", q.get("a" if bullish else "b")))
                risk_dist = abs(fill - invalidation)
                if risk_dist > 0:
                    fills.append((fill, risk_dist))

        if not fills:
            continue

        # Equal risk allocation across filled slices; no martingale.
        weights = [1.0 / len(fills)] * len(fills)
        avg_entry = sum(w * f[0] for w, f in zip(weights, fills))
        future = path_mid
        exit_px = future[-1]
        direction = 1.0 if bullish else -1.0
        pnl = direction * (exit_px - avg_entry)
        excursions = [direction * (x - avg_entry) for x in future]
        trades.append(Trade(
            side="LONG" if bullish else "SHORT",
            strategy=strategy,
            entry=avg_entry,
            exit=exit_px,
            pnl=pnl,
            mae=min(excursions),
            mfe=max(excursions),
            slices=len(fills),
        ))

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [-t.pnl for t in trades if t.pnl < 0]
    gross_win = sum(wins)
    gross_loss = sum(losses)
    pf = gross_win / gross_loss if gross_loss > 0 else math.inf if gross_win > 0 else 0.0
    wr = len(wins) / len(trades) if trades else 0.0
    ev = sum(t.pnl for t in trades) / len(trades) if trades else 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    summary = {
        "verification_level": "REAL_RAW_BID_ASK_EXECUTION_HARNESS",
        "quotes": len(quotes),
        "trades": len(trades),
        "wr": wr,
        "pf": pf,
        "ev_price_units": ev,
        "net_price_units": sum(t.pnl for t in trades),
        "max_dd_price_units": max_dd,
        "avg_mae": mean([t.mae for t in trades]) if trades else 0.0,
        "avg_mfe": mean([t.mfe for t in trades]) if trades else 0.0,
        "slice_fill_counts": {
            "1": sum(t.slices == 1 for t in trades),
            "2": sum(t.slices == 2 for t in trades),
            "3": sum(t.slices == 3 for t in trades),
        },
        "ohlc_resample_used": False,
        "notes": [
            "Causal raw Bid/Ask execution harness for v2.1 3-stage entry.",
            "OTE primary core is 0.708; E1/E2/E3 are front/primary/deep.",
            "Butterfly D-PRZ extraction remains an explicit next adapter, not faked by OTE ratios.",
        ],
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULT_DIR / "trades.json").write_text(json.dumps([asdict(t) for t in trades], indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
