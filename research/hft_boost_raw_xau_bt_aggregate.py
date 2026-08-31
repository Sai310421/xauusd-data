from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.hft_boost_raw_xau_bt import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing downloaded shard artifacts")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--days", type=float, default=30.0)
    args = ap.parse_args()

    root = Path(args.root)
    summaries = sorted(root.rglob("summary.json"))
    if not summaries:
        raise SystemExit("no shard summary.json files found")

    rows = []
    totals = {
        "raw_ticks": 0,
        "signals": 0,
        "entries_submitted": 0,
        "order_fills": 0,
        "order_rejects": 0,
        "order_denials": 0,
        "order_cancels": 0,
        "closed_positions": 0,
    }
    shard_meta = []

    for sp in summaries:
        s = json.loads(sp.read_text())
        if s.get("verification_level") != "NAUTILUS_BT_RAW_BIDASK_SHARD":
            continue
        trade_path = sp.parent / "trades.csv"
        if trade_path.exists() and trade_path.stat().st_size > 0:
            df = pd.read_csv(trade_path)
            if not df.empty:
                rows.extend(df[["pnl", "ts_closed"]].to_dict("records"))
        for key in totals:
            totals[key] += int(s.get(key, 0) or 0)
        shard_meta.append({"period": s.get("period"), "raw_ticks": s.get("raw_ticks"), "closed_positions": s.get("closed_positions")})

    if not shard_meta:
        raise SystemExit("no valid Raw BidAsk shard summaries found")

    rows.sort(key=lambda x: int(x.get("ts_closed", 0)))
    k = metrics(rows, args.initial, args.days)

    outdir = Path("results/ae-bt") / args.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["pnl", "ts_closed"]).to_csv(outdir / "trades.csv", index=False)

    summary = {
        "verification_level": "NAUTILUS_BT_RAW_BIDASK_SHARDED_AGGREGATE",
        "edge": "HFT_BOOST_BASE_v0.7",
        "data_kind": "RAW_BIDASK QuoteTick",
        "ohlc_resample_used": False,
        "aggregation": "Trade-level recomputation across chronological shard outputs; counters summed across shards",
        "shard_count": len(shard_meta),
        "period_days": args.days,
        **totals,
        "kpi": k,
        "shards": shard_meta,
        "limitations": [
            "Each shard starts strategy state fresh at its boundary; no synthetic OHLC/resample is used.",
            "MaxClosedDD is recomputed from the chronological realized-PnL stream; floating DD is not included in this BASE gate."
        ],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True)
    (outdir / "summary.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
