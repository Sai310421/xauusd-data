from __future__ import annotations

"""Select the strongest symbol/timeframe cells from a Raw Bid/Ask Nautilus summary.

This is a screening/routing layer, not a proof of out-of-sample superiority.
It uses only already-produced cell KPIs and emits immutable selection evidence.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

WEIGHTS = {
    "Monthly21_pct": 0.24,
    "PF": 0.22,
    "RF": 0.18,
    "WR_pct": 0.12,
    "N": 0.14,
    "MaxDD_pct": 0.10,
}


def _finite(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _percentile_ranks(values, reverse=False):
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    out = [0.0] * len(values)
    if len(values) == 1:
        return [1.0]
    for pos, idx in enumerate(order):
        # best receives 1.0, worst 0.0
        out[idx] = 1.0 - pos / (len(values) - 1)
    return out


def _pick_cells(summary):
    if isinstance(summary.get("cell_metrics"), dict):
        return summary["cell_metrics"], "cell_metrics"
    if isinstance(summary.get("edge_cells"), dict):
        return summary["edge_cells"], "edge_cells"
    if isinstance(summary.get("baseline_cells"), dict):
        return summary["baseline_cells"], "baseline_cells"
    raise SystemExit("No cell metrics found in summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--min-pf", type=float, default=1.0)
    ap.add_argument("--max-dd", type=float, default=20.0)
    ap.add_argument("--max-per-symbol", type=int, default=2)
    args = ap.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cells, source_key = _pick_cells(summary)
    rows = []
    for cell, m in sorted(cells.items()):
        if ":" not in cell:
            continue
        symbol, tf = cell.split(":", 1)
        row = {
            "cell": cell,
            "symbol": symbol,
            "tf": tf,
            "N": int(_finite(m.get("N"), 0)),
            "WR_pct": _finite(m.get("WR_pct"), 0.0),
            "PF": _finite(m.get("PF"), 0.0),
            "RF": _finite(m.get("RF"), -1e9),
            "NetProfit": _finite(m.get("NetProfit"), 0.0),
            "MaxDD_pct": _finite(m.get("MaxDD_pct"), 1e9),
            "Monthly21_pct": _finite(m.get("Monthly21_pct"), -1e9),
            "raw_ticks": int(_finite(m.get("raw_ticks"), 0)),
            "signals_submitted": int(_finite(m.get("signals_submitted"), 0)),
        }
        reasons = []
        if row["N"] < args.min_n:
            reasons.append(f"N<{args.min_n}")
        if row["NetProfit"] <= 0:
            reasons.append("NetProfit<=0")
        if row["PF"] < args.min_pf:
            reasons.append(f"PF<{args.min_pf}")
        if row["MaxDD_pct"] > args.max_dd:
            reasons.append(f"MaxDD>{args.max_dd}")
        row["eligible"] = not reasons
        row["reject_reasons"] = reasons
        rows.append(row)

    eligible = [r for r in rows if r["eligible"]]
    if eligible:
        for metric in ["Monthly21_pct", "PF", "RF", "WR_pct", "N"]:
            ranks = _percentile_ranks([r[metric] for r in eligible], reverse=True)
            for r, rank in zip(eligible, ranks):
                r[f"rank_{metric}"] = rank
        dd_ranks = _percentile_ranks([r["MaxDD_pct"] for r in eligible], reverse=False)
        for r, rank in zip(eligible, dd_ranks):
            r["rank_MaxDD_pct"] = rank
        for r in eligible:
            r["score"] = sum(WEIGHTS[k] * r[f"rank_{k}"] for k in WEIGHTS)
    for r in rows:
        r.setdefault("score", None)

    ranked = sorted(eligible, key=lambda r: (r["score"], r["Monthly21_pct"], r["PF"], r["N"]), reverse=True)
    selected = []
    per_symbol = {}
    for r in ranked:
        if len(selected) >= args.top_k:
            break
        if per_symbol.get(r["symbol"], 0) >= args.max_per_symbol:
            continue
        selected.append(r)
        per_symbol[r["symbol"]] = per_symbol.get(r["symbol"], 0) + 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset_sha = summary.get("dataset_sha256")
    result = {
        "selection_status": "SCREENING_ROUTER",
        "verification_level": summary.get("verification_level"),
        "data_kind": summary.get("data_kind", "RAW_BIDASK QuoteTick"),
        "ohlc_resample_used": summary.get("ohlc_resample_used"),
        "dataset_sha256": dataset_sha,
        "source_summary": str(summary_path),
        "source_cells_key": source_key,
        "weights": WEIGHTS,
        "gates": {
            "min_n": args.min_n,
            "min_pf": args.min_pf,
            "max_dd_pct": args.max_dd,
            "positive_net_profit": True,
            "top_k": args.top_k,
            "max_per_symbol": args.max_per_symbol,
        },
        "selected_cells": [r["cell"] for r in selected],
        "selected_symbols": sorted({r["symbol"] for r in selected}),
        "selected_timeframes": sorted({r["tf"] for r in selected}),
        "selected_mean_score": mean([r["score"] for r in selected]) if selected else None,
        "ranking": ranked,
        "all_cells": rows,
        "note": "Selection is a routing/screening decision only; promotion still requires sealed OOS Raw BidAsk validation and equal-exposure checks where applicable.",
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# Raw BidAsk Cell Selection",
        "",
        f"- Verification: {result['verification_level']}",
        f"- Dataset SHA: {dataset_sha}",
        f"- Selected: {', '.join(result['selected_cells']) if result['selected_cells'] else 'NONE'}",
        "",
        "| Rank | Cell | Score | Monthly21% | PF | RF | WR% | N | MaxDD% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['cell']} | {r['score']:.4f} | {r['Monthly21_pct']:.3f} | {r['PF']:.3f} | {r['RF']:.3f} | {r['WR_pct']:.2f} | {r['N']} | {r['MaxDD_pct']:.3f} |"
        )
    lines += ["", "## Rejected", ""]
    for r in rows:
        if not r["eligible"]:
            lines.append(f"- {r['cell']}: {', '.join(r['reject_reasons'])}")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_cells": result["selected_cells"], "dataset_sha256": dataset_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
