from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

from research.g75_multihorizon_fixed_debt_probe import (
    NS,
    Q,
    extract_cycles,
    load_catalog_quotes,
    run_shadow,
)

HORIZONS_S = (300, 3600, 4000, 4500, 5400, 21600, 43200, 86400)
POSITIVE_TARGETS = (0.10, 0.25, 0.50)


def analyze_positive_exit(qs: list[Q], max_layers: int) -> tuple[dict, list[dict]]:
    shadow = run_shadow(qs, max_layers)
    cycles = extract_cycles(shadow)
    ts = [q.ts for q in qs]
    idx_by_ts = {q.ts: i for i, q in enumerate(qs)}
    rows: list[dict] = []

    for c in cycles:
        i0 = idx_by_ts.get(c["loss_state_ts"])
        if i0 is None:
            continue

        side = int(c["side"])
        entries = [float(x) for x in c["entries"]]
        q0 = qs[i0]

        def basket_pnl(q: Q) -> float:
            px = q.bid if side > 0 else q.ask
            return sum(side * (px - e) for e in entries)

        p0 = basket_pnl(q0)
        if p0 >= 0:
            continue

        debt = -p0
        max_end = bisect.bisect_right(ts, c["loss_state_ts"] + HORIZONS_S[-1] * NS, lo=i0)
        hit_times: dict[str, float | None] = {"be": None}
        for frac in POSITIVE_TARGETS:
            hit_times[f"plus_{int(frac * 100)}pct_debt"] = None

        for q in qs[i0:max_end]:
            pnl = basket_pnl(q)
            elapsed = (q.ts - c["loss_state_ts"]) / NS
            if hit_times["be"] is None and pnl >= 0:
                hit_times["be"] = elapsed
            for frac in POSITIVE_TARGETS:
                key = f"plus_{int(frac * 100)}pct_debt"
                if hit_times[key] is None and pnl >= debt * frac:
                    hit_times[key] = elapsed
            if all(v is not None for v in hit_times.values()):
                break

        row: dict[str, object] = {
            "side": side,
            "layers": len(entries),
            "loss_state_ts": c["loss_state_ts"],
            "locked_debt_distance": debt,
            "spread_at_lock": q0.ask - q0.bid,
        }
        for name, tau in hit_times.items():
            row[f"tau_{name}_s"] = tau
            for h in HORIZONS_S:
                row[f"{name}_le_{h}s"] = bool(tau is not None and tau <= h)
        rows.append(row)

    def rate(field: str) -> float | None:
        if not rows:
            return None
        return sum(bool(r[field]) for r in rows) / len(rows)

    unresolved_be_3600 = [r for r in rows if not r["be_le_3600s"]]
    post_3600_be = {
        str(h): (
            sum(bool(r[f"be_le_{h}s"]) for r in unresolved_be_3600) / len(unresolved_be_3600)
            if unresolved_be_3600 else None
        )
        for h in (4000, 4500, 5400)
    }

    summary = {
        "classification": "G75_NATURAL_POSITIVE_EXIT_PROBE_V1",
        "truth_boundary": (
            "Raw Bid/Ask discovery probe. It measures whether the original executable basket price revisits "
            "Economic BE and positive PnL targets equal to +10%, +25% and +50% of the locked debt after the "
            "loss/lock timestamp. It does not simulate hedge orders, rescue trades, commissions, slippage, "
            "latency, swap/carry, margin or cashback. Therefore these are natural price-path opportunity rates, "
            "not realized rescue performance, and WR5 remains INVALID."
        ),
        "max_layers": max_layers,
        "loss_lock_candidates": len(rows),
        "horizons_s": list(HORIZONS_S),
        "be_rate_by_horizon": {str(h): rate(f"be_le_{h}s") for h in HORIZONS_S},
        "positive_exit_rate_by_horizon": {
            str(h): {
                f"plus_{int(frac * 100)}pct_debt": rate(
                    f"plus_{int(frac * 100)}pct_debt_le_{h}s"
                )
                for frac in POSITIVE_TARGETS
            }
            for h in HORIZONS_S
        },
        "post_3600_tail": {
            "unresolved_at_3600_count": len(unresolved_be_3600),
            "conditional_be_recovery_rate": post_3600_be,
            "interpretation": "P(BE<=h | BE>3600s) for h in 4000/4500/5400s",
        },
        "six_hour": {
            "be_rate": rate("be_le_21600s"),
            "plus_10pct_debt_rate": rate("plus_10pct_debt_le_21600s"),
            "plus_25pct_debt_rate": rate("plus_25pct_debt_le_21600s"),
            "plus_50pct_debt_rate": rate("plus_50pct_debt_le_21600s"),
        },
        "wr5": "INVALID",
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--raw-bidask-only", action="store_true")
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit("raw-bidask-only is mandatory")

    qs, manifest = load_catalog_quotes(Path(args.catalog), args.symbol)
    out = Path("results/ae-bt") / args.experiment_id
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "verification_level": "NAUTILUS_RAW_BIDASK_G75_NATURAL_POSITIVE_EXIT_PROBE",
        "symbol": args.symbol,
        "quote_rows": len(qs),
        "period": {
            "start": manifest.get("start"),
            "days": manifest.get("days"),
            "end_exclusive": manifest.get("end_exclusive"),
        },
        "ohlc_resample_used": False,
    }

    for L in (10, 20):
        summary, rows = analyze_positive_exit(qs, L)
        result[f"L{L}"] = summary
        if rows:
            with (out / f"g75_positive_exit_l{L}_events.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    (out / "g75_6h_positive_exit_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
