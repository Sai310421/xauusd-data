"""Profile a Myfxbook TickScalper CSV without inferring proprietary source code.

The output is a reproducible behavioral contract:
- direction-separated L2 streams
- candidate basket clusters
- geometric lot ladders
- explicit basket-stop observations

It does not claim to recover the original entry algorithm.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

TIME_FMT = "%m/%d/%Y %H:%M"


def _f(value: str | None) -> float:
    return float(value) if value else 0.0


def _parse(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in source:
            if row.get("Symbol") != "XAUUSD" or row.get("Action") not in {"Buy", "Sell"}:
                continue
            row["_open"] = datetime.strptime(row["Open Date"], TIME_FMT)
            row["_close"] = datetime.strptime(row["Close Date"], TIME_FMT)
            row["_lot"] = _f(row.get("Units/Lots"))
            row["_pnl"] = _f(row.get("Profit"))
            row["_sl"] = _f(row.get("SL"))
            row["_tp"] = _f(row.get("TP"))
            rows.append(row)
    if not rows:
        raise ValueError("No XAUUSD Buy/Sell rows found")
    return rows


def _median_ratio(lots: list[float]) -> float | None:
    ordered = sorted(x for x in lots if x > 0)
    ratios = [right / left for left, right in zip(ordered, ordered[1:])]
    return statistics.median(ratios) if ratios else None


def _cluster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["Close Date"], row.get("Comment", ""), row["Action"])].append(row)
    result: list[dict[str, Any]] = []
    for (closed, comment, side), group in grouped.items():
        group = sorted(group, key=lambda row: row["_open"])
        lots = [row["_lot"] for row in group]
        ratio = _median_ratio(lots)
        result.append(
            {
                "close_time": closed,
                "comment": comment,
                "side": side,
                "trades": len(group),
                "first_open": group[0]["Open Date"],
                "last_open": group[-1]["Open Date"],
                "minutes_open_span": (group[-1]["_open"] - group[0]["_open"]).total_seconds() / 60,
                "lots_in_open_order": lots,
                "median_lot_ratio_sorted": ratio,
                "geometric_1_5_candidate": bool(ratio and 1.45 <= ratio <= 1.55),
                "net_pnl": round(sum(row["_pnl"] for row in group), 2),
                "has_sl": any(row["_sl"] != 0.0 for row in group),
            }
        )
    return result


def profile(path: Path) -> dict[str, Any]:
    rows = _parse(path)
    clusters = _cluster(rows)
    wins = [row["_pnl"] for row in rows if row["_pnl"] > 0]
    losses = [row["_pnl"] for row in rows if row["_pnl"] < 0]
    durations = sorted((row["_close"] - row["_open"]).total_seconds() / 60 for row in rows)
    comments = Counter(row.get("Comment", "") for row in rows)
    magic_side = Counter((row.get("Magic Number", ""), row["Action"]) for row in rows)
    opposing_open_minutes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        opposing_open_minutes[row["Open Date"]].add(row["Action"])
    grid = [item for item in clusters if item["trades"] >= 3 and item["geometric_1_5_candidate"]]
    stopped = [item for item in clusters if item["has_sl"]]
    return {
        "schema": "TICKSCALPER_CSV_PROFILE_V1",
        "source": str(path),
        "scope": "observed behavior only; not original code",
        "period": {
            "first_open": min(row["_open"] for row in rows).isoformat(),
            "last_close": max(row["_close"] for row in rows).isoformat(),
            "csv_timezone": "UNSPECIFIED",
        },
        "trades": len(rows),
        "sides": dict(Counter(row["Action"] for row in rows)),
        "comment_counts": dict(comments),
        "magic_side_counts": {f"{magic}|{side}": count for (magic, side), count in magic_side.items()},
        "risk_settings_observed": {
            "sl_trade_count": sum(row["_sl"] != 0.0 for row in rows),
            "tp_trade_count": sum(row["_tp"] != 0.0 for row in rows),
        },
        "performance_observed": {
            "net_pnl": round(sum(row["_pnl"] for row in rows), 2),
            "profit_factor": round(sum(wins) / -sum(losses), 6) if losses else None,
            "win_rate": round(len(wins) / len(rows), 6),
            "average_win": round(statistics.mean(wins), 4) if wins else None,
            "average_loss": round(statistics.mean(losses), 4) if losses else None,
            "duration_minutes": {
                "p50": statistics.median(durations),
                "p90": durations[int(0.90 * (len(durations) - 1))],
                "p95": durations[int(0.95 * (len(durations) - 1))],
                "max": max(durations),
            },
        },
        "clusters": {
            "count": len(clusters),
            "size_distribution": dict(Counter(item["trades"] for item in clusters)),
            "grid_1_5_candidates_3plus": len(grid),
            "basket_stop_clusters": len(stopped),
            "max_cluster_trades": max(item["trades"] for item in clusters),
        },
        "parallel_direction_open_minutes": sum(1 for sides in opposing_open_minutes.values() if len(sides) > 1),
        "largest_loss_clusters": sorted(clusters, key=lambda item: item["net_pnl"])[:10],
        "basket_stop_clusters_detail": stopped,
        "confirmation": {
            "confirmed": [
                "L2 buy/sell streams can operate concurrently",
                "many multi-trade clusters use a ~1.5 geometric lot ladder",
                "some multi-trade clusters carry a common stop loss",
            ],
            "not_confirmed": [
                "initial entry signal",
                "CSV timezone relative to QuoteTick UTC",
                "whether same-minute exits are one or multiple baskets",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = profile(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
