from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DD_TARGETS = [3.0, 4.5, 5.0, 10.0, 15.0, 20.0]
MODES = ["BASE", "A0", "A1"]
LOTS = ["0.005", "0.01", "0.015", "0.02"]
LAYERS = [5, 10]


def run_cell(catalog: str, root: Path, mode: str, lot: str, layers: int) -> dict:
    eid = f"g75-frontier-{mode}-lot{lot.replace('.', 'p')}-L{layers}"
    cmd = [
        sys.executable, "research/g75_vgrsi_raw_bidask_bt.py",
        "--catalog", catalog,
        "--experiment-id", eid,
        "--symbol", "XAUUSD",
        "--modes", mode,
        "--max-layers", str(layers),
        "--base-lot", lot,
        "--raw-bidask-only",
    ]
    subprocess.run(cmd, check=True)
    d = json.loads((root / eid / "summary.json").read_text(encoding="utf-8"))
    m = d["modes"][mode]
    return {
        "mode": mode,
        "base_lot": float(lot),
        "max_layers": layers,
        **m,
    }


def feasible(cell: dict, dd: float) -> bool:
    return (
        cell.get("N", 0) > 0
        and cell.get("OrderRejected", 0) == 0
        and cell.get("MaxFloatingDD_pct") is not None
        and cell["MaxFloatingDD_pct"] <= dd
        and cell.get("Monthly21_pct") is not None
    )


def score(cell: dict) -> tuple:
    # Primary objective: maximum 21-day compounded return.
    # Tie-breakers reward robustness rather than simply more leverage.
    return (
        float(cell.get("Monthly21_pct") or -1e18),
        float(cell.get("RF_floating") or -1e18),
        float(cell.get("PF") or -1e18),
        float(cell.get("N_per_day") or 0.0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", default="results/ae-bt/g75-dd-return-frontier")
    args = ap.parse_args()
    root = Path("results/ae-bt")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cells = []
    for mode in MODES:
        for lot in LOTS:
            for layers in LAYERS:
                cells.append(run_cell(args.catalog, root, mode, lot, layers))

    if not any(c["N"] > 0 for c in cells):
        reject_summary = {
            "status": "EXECUTION_GATE_BLOCKED",
            "reason": "No filled cycles. Frontier is invalid until Raw BidAsk Nautilus execution produces N>0.",
            "cells": cells,
        }
        (outdir / "frontier.json").write_text(json.dumps(reject_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(reject_summary, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    frontier = {}
    for dd in DD_TARGETS:
        xs = [c for c in cells if feasible(c, dd)]
        frontier[str(dd)] = max(xs, key=score) if xs else None

    best_unconstrained = max([c for c in cells if c.get("N", 0) > 0 and c.get("OrderRejected", 0) == 0], key=score, default=None)
    summary = {
        "status": "VALID_RAW_BIDASK_FRONTIER",
        "objective": "maximize Monthly21_pct subject to MaxFloatingDD_pct <= DD target",
        "constraints": DD_TARGETS,
        "search_space": {"modes": MODES, "base_lots": [float(x) for x in LOTS], "max_layers": LAYERS},
        "frontier": frontier,
        "best_unconstrained_in_grid": best_unconstrained,
        "cells": cells,
        "validity": {
            "raw_bidask_required": True,
            "ohlc_execution_forbidden": True,
            "commission_slippage_latency": "NOT_INCLUDED_YET; results are research frontier, not production expectancy",
        },
    }
    (outdir / "frontier.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["DD_pct,mode,base_lot,max_layers,Monthly21_pct,DailyCompound_pct,WR_pct,PF,RF_floating,MaxFloatingDD_pct,N,N_per_day"]
    for dd in DD_TARGETS:
        c = frontier[str(dd)]
        if c is None:
            lines.append(f"{dd},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA")
        else:
            lines.append(",".join(str(x) for x in [dd,c['mode'],c['base_lot'],c['max_layers'],c['Monthly21_pct'],c['DailyCompound_pct'],c['WR_pct'],c['PF'],c['RF_floating'],c['MaxFloatingDD_pct'],c['N'],c['N_per_day']]))
    (outdir / "frontier.csv").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
