from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(df: pd.DataFrame, initial: float = 1000.0, days: int = 30) -> dict:
    if df.empty:
        return {
            "N": 0,
            "WR_pct": 0.0,
            "PF": 0.0,
            "NetProfit": 0.0,
            "MaxDD_pct": 0.0,
            "RF": None,
            "Monthly21_pct": 0.0,
            "Daily_pct": 0.0,
        }
    a = df["pnl"].astype(float).to_numpy()
    wins = a[a > 0]
    losses = a[a < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (float("inf") if len(wins) else 0.0)
    eq = initial
    peak = initial
    max_dd = 0.0
    for p in a:
        eq += float(p)
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    net = float(a.sum())
    dd_pct = max_dd / peak * 100.0 if peak > 0 else 0.0
    monthly = ((max(eq, 1e-9) / initial) ** (21.0 / days) - 1.0) * 100.0
    daily = ((1.0 + monthly / 100.0) ** (1.0 / 21.0) - 1.0) * 100.0 if monthly > -100 else -100.0
    return {
        "N": int(len(a)),
        "WR_pct": float((a > 0).mean() * 100.0),
        "PF": pf,
        "NetProfit": net,
        "MaxDD_pct": float(dd_pct),
        "RF": float(net / max_dd) if max_dd > 0 else None,
        "Monthly21_pct": float(monthly),
        "Daily_pct": float(daily),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-experiment-id", required=True)
    args = ap.parse_args()

    root = Path(args.input_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    trades = []
    manifests = []
    for tf in ("M1", "M5", "M15"):
        candidates = list(root.rglob(f"{args.base_experiment_id}-{tf}/summary.json"))
        if not candidates:
            raise SystemExit(f"missing summary for {tf}")
        summary_path = candidates[0]
        run_dir = summary_path.parent
        summaries.append(json.loads(summary_path.read_text()))
        trade_path = run_dir / "trades.csv"
        if trade_path.exists() and trade_path.stat().st_size > 0:
            df = pd.read_csv(trade_path)
            if not df.empty:
                trades.append(df)
        manifest_path = run_dir / "catalog_manifest.json"
        if manifest_path.exists():
            manifests.append(json.loads(manifest_path.read_text()))

    if not manifests:
        raise SystemExit("catalog manifest missing")
    manifest = manifests[0]
    days = int(manifest["days"])
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["symbol", "tf", "scene", "pnl", "ts_closed"])
    if not all_trades.empty:
        all_trades = all_trades.sort_values(["ts_closed", "tf"], kind="stable").reset_index(drop=True)

    scene_metrics = {}
    if not all_trades.empty:
        for scene, g in all_trades.groupby("scene"):
            scene_metrics[str(scene)] = metrics(g, days=days)

    combined = {
        "verification_level": "NAUTILUS_BT_RAW_BIDASK_MULTI_TF",
        "strategy": "AMOS_AllWeather_XAUUSD_MetaBot_v0.2",
        "data_kind": "RAW_BIDASK QuoteTick",
        "ohlc_resample_used": False,
        "timeframes": ["M1", "M5", "M15"],
        "period": {
            "start": manifest["start"],
            "days": days,
            "end_exclusive": manifest["end_exclusive"],
        },
        "portfolio_realized_close_ordered": metrics(all_trades, days=days),
        "tf_summaries": {s["timeframes"][0]: s for s in summaries},
        "scene_metrics": scene_metrics,
        "aggregation_note": "Portfolio order is synchronized by realized PositionClosed ts_closed across independent M1/M5/M15 Nautilus runs. MaxDD is realized-close DD, not mark-to-market floating DD.",
    }

    all_trades.to_csv(out / "portfolio_trades.csv", index=False)
    (out / "portfolio_summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    (out / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(combined["portfolio_realized_close_ordered"], indent=2))


if __name__ == "__main__":
    main()
