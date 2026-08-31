from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ID = "BIGPLAYER_STANDALONE"
EXPERIMENT_ID = "BP2EDGE_XAUUSD_M1_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_edges(df: pd.DataFrame, lookback: int, vol_sigma: float, range_mult: float, wick_ratio: float, point: float) -> pd.DataFrame:
    out = df.copy()
    rng = out["high"] - out["low"]
    vol = out["volume"].astype(float)

    vol_mean = vol.shift(1).rolling(lookback, min_periods=lookback).mean()
    vol_std = vol.shift(1).rolling(lookback, min_periods=lookback).std(ddof=0)
    range_mean = rng.shift(1).rolling(lookback, min_periods=lookback).mean()
    z = (vol - vol_mean) / vol_std.replace(0.0, np.nan)

    body = (out["close"] - out["open"]).abs()
    body_safe = body.clip(lower=point)
    body_ratio = body / rng.replace(0.0, np.nan)
    upper = out["high"] - out[["open", "close"]].max(axis=1)
    lower = out[["open", "close"]].min(axis=1) - out["low"]
    high_volume = (z >= vol_sigma) & (rng > 0.0)

    imbalance = pd.Series(0, index=out.index, dtype="int8")
    imb_gate = high_volume & ((rng / range_mean) >= range_mult) & (body_ratio >= 0.60)
    imbalance.loc[imb_gate & (out["close"] > out["open"])] = 1
    imbalance.loc[imb_gate & (out["close"] < out["open"])] = -1

    absorption = pd.Series(0, index=out.index, dtype="int8")
    long_lower = high_volume & (lower >= body_safe * wick_ratio) & (lower > upper)
    long_upper = high_volume & (upper >= body_safe * wick_ratio) & (upper > lower)
    absorption.loc[long_lower] = 1
    absorption.loc[long_upper] = -1

    out["bp_imbalance"] = imbalance
    out["bp_absorption"] = absorption
    return out


def signal_for_variant(df: pd.DataFrame, variant: str) -> pd.Series:
    i = df["bp_imbalance"]
    a = df["bp_absorption"]
    if variant == "IMBALANCE_ONLY":
        return i.astype("int8")
    if variant == "ABSORPTION_ONLY":
        return a.astype("int8")
    if variant == "BOTH_AND":
        return i.where((i != 0) & (a != 0), 0).astype("int8")
    if variant == "BOTH_DIRECTION_AGREE":
        return i.where((i != 0) & (a != 0) & (i == a), 0).astype("int8")
    raise ValueError(variant)


def run_variant(df: pd.DataFrame, signal: pd.Series, horizon: int, initial_balance: float, lot: float, contract_size: float, round_trip_cost_usd: float):
    qty = lot * contract_size
    trades = []
    next_free = 0
    nrows = len(df)

    for idx in np.flatnonzero(signal.to_numpy() != 0):
        if idx < next_free or idx + 1 + horizon >= nrows:
            continue
        direction = int(signal.iat[idx])
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon - 1
        entry = float(df["open"].iat[entry_idx])
        exit_ = float(df["close"].iat[exit_idx])
        pnl = direction * (exit_ - entry) * qty - round_trip_cost_usd
        trades.append((idx, entry_idx, exit_idx, direction, entry, exit_, pnl))
        next_free = exit_idx + 1

    if not trades:
        return {
            "N": 0, "N_per_day": 0.0, "WR": 0.0, "PF": 0.0, "RF": 0.0,
            "net_profit": 0.0, "return_pct": 0.0, "max_dd_pct": 0.0,
            "max_dd_usd": 0.0, "final_balance": initial_balance,
            "event_N": int((signal != 0).sum()),
        }, pd.DataFrame(columns=["signal_idx","entry_idx","exit_idx","direction","entry","exit","pnl"])

    t = pd.DataFrame(trades, columns=["signal_idx","entry_idx","exit_idx","direction","entry","exit","pnl"])
    gross_profit = float(t.loc[t.pnl > 0, "pnl"].sum())
    gross_loss = float(-t.loc[t.pnl < 0, "pnl"].sum())
    net = float(t.pnl.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    wr = float((t.pnl > 0).mean() * 100.0)

    eq = initial_balance + t.pnl.cumsum()
    peak = pd.concat([pd.Series([initial_balance]), eq], ignore_index=True).cummax().iloc[1:].to_numpy()
    dd_usd = peak - eq.to_numpy()
    max_dd_usd = float(dd_usd.max()) if len(dd_usd) else 0.0
    dd_pct_series = np.where(peak > 0, dd_usd / peak * 100.0, 0.0)
    max_dd_pct = float(dd_pct_series.max()) if len(dd_pct_series) else 0.0
    rf = net / max_dd_usd if max_dd_usd > 0 else (math.inf if net > 0 else 0.0)

    days = max((df["datetime"].iat[-1] - df["datetime"].iat[0]).total_seconds() / 86400.0, 1.0)
    final_balance = initial_balance + net
    return {
        "N": int(len(t)),
        "N_per_day": float(len(t) / days),
        "WR": wr,
        "PF": pf,
        "RF": rf,
        "net_profit": net,
        "return_pct": float(net / initial_balance * 100.0),
        "max_dd_pct": max_dd_pct,
        "max_dd_usd": max_dd_usd,
        "final_balance": final_balance,
        "event_N": int((signal != 0).sum()),
    }, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/bigplayer_2edge")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--initial-balance", type=float, default=1000.0)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--contract-size", type=float, default=100.0)
    ap.add_argument("--round-trip-cost-usd", type=float, default=0.0)
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--vol-sigma", type=float, default=2.0)
    ap.add_argument("--range-mult", type=float, default=1.5)
    ap.add_argument("--wick-ratio", type=float, default=1.2)
    ap.add_argument("--point", type=float, default=0.01)
    args = ap.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    df = build_edges(df, args.lookback, args.vol_sigma, args.range_mult, args.wick_ratio, args.point)
    dataset_hash = sha256_file(data_path)
    git_sha = os.getenv("GITHUB_SHA", "LOCAL")
    dataset_id = data_path.name

    variants = ["IMBALANCE_ONLY", "ABSORPTION_ONLY", "BOTH_AND", "BOTH_DIRECTION_AGREE"]
    rows = []
    for v in variants:
        sig = signal_for_variant(df, v)
        metrics, trades = run_variant(
            df, sig, args.horizon, args.initial_balance, args.lot, args.contract_size, args.round_trip_cost_usd
        )
        row = {
            "project_id": PROJECT_ID,
            "experiment_id": EXPERIMENT_ID,
            "variant": v,
            "git_sha": git_sha,
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "execution_model": f"closed_M1_signal_next_open_fixed_{args.horizon}m_nonoverlap",
            "spread_model": "included_in_round_trip_cost_usd",
            "slippage_model": "included_in_round_trip_cost_usd",
            "round_trip_cost_usd": args.round_trip_cost_usd,
            "lot": args.lot,
            "contract_size": args.contract_size,
            **metrics,
        }
        rows.append(row)
        trades.to_csv(out_dir / f"trades_{v}.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
