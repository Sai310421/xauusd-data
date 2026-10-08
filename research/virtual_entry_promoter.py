from __future__ import annotations
import argparse, ast, json, math, re
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = {
    "A": ["z", "rsi", "adx", "mid_slope_atr", "entry_spread_points"],
    "B": ["adx", "atr_exp", "breakout_up_atr", "breakout_dn_atr", "entry_spread_points"],
    "C": ["z", "mom_points", "ewma_sd", "entry_spread_points"],
}
MIN_N = {"A": 10, "B": 8, "C": 100}
QUANTILES = (0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85)

def parse_features(s: str) -> dict:
    s = re.sub(r"np\.float64\(([^()]*)\)", r"\1", str(s))
    return ast.literal_eval(s)

def stats(df: pd.DataFrame) -> dict:
    p = df["pnl_r"].to_numpy(float)
    if len(p) == 0:
        return {"N": 0, "PF": None, "WR_pct": None, "expectancy_R": None}
    gp = float(p[p > 0].sum())
    gl = abs(float(p[p < 0].sum()))
    pf = gp / gl if gl > 0 else None
    return {
        "N": int(len(p)),
        "PF": pf,
        "WR_pct": float((p > 0).mean() * 100.0),
        "expectancy_R": float(p.mean()),
    }

def add_features(df: pd.DataFrame, engine: str) -> pd.DataFrame:
    f = pd.json_normalize(df["features"].map(parse_features))
    x = pd.concat([df.reset_index(drop=True), f], axis=1)
    x["entry_dt"] = pd.to_datetime(x["entry_ts"], unit="ns", utc=True)
    x["date"] = x["entry_dt"].dt.date
    for col in FEATURES[engine]:
        if col in ("z", "mom_points", "breakout_up_atr", "breakout_dn_atr") and col in x:
            x["abs_" + col] = x[col].abs()
    return x

def make_conditions(w1: pd.DataFrame, engine: str):
    feats = FEATURES[engine] + [c for c in w1.columns if c.startswith("abs_")]
    out = []
    for feat in feats:
        if feat not in w1:
            continue
        vals = w1[feat].dropna()
        if vals.empty:
            continue
        for q in QUANTILES:
            t = float(vals.quantile(q))
            out.append((f"{feat}<={t:.8g}", {"kind":"le","feature":feat,"value":t}))
            out.append((f"{feat}>={t:.8g}", {"kind":"ge","feature":feat,"value":t}))
    out.append(("side=BUY", {"kind":"side","value":"BUY"}))
    out.append(("side=SELL", {"kind":"side","value":"SELL"}))
    for h0, h1 in ((0,6),(6,12),(12,18),(18,24),(7,10),(12,16),(16,20),(20,24)):
        out.append((f"hour[{h0},{h1})", {"kind":"hour","lo":h0,"hi":h1}))
    return out

def mask(df: pd.DataFrame, conds) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for cond in conds:
        kind = cond["kind"]
        if kind == "le":
            m &= df[cond["feature"]].to_numpy(float) <= cond["value"]
        elif kind == "ge":
            m &= df[cond["feature"]].to_numpy(float) >= cond["value"]
        elif kind == "side":
            m &= df["side"].to_numpy(str) == cond["value"]
        elif kind == "hour":
            h = df["entry_dt"].dt.hour.to_numpy(int)
            m &= (h >= cond["lo"]) & (h < cond["hi"])
    return m

def scan_one(x: pd.DataFrame, engine: str, variant: str) -> dict:
    dates = sorted(x["date"].unique())
    if len(dates) < 15:
        return {"variant": variant, "status": "INVALID", "reason": "insufficient_unique_dates", "unique_dates": len(dates)}
    d1, d2, d3 = set(dates[:7]), set(dates[7:14]), set(dates[14:])
    folds = [x[x["date"].isin(ds)].copy() for ds in (d1,d2,d3)]
    base = [stats(d) for d in folds]
    conditions = make_conditions(folds[0], engine)
    candidates = [(c,) for c in conditions] + list(combinations(conditions, 2))
    good = []
    for combo in candidates:
        fold_stats = []
        valid = True
        for d in folds:
            mm = mask(d, [c[1] for c in combo])
            s = stats(d[mm])
            fold_stats.append(s)
            if s["N"] < MIN_N[engine] or s["PF"] is None or s["PF"] <= 1.0 or s["expectancy_R"] is None or s["expectancy_R"] <= 0:
                valid = False
                break
        if valid:
            min_pf = min(s["PF"] for s in fold_stats)
            total_n = sum(s["N"] for s in fold_stats)
            good.append({
                "rule": " & ".join(c[0] for c in combo),
                "conditions": [c[1] for c in combo],
                "folds": fold_stats,
                "min_fold_pf": float(min_pf),
                "total_N": int(total_n),
            })
    good.sort(key=lambda z: (z["min_fold_pf"], z["total_N"]), reverse=True)
    return {
        "variant": variant,
        "status": "STABLE_SHADOW" if good else "REJECT_SHADOW",
        "base_folds": base,
        "stable_candidate_count": len(good),
        "top_candidates": good[:20],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("A","B","C"), required=True)
    ap.add_argument("--virtual-trades", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.virtual_trades)
    x = add_features(df, args.engine)
    variants = sorted(x["variant"].dropna().astype(str).unique()) if "variant" in x else ["legacy"]
    result = {
        "method": "7d_calibration_plus_7d_validation_plus_7d_validation",
        "engine": args.engine,
        "promotion_policy": "No live promotion. Candidate must keep PF>1 and expectancy_R>0 in all three sequential folds.",
        "variants": [],
    }
    for v in variants:
        xv = x[x["variant"].astype(str) == v].copy() if "variant" in x else x.copy()
        result["variants"].append(scan_one(xv, args.engine, v))
    result["status"] = "WATCH_ONLY"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
