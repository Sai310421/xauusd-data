"""Fail closed on missing/invalid ABC Nautilus Raw QuoteTick metrics; no live orders."""
import argparse
import json
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/amos-abc/rawtick-kpi.json")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.is_file():
        raise SystemExit("FAIL: missing KPI artifact")
    data = json.loads(path.read_text())
    if data.get("verification") != "NAUTILUS_RAW_QUOTETICK_VIRTUAL_EXECUTION":
        raise SystemExit("FAIL: not genuine Nautilus Raw QuoteTick verification")
    ticks = data.get("raw_ticks")
    if not isinstance(ticks, int) or ticks < 1000:
        raise SystemExit("FAIL: insufficient raw QuoteTicks")
    if not isinstance(data.get("first_ns"), int) or not isinstance(data.get("last_ns"), int) or data["last_ns"] <= data["first_ns"]:
        raise SystemExit("FAIL: invalid tick time range")
    dd = data.get("max_floating_dd_pct_combined")
    if not isinstance(dd, (int, float)) or not math.isfinite(dd) or dd < 0:
        raise SystemExit("FAIL: invalid combined floating DD")
    edges = data.get("edges")
    if not isinstance(edges, dict) or set(edges) != {"A01", "B01", "C01"}:
        raise SystemExit("FAIL: incomplete ABC results")
    for name, edge in edges.items():
        n = edge.get("N")
        if not isinstance(n, int) or n <= 0:
            raise SystemExit(f"FAIL: {name} has no closed trades; KPI not established")
        for key in ("WR_pct", "net_usd", "EV_usd", "mean_MAE_usd", "mean_MFE_usd"):
            value = edge.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise SystemExit(f"FAIL: {name} invalid {key}")
        pf = edge.get("PF")
        if not ((isinstance(pf, (int, float)) and math.isfinite(pf) and pf >= 0) or pf == "inf"):
            raise SystemExit(f"FAIL: {name} invalid PF")
    print("PASS: ABC Raw QuoteTick KPI schema, nonzero trades and finite risk metrics")
    print(json.dumps({"raw_ticks": ticks, "max_floating_dd_pct_combined": dd,
                      "edges": {k: {"N": v["N"], "PF": v["PF"], "WR_pct": v["WR_pct"]} for k, v in edges.items()}}, indent=2))

if __name__ == "__main__":
    main()
