from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_json(root: Path, names: tuple[str, ...]) -> Any:
    for name in names:
        p = root / name
        obj = read_json(p)
        if obj is not None:
            return obj
    return None


def assemble(root: Path) -> dict[str, Any]:
    pre = read_kv(root / "preflight" / "github_run.txt")
    catalog = read_json(root / "preflight" / "catalog_manifest.json")
    config = first_json(root, ("configuration.json", "config.json", "bt_config.json"))
    metrics = first_json(root, ("metrics.json", "summary.json", "bt_summary.json", "result.json"))
    wr5 = first_json(root, ("wr5.json", "reality_filter.json", "reality.json"))
    test_period = first_json(root, ("test_period.json", "period.json"))
    sizing = first_json(root, ("sizing.json", "risk.json"))
    robustness = first_json(root, ("robustness.json", "wfo_mc.json"))
    live_parity = first_json(root, ("live_parity.json", "shadow_parity.json"))
    portfolio = first_json(root, ("portfolio.json", "portfolio_risk.json"))

    return {
        "schema": "amos-experiment-evidence-v1",
        "experiment_id": pre.get("experiment_id"),
        "run_id": pre.get("run_id"),
        "git_sha": pre.get("git_sha"),
        "lane_class": pre.get("lane_class"),
        "runner": pre.get("runner"),
        "dataset_or_catalog_manifest": catalog,
        "configuration": config,
        "test_period": test_period,
        "sizing": sizing,
        "metrics": metrics,
        "wr5": wr5,
        "robustness": robustness,
        "live_parity": live_parity,
        "portfolio": portfolio,
        "source_root": str(root),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    payload = assemble(args.root)
    out = args.output or (args.root / "amos_evidence.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
