#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALGO = ROOT / "research" / "qc" / "xauusd_hour_cloud.py"
CONFIG = ROOT / "research" / "qc_bt_standard_config.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    experiment_id = os.environ.get("EXPERIMENT_ID", "").strip()
    if not experiment_id or not re.fullmatch(r"[A-Za-z0-9._-]+", experiment_id):
        print("Invalid or missing EXPERIMENT_ID", file=sys.stderr)
        return 2

    out = ROOT / "results" / "qc-bt" / experiment_id
    out.mkdir(parents=True, exist_ok=True)

    user = os.environ.get("QC_USER_ID", "").strip()
    token = os.environ.get("QC_API_TOKEN", "").strip()
    start = os.environ.get("QC_START", "2025-09-01")
    end = os.environ.get("QC_END", "2026-08-28")
    notes = os.environ.get("NOTES", "")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    manifest = {
        "experiment_id": experiment_id,
        "verification_level": "QC_CLOUD_BT",
        "engine": "QuantConnect Lean Cloud",
        "git_sha": git_sha,
        "dataset_id": "QC_CFD_XAUUSD_QUOTEBAR",
        "strategy_sha256": sha256_file(ALGO) if ALGO.exists() else None,
        "config_sha256": sha256_file(CONFIG) if CONFIG.exists() else None,
        "seed": int(os.environ.get("SEED", "42")),
        "start_date": start,
        "end_date": end,
        "notes": notes,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "started_at_utc": utc_now(),
        "status": "STARTED",
        "not_nautilus": True,
        "not_raw_bidask_tick": True,
        "display_basis": {"initial": 1000, "horizon": "21 business days"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if ALGO.exists():
        (out / "xauusd_hour_cloud.py").write_text(ALGO.read_text(encoding="utf-8"), encoding="utf-8")

    if not user or not token:
        manifest["status"] = "FAIL_CLOSED_NO_CREDENTIALS"
        manifest["finished_at_utc"] = utc_now()
        manifest["how_to"] = [
            "Request API token at https://www.quantconnect.com/account",
            "Set GitHub Actions secrets QC_USER_ID and QC_API_TOKEN",
            "Or paste research/qc/xauusd_hour_cloud.py into the QC web IDE",
            "Paid Organization seat is required for Lean CLI cloud commands",
        ]
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 2

    cred_dir = Path.home() / ".lean"
    cred_dir.mkdir(parents=True, exist_ok=True)
    (cred_dir / "credentials").write_text(json.dumps({"user-id": user, "api-token": token}), encoding="utf-8")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "lean"], cwd=ROOT)
    if not (ROOT / "lean.json").exists():
        subprocess.check_call(["lean", "init"], cwd=ROOT)

    project = "QC-XAUUSD-Hour-Cloud"
    project_dir = ROOT / project
    if not project_dir.exists():
        subprocess.check_call(["lean", "create-project", project, "--language", "python"], cwd=ROOT)
    (project_dir / "main.py").write_text(ALGO.read_text(encoding="utf-8"), encoding="utf-8")

    cmd = [
        "lean", "cloud", "backtest", project, "--push",
        "--parameter", f"experiment_id:{experiment_id}",
        "--parameter", f"start_date:{start}",
        "--parameter", f"end_date:{end}",
    ]
    log_path = out / "lean_cloud.log"
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=1800)
        log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
        kpi = None
        for line in (proc.stdout or "").splitlines():
            if "QC_KPI_JSON=" in line:
                raw = line.split("QC_KPI_JSON=", 1)[1].strip()
                try:
                    kpi = json.loads(raw)
                except json.JSONDecodeError:
                    kpi = {"raw": raw}
        if kpi:
            kpi["run_id"] = os.environ.get("GITHUB_RUN_ID")
            kpi["sha"] = git_sha
            kpi["experiment_id"] = experiment_id
            (out / "kpi.json").write_text(json.dumps(kpi, indent=2), encoding="utf-8")
        manifest["status"] = "QC_CLOUD_COMPLETED" if proc.returncode == 0 else "QC_CLOUD_FAILED"
        manifest["lean_exit"] = proc.returncode
        manifest["finished_at_utc"] = utc_now()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0 if proc.returncode == 0 else 1
    except Exception as exc:
        manifest["status"] = "QC_CLOUD_EXCEPTION"
        manifest["error"] = str(exc)
        manifest["finished_at_utc"] = utc_now()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
