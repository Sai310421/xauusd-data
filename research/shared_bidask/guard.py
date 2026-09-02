from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/shared-bidask/shared_manifest.json")
    ap.add_argument("--require-lean", action="store_true")
    ap.add_argument("--require-nautilus", action="store_true")
    args = ap.parse_args()
    path = Path(args.manifest)
    if not path.exists():
        raise SystemExit(f"SHARED_BIDASK_FAIL_CLOSED\nmanifest missing: {path}")
    m = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if m.get("data_kind") != "RAW_BIDASK":
        errors.append("data_kind must be RAW_BIDASK")
    if m.get("ohlc_resample_used") is not False:
        errors.append("ohlc_resample_used must be false")
    arts = m.get("artifacts", {})
    if args.require_lean and arts.get("lean_data", {}).get("status") != "COMPLETE":
        errors.append("lean_data artifact is not COMPLETE")
    if args.require_nautilus and arts.get("nautilus_catalog", {}).get("status") != "COMPLETE":
        errors.append("nautilus_catalog artifact is not COMPLETE")
    if args.require_lean or args.require_nautilus:
        if m.get("status") != "COMPLETE":
            errors.append("shared manifest status must be COMPLETE")
    if errors:
        raise SystemExit("SHARED_BIDASK_FAIL_CLOSED\n" + "\n".join(errors))
    print(json.dumps({"status": "SHARED_BIDASK_OK", "manifest": str(path)}, indent=2))


if __name__ == "__main__":
    main()
