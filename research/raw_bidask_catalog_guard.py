from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
TFS = ["M1", "M5", "M15"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="catalog/raw_bidask")
    ap.add_argument("--manifest", default="catalog/raw_bidask/catalog_manifest.json")
    args = ap.parse_args()

    catalog = Path(args.catalog)
    manifest_path = Path(args.manifest)
    errors: list[str] = []

    if not catalog.exists():
        errors.append(f"catalog missing: {catalog}")
    if not manifest_path.exists():
        errors.append(f"manifest missing: {manifest_path}")

    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("data_kind") != "RAW_BIDASK":
            errors.append("catalog manifest data_kind must be RAW_BIDASK")
        present = set(manifest.get("symbols", []))
        for symbol in SYMBOLS:
            if symbol not in present:
                errors.append(f"missing symbol: {symbol}")

    if errors:
        raise SystemExit("RAW_BIDASK_FAIL_CLOSED\n" + "\n".join(errors))

    out = {
        "status": "RAW_BIDASK_CATALOG_OK",
        "catalog": str(catalog),
        "symbols": SYMBOLS,
        "timeframes": TFS,
        "manifest_sha256": sha256_file(manifest_path),
        "rule": "No OHLC-resample fallback permitted",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
