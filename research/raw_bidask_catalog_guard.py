from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
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
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    args = ap.parse_args()

    catalog = Path(args.catalog)
    manifest_path = Path(args.manifest) if args.manifest else catalog / "catalog_manifest.json"
    required = [s.upper() for s in args.symbols]
    errors: list[str] = []

    if not catalog.exists():
        errors.append(f"catalog missing: {catalog}")
    if not manifest_path.exists():
        errors.append(f"manifest missing: {manifest_path}")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE":
            errors.append("catalog manifest status must be COMPLETE")
        if manifest.get("data_kind") != "RAW_BIDASK":
            errors.append("catalog manifest data_kind must be RAW_BIDASK")
        if manifest.get("ohlc_resample_used") is not False:
            errors.append("ohlc_resample_used must be false")
        present = set(manifest.get("symbols", []))
        for symbol in required:
            if symbol not in present:
                errors.append(f"missing symbol: {symbol}")
        stats = manifest.get("stats", {})
        for symbol in required:
            if int(stats.get(symbol, {}).get("ticks", 0)) <= 0:
                errors.append(f"no raw ticks recorded for: {symbol}")

    if errors:
        raise SystemExit("RAW_BIDASK_FAIL_CLOSED\n" + "\n".join(errors))

    out = {
        "status": "RAW_BIDASK_CATALOG_OK",
        "catalog": str(catalog),
        "symbols": required,
        "timeframes": TFS,
        "manifest_sha256": sha256_file(manifest_path),
        "rule": "No OHLC-resample fallback permitted",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
