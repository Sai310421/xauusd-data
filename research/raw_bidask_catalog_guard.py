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


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return SYMBOLS
    names = [x.strip().upper() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in SYMBOLS]
    if unknown:
        raise SystemExit("unknown symbols: " + ",".join(unknown))
    if not names:
        raise SystemExit("--symbols resolved to empty set")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="catalog/raw_bidask")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--symbols", default=None, help="comma-separated required subset; default is all six")
    args = ap.parse_args()

    required_symbols = parse_symbols(args.symbols)
    catalog = Path(args.catalog)
    manifest_path = Path(args.manifest) if args.manifest else catalog / "catalog_manifest.json"
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
        if manifest.get("status") != "COMPLETE":
            errors.append("catalog manifest status must be COMPLETE")
        if manifest.get("ohlc_resample_used") is not False:
            errors.append("ohlc_resample_used must be false")
        present = set(manifest.get("symbols", []))
        for symbol in required_symbols:
            if symbol not in present:
                errors.append(f"missing symbol: {symbol}")
        stats = manifest.get("stats", {})
        for symbol in required_symbols:
            row = stats.get(symbol, {})
            if int(row.get("ticks", 0) or 0) <= 0:
                errors.append(f"zero quote ticks: {symbol}")

    if errors:
        raise SystemExit("RAW_BIDASK_FAIL_CLOSED\n" + "\n".join(errors))

    out = {
        "status": "RAW_BIDASK_CATALOG_OK",
        "catalog": str(catalog),
        "symbols": required_symbols,
        "timeframes": TFS,
        "manifest_sha256": sha256_file(manifest_path),
        "rule": "No OHLC-resample fallback permitted",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
