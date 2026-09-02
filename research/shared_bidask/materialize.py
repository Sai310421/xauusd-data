from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lean_export import hour_quote_bars, write_hour_quote_bars, write_quote_tick_day, write_symbol_properties
from source import SYMBOLS, fetch_day, iter_days


def sha256_tree(root: Path) -> str:
    if not root.exists():
        return ""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in {"shared_manifest.json", "catalog_manifest.json"}:
            continue
        h.update(str(p.relative_to(root)).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07-27")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD"])
    ap.add_argument("--lean-root", default="lean_data")
    ap.add_argument("--catalog", default="catalog/raw_bidask")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--write-nautilus", action="store_true")
    ap.add_argument("--no-lean", action="store_true")
    args = ap.parse_args()

    write_lean = not args.no_lean
    selected = []
    for s in args.symbols:
        s = s.upper()
        if s not in SYMBOLS:
            raise SystemExit(f"unsupported symbol: {s}")
        if s not in selected:
            selected.append(s)

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    lean_root = Path(args.lean_root)
    catalog = Path(args.catalog)
    if write_lean:
        lean_root.mkdir(parents=True, exist_ok=True)
        write_symbol_properties(lean_root)

    stats = {}
    hour_store = {s: {} for s in selected}
    missing = []

    for symbol in selected:
        meta = SYMBOLS[symbol]
        total = 0
        written_days = 0
        http_counts = {}
        for day in iter_days(start, args.days):
            rows, status_counts = fetch_day(symbol, meta["scale"], day, workers=args.workers)
            for k, v in status_counts.items():
                http_counts[k] = http_counts.get(k, 0) + v
            if not rows:
                continue
            if write_lean:
                write_quote_tick_day(
                    lean_root,
                    meta["lean_security_type"],
                    meta["lean_market"],
                    symbol,
                    day,
                    rows,
                )
                hour_store[symbol][day.strftime("%Y%m%d")] = hour_quote_bars(rows)
            total += len(rows)
            written_days += 1
        if write_lean and hour_store[symbol]:
            write_hour_quote_bars(
                lean_root,
                meta["lean_security_type"],
                meta["lean_market"],
                symbol,
                hour_store[symbol],
            )
        stats[symbol] = {
            "ticks": total,
            "written_days": written_days,
            "http_status_counts": http_counts,
            "lean_security_type": meta["lean_security_type"],
            "lean_market": meta["lean_market"],
        }
        if total <= 0:
            missing.append(symbol)

    nautilus_status = "SKIPPED"
    if args.write_nautilus:
        builder = HERE.parent / "build_raw_bidask_catalog_duka.py"
        cmd = [
            sys.executable,
            str(builder),
            "--start",
            args.start,
            "--days",
            str(args.days),
            "--catalog",
            str(catalog),
            "--workers",
            str(args.workers),
            "--symbols",
            *selected,
        ]
        rc = subprocess.call(cmd)
        nautilus_status = "COMPLETE" if rc == 0 else "FAILED"
        if rc != 0:
            missing.append("NAUTILUS_CATALOG")

    if not write_lean:
        lean_status = "SKIPPED"
    elif any(stats[s]["ticks"] <= 0 for s in selected):
        lean_status = "INCOMPLETE"
    else:
        lean_status = "COMPLETE"

    status = "COMPLETE" if not missing else "INCOMPLETE"
    manifest = {
        "status": status,
        "data_kind": "RAW_BIDASK",
        "source": "Dukascopy BI5 QuoteTick Bid/Ask",
        "ohlc_resample_used": False,
        "symbols": selected,
        "start": args.start,
        "days": args.days,
        "end_exclusive": (start + dt.timedelta(days=args.days)).date().isoformat(),
        "artifacts": {
            "nautilus_catalog": {
                "path": str(catalog),
                "status": nautilus_status,
                "sha256": sha256_tree(catalog) if catalog.exists() else "",
            },
            "lean_data": {
                "path": str(lean_root),
                "status": lean_status,
                "sha256": sha256_tree(lean_root) if lean_root.exists() else "",
            },
        },
        "verification_split": {
            "NAUTILUS_BT": "catalog/raw_bidask only",
            "LEAN_LOCAL_BT": "lean_data only",
            "QC_CLOUD_BT": "QuantConnect official CFD, not this tree",
        },
        "stats": stats,
        "missing": missing,
    }
    out_dir = Path("results/shared-bidask")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    (out_dir / "shared_manifest.json").write_text(payload, encoding="utf-8")
    if write_lean:
        (lean_root / "shared_manifest.json").write_text(payload, encoding="utf-8")
    print(payload)
    if missing:
        raise SystemExit("SHARED_BIDASK_INCOMPLETE: " + ",".join(missing))


if __name__ == "__main__":
    main()
