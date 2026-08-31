from __future__ import annotations

import json
import lzma
import struct
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Native probe only. OHLC resampling is intentionally prohibited here.
REC = struct.Struct(">5If")
SCALE = 1000.0
HOST = "https://datafeed.dukascopy.com/datafeed"
OUT = Path("results/native_mtf_probe")


def fetch(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return int(r.status), r.read()
    except Exception as e:
        return 0, str(e).encode()


def parse(blob: bytes, origin: datetime) -> list[dict]:
    raw = lzma.decompress(blob)
    rows = []
    for i in range(0, len(raw) - REC.size + 1, REC.size):
        sec, o, c, lo, hi, vol = REC.unpack_from(raw, i)
        ts = origin.timestamp() + int(sec)
        o, c, lo, hi = o / SCALE, c / SCALE, lo / SCALE, hi / SCALE
        if o <= 0 or hi < lo:
            continue
        rows.append({"t": int(ts), "o": o, "h": hi, "l": lo, "c": c, "v": float(vol)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y, m = 2026, 5  # June 2026; Dukascopy month path is zero-based.
    origin = datetime(y, m + 1, 1, tzinfo=timezone.utc)
    report = {"policy": "NATIVE_ONLY_NO_RESAMPLE", "period": "2026-06", "checks": []}

    for tf, kind in (("M15", "min_15"), ("M30", "min_30")):
        side_rows = {}
        for side in ("BID", "ASK"):
            url = f"{HOST}/XAUUSD/{y}/{m:02d}/{side}_candles_{kind}.bi5"
            status, blob = fetch(url)
            item = {"tf": tf, "side": side, "url": url, "http": status, "bytes": len(blob)}
            try:
                rows = parse(blob, origin) if status == 200 else []
                item["rows"] = len(rows)
                item["first"] = rows[0]["t"] if rows else None
                item["last"] = rows[-1]["t"] if rows else None
                side_rows[side] = rows
            except Exception as e:
                item["parse_error"] = repr(e)
                side_rows[side] = []
            report["checks"].append(item)

        bid = {r["t"] for r in side_rows.get("BID", [])}
        ask = {r["t"] for r in side_rows.get("ASK", [])}
        report[f"{tf}_paired_rows"] = len(bid & ask)
        report[f"{tf}_native_ok"] = bool(bid and ask and (bid & ask))

    (OUT / "status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
