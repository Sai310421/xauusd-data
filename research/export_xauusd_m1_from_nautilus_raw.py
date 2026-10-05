from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    def _qqt(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _qqt

def f(x):
    try:
        return float(x)
    except Exception:
        return float(str(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-days", type=int, default=30)
    a = ap.parse_args()

    cat = ParquetDataCatalog(a.catalog)
    inst = next((x for x in cat.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD"), None)
    if inst is None:
        raise SystemExit("XAUUSD missing from Nautilus catalog")

    ticks = cat.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:
        raise SystemExit("No QuoteTick data")

    rows = []
    for t in ticks:
        ts = pd.Timestamp(int(t.ts_event), unit="ns", tz="UTC")
        rows.append((ts, f(t.bid_price), f(t.ask_price)))

    df = pd.DataFrame(rows, columns=["datetime","bid","ask"]).drop_duplicates("datetime").sort_values("datetime")
    if a.limit_days > 0:
        cutoff = df.datetime.max() - pd.Timedelta(days=a.limit_days)
        df = df[df.datetime >= cutoff].copy()

    df["mid"] = (df.bid + df.ask) / 2.0
    df["bucket"] = df.datetime.dt.floor("1min")
    m1 = df.groupby("bucket", sort=True).agg(
        open=("mid","first"),
        high=("mid","max"),
        low=("mid","min"),
        close=("mid","last"),
        volume=("mid","size"),
    ).reset_index().rename(columns={"bucket":"datetime"})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m1.to_csv(out, index=False)
    print({
        "instrument": inst.id.value,
        "raw_ticks": len(df),
        "m1_bars": len(m1),
        "start": str(df.datetime.min()),
        "end": str(df.datetime.max()),
        "output": str(out),
    })

if __name__ == "__main__":
    main()
