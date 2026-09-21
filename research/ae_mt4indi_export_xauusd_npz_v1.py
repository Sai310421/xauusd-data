"""Adapter: canonical Nautilus catalog -> compact XAUUSD Bid/Ask NPZ for AE 4Engine research."""
import argparse, numpy as np
from pathlib import Path
from nautilus_trader.persistence.catalog import ParquetDataCatalog
def main():
    p=argparse.ArgumentParser(); p.add_argument("--catalog",required=True); p.add_argument("--out",required=True); a=p.parse_args()
    cat=ParquetDataCatalog(a.catalog)
    # Catalog query API differs across Nautilus releases; use generic QuoteTick class query.
    from nautilus_trader.model.data import QuoteTick
    ticks=cat.query(QuoteTick)
    rows=[x for x in ticks if "XAUUSD" in str(x.instrument_id)]
    if not rows: raise SystemExit("No XAUUSD QuoteTick rows in canonical catalog")
    ts=np.fromiter((int(x.ts_event) for x in rows),dtype=np.int64)
    bid=np.fromiter((float(x.bid_price) for x in rows),dtype=float)
    ask=np.fromiter((float(x.ask_price) for x in rows),dtype=float)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.out,ts_ns=ts,bid=bid,ask=ask)
    print("rows",len(rows),"out",a.out)
if __name__=="__main__": main()
