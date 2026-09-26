#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
from nautilus_trader.persistence.catalog import ParquetDataCatalog
def main():
 p=argparse.ArgumentParser();p.add_argument("--catalog",required=True);p.add_argument("--out",required=True);a=p.parse_args()
 c=ParquetDataCatalog(a.catalog); inst=next((x for x in c.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"),None)
 if inst is None: raise SystemExit("RAW_BIDASK_FAIL_CLOSED: XAUUSD missing")
 ticks=c.query_quote_ticks(identifiers=[inst.id.value])
 if len(ticks)<100000: raise SystemExit(f"RAW_BIDASK_FAIL_CLOSED: ticks={len(ticks)}")
 o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True)
 with o.open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["timestamp","bid","ask"])
  for t in ticks:w.writerow([int(t.ts_event)/1e9,t.bid_price.as_double(),t.ask_price.as_double()])
 print(f"XAUUSD_RAW_TICKS={len(ticks)} instrument={inst.id.value} out={o}")
if __name__=="__main__":main()
