"""Fail-closed raw QuoteTick quality gate; never accepts OHLC stand-ins."""
import argparse
import json
from pathlib import Path
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True)
    ap.add_argument("--symbol",default="XAUUSD")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    catalog=ParquetDataCatalog(args.catalog)
    instruments=[i for i in catalog.instruments() if i.id.symbol.value.replace("/","")==args.symbol]
    if len(instruments)!=1:
        raise SystemExit(f"RAW_QUOTETICK_FAIL instrument_count={len(instruments)}")
    ticks=catalog.query(data_cls=QuoteTick,identifiers=[instruments[0].id.value])
    if len(ticks)<1000:
        raise SystemExit(f"RAW_QUOTETICK_FAIL insufficient_ticks={len(ticks)}")
    previous=None
    for index,tick in enumerate(ticks):
        timestamp=int(tick.ts_event)
        if previous is not None and timestamp<previous:
            raise SystemExit(f"RAW_QUOTETICK_FAIL non_monotonic index={index}")
        if tick.ask_price<tick.bid_price:
            raise SystemExit(f"RAW_QUOTETICK_FAIL crossed_quote index={index}")
        previous=timestamp
    result={"status":"RAW_QUOTETICK_QUALITY_OK","data_class":"QuoteTick","ohlc_fallback":False,"symbol":args.symbol,"raw_ticks":len(ticks),"monotonic_ts_event":True,"ask_gte_bid":True,"first_ts_event":int(ticks[0].ts_event),"last_ts_event":int(ticks[-1].ts_event)}
    output=Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
