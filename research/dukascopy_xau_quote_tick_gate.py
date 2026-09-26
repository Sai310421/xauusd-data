#!/usr/bin/env python3
import argparse,csv,datetime as dt,lzma,struct,urllib.request
from pathlib import Path
REC=struct.Struct(">IIIff")
def main():
 p=argparse.ArgumentParser();p.add_argument("--date",required=True);p.add_argument("--scale",type=float,required=True);p.add_argument("--out",required=True);a=p.parse_args()
 d=dt.date.fromisoformat(a.date); m=d.month-1
 url=f"https://datafeed.dukascopy.com/datafeed/XAUUSD/{d.year}/{m:02d}/{d.day:02d}/BID_candles_min_1.bi5"
 # Deliberate guard: minute candles are NOT acceptable as QuoteTick.
 raise SystemExit("FAIL-CLOSED: legacy candle URL cannot be used for sub-second QuoteTick BT; wire verified daily tick object before execution")
if __name__=="__main__": main()
