# QuoteTick acquisition gate
Dukascopy documents daily tick BI5 as 20-byte big-endian records: millisecond-of-day, ask, bid, ask-volume, bid-volume.
For XAUUSD the commodity price scale must be independently verified before decoding; do not silently assume an FX scale.

This branch intentionally rejects the existing BID/ASK minute-candle cache as sub-second evidence.
A run may proceed only after:
1. daily XAUUSD tick object is fetched;
2. decoded prices pass plausible XAUUSD bid/ask sanity checks;
3. timestamps are monotonic millisecond UTC;
4. ask >= bid;
5. the raw file contains enough distinct sub-second timestamps;
6. provenance/date/hash are written beside KPI output.
