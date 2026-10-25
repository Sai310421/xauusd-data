# AMOS Nautilus Raw Tick CI (research only)
This provides a non-GitHub-Actions CI path using CircleCI and a pinned NautilusTrader 1.230.0 Docker environment.
- Input: Dukascopy XAUUSD genuine Bid/Ask QuoteTicks; never OHLC fallback.
- Research-only A01/B01/C01 script currently uses virtual Bid/Ask fills, NOT Nautilus order matching. Results are provisional, not validated execution KPIs.
- Run locally: docker build -f ci/nautilus/Dockerfile -t amos-nautilus-abc . && docker run --rm -v "$PWD/results/amos-abc:/workspace/results/amos-abc" amos-nautilus-abc
- CircleCI requires repository authorization and available runner minutes; committing this config does NOT confirm CircleCI connection or execution.
- No live orders.
