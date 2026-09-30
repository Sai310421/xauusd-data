# AE Universal EA Potential Maximizer — Nautilus KPI Standard v1.0

This implementation follows the supplied framework literally:

SOURCE_INGEST -> PARSE -> STRATEGY_GENOME -> NAUTILUS_BASELINE -> ABLATION -> PARAMETER_SEARCH -> NAUTILUS_PARAMETER_MAX -> A17_SINGLETONS -> A17_PAIRS/TRIPLES -> OOS -> STRESS -> ROBUST_MAX -> REPORT

## Non-negotiable rules
- Only Nautilus Raw Tick / QuoteTick measurements may become official KPI.
- Python/MT5/static/optimizer outputs remain PROXY/PREDICTED evidence.
- Three variants are always compared: ORIGINAL, PARAMETER_MAX, A17_ROBUST_MAX.
- Parameter selection must not use final OOS.
- A17 starts with singleton marginal-edge tests. Pair/triple search is blocked until singleton evidence is positive.
- Stable-region selection is preferred over one-point maxima.
- VALIDATED requires OOS + cost + spread/slippage + regime + stress gates.

## Current first subject
ML66 WalkForward + DirectionGate v5 is ingested only as a PROXY baseline. Its PF 1.248 is not promoted to official KPI until reproduced under the Nautilus Raw Tick contract.
