# Validation protocol

1. Formula tests for v1 against the expanded expression, bounds, units, and the λ3 leak.
2. Cash conservation across fills, transfers, and pool payments. Short marks use the ask.
3. Quote gate: at least 1000 rows, ask ≥ bid, non-decreasing timestamps, both prices present. OHLC is a failure.
4. Walk-forward: test indexes start after train plus embargo. No market score until a catalog exists.
5. Monte Carlo: resample aligned blocks. An independent shuffle of the hedge series is implemented only to prove it destroys correlation. Official code paths do not call it on a claim.
6. Hypothesis labels stay BLOCKED where the input is missing.

A synthetic structural overlay (Gaussian months, zero main alpha, hedge = −β × main − cost) is allowed as a model of the conflict. Its JSON label is `SYNTHETIC_STRUCTURAL_NOT_MARKET`.
