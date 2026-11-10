# RAGE Phase 1 — Foundation Audit and Boundary Contract

**Audit date:** 2026-11-10 UTC  
**Status:** IMPLEMENTATION BASELINE FIXED — no live trading or order routing

## 1. Canonical locations

| Responsibility | Canonical location | Status |
|---|---|---|
| RAGE research implementation | `rage-dual-compound-hedge/` in this repository | Present |
| ProtectiveHedge baseline | `rage-dual-compound-hedge/src/rage_dch/protective_hedge.py` | Present; original formula only |
| RAGE research configuration | `rage-dual-compound-hedge/configs/research.yaml` | Present; `RAW_BIDASK`, live trading OFF |
| Execution-quality historical data | `catalog/raw_bidask` materialized by Raw BidAsk Catalog Factory | Present route; artifact/cache must be verified per run |
| Tick source | Dukascopy BI5 raw Bid/Ask -> Nautilus `QuoteTick` -> `ParquetDataCatalog` | Canonical |
| H1 sensor -> RAGE contract | `rage-dual-compound-hedge/contracts/h1_macro_signal_v1.schema.json` | Added in this phase |
| Contract validation CI | `.github/workflows/rage-phase1-contract.yml` | Added in this phase |

`AMOS-Quant-Research/amos-public-ci` is **not** the RAGE implementation source. It may be used later for independent CI orchestration only. `Sai310421/xauusd-duka-feed` is a raw-data helper, not a RAGE runtime.

## 2. Responsibility boundary

| Layer | May do | Must not do |
|---|---|---|
| USDJPY Institutional Risk AI | H1 probability, regime, drift and portfolio-risk estimation | Place orders, calculate lots, use tick fills |
| H1-to-tick adapter | Validate, expire and smooth accepted H1 signal snapshots; expose immutable state | Recalculate the model, invent a missing H1 signal, bypass STOP status |
| RAGE / ProtectiveHedge | Calculate protection allocation from validated macro state plus execution-risk inputs | Treat `portfolio_risk` as an execution-cost denominator; accept stale/invalid signals |
| Nautilus runtime | Replay/execute against raw Bid/Ask events and record fills | Replace raw quotes with OHLC or mid-price proxy |
| Broker gateway | Execute already-approved orders | Override risk gates or model status |

There is no order-routing component in this phase. Any execution remains OFF.

## 3. Data lineage fixed

```
Dukascopy BI5 raw Bid/Ask
  -> Nautilus QuoteTick
  -> ParquetDataCatalog
  -> RAGE research / Nautilus replay
```

The following are prohibited as canonical RAGE evidence: OHLC resampling, synthetic intrabar ordering, mid-price execution, or fallback when the raw catalog is missing.

A raw-data run is acceptable only when its evidence includes the catalog manifest, QuoteTick count, bid/ask integrity check, data hash, strategy/config hash, commit SHA, and Nautilus version.

## 4. Macro-risk semantics

`portfolio_risk` is a **portfolio crisis severity** on 0–100. It is not the `Risk_t` term in the original ProtectiveHedge denominator.

The later HedgeOptimizer implementation shall derive:

```
macro_crisis_score = clip(portfolio_risk / 100, 0, 1)
hedge_execution_risk = clip(cost/liquidity/volatility execution measure, 0, 1)
```

Macro crisis severity may increase hedge demand; execution risk may reduce or cap executable intensity. They are separate inputs and must remain separately logged.

## 5. Phase 1 acceptance criteria

1. The RAGE codebase and the raw Bid/Ask source are identified by exact repository paths.
2. The H1 input contract is versioned and machine-validated.
3. Invalid, stale, STOPPED, unordered, or out-of-range sensor snapshots are rejected by contract.
4. The ownership boundary explicitly prevents the sensor from placing orders.
5. A CI job validates the JSON contract on pull requests and manual dispatch.

Completion of this phase does **not** prove profitability, hedge efficacy, data availability for a future run, or broker execution. Those require later raw-tick, OOS/WFO, Monte Carlo, shadow, and broker-reality evidence.
