# AMOS Dual-Arena v1

## Purpose

Separate two different questions which should not be optimized by one leaderboard:

1. **Arena A — EDGE Discovery League**: Which candidate contains the most reusable, durable trading edge?
2. **Arena B — Profit Championship**: Which agent/strategy actually grows capital the most under common execution assumptions?

Winners are not treated as final truth. They are routed back into Math EDGE Diagnostic for decomposition and re-testing.

## Flow

```text
BT runners / extracted EAs / GitHub bots / G75 / ICT-Fib / NOTE-HFT
        |
        v
standard arena_result.json
        |
        +--> Arena A: reusable EDGE quality
        |
        +--> Arena B: absolute profit / risk-adjusted / Prop / Broker leagues
        |
        v
leaderboards
        |
        v
feedback_queue.json
        |
        v
Math EDGE Diagnostic
(direction/timing/exit/cost/regime/tail/sizing/recovery)
        |
        v
EDGE Library -> new candidates -> BT
```

## Arena A score

Arena A intentionally prevents raw profit from dominating. v1 weights:

- Durability (OOS + cost + regime retention): 30%
- Profit factor quality: 15%
- Drawdown quality: 15%
- Ruin/survival: 10%
- Cost retention: 10%
- Regime pass rate: 10%
- Discovery efficiency: 5%
- Trade-count confidence: 5%

The formula is implemented in `research/amos_dual_arena/core.py` and is deterministic.

## Arena B leagues

Five outputs are produced in parallel:

- `arena_b_absolute_profit`: highest net profit / final capital growth
- `arena_b_risk_adjusted`: return relative to DD, with PF/survival modifiers
- `arena_b_prop_dd5`: MaxDD <= 5% and ruin probability <= 1%
- `arena_b_broker_dd20`: MaxDD <= 20% and ruin probability <= 5%
- Arena A remains separate as `arena_a_edge_discovery`

This allows an aggressive strategy to win the absolute-profit championship while a different strategy wins the durable or Prop championship.

## Standard result contract

Every real BT runner should emit one JSON result using `emit_result.py` or an equivalent adapter.

Required core fields:

```json
{
  "schema_version": "amos.arena.v1",
  "candidate_id": "20260908-g75-gpt-xau-m1",
  "agent": "gpt",
  "strategy": "g75",
  "symbol": "XAUUSD",
  "timeframe": "M1",
  "initial_equity": 1000.0,
  "final_equity": 2400.0,
  "return_pct": 140.0,
  "max_dd_pct": 4.2,
  "profit_factor": 2.1,
  "win_rate_pct": 67.0,
  "trades": 12000,
  "ruin_probability": 0.0,
  "oos_retention": 0.84,
  "cost_retention": 0.79,
  "regime_pass_rate": 0.82,
  "discovery_efficiency": 0.35,
  "verification_level": "NAUTILUS_RAW_TICK"
}
```

Retention values are normalized to 0..1. Missing retention evidence is intentionally scored as zero rather than guessed.

## GitHub Actions

`.github/workflows/amos-dual-arena.yml` can be started manually or called as a reusable workflow. It:

1. runs unit tests,
2. scans JSON results,
3. builds all leaderboards,
4. builds `feedback_queue.json`,
5. uploads the complete Arena artifact.

## Integration rule

Existing backtest workflows remain authoritative for execution. Do not replace them with the Arena workflow. Instead, add a standardized result-emission step after a runner has produced verified metrics, then score those results with Dual-Arena.

The current `AE Multi-Chat Nautilus BT Hub` is a compatibility/smoke gate; smoke manifests without performance metrics are ignored by design. Only actual strategy results are ranked.

## Next adapters

Priority order for connecting live AMOS assets:

1. G75 frozen baseline / variants
2. NOTE-HFT
3. Fib / ICT / VGRSI family
4. Recovery / Hedge Lock variants
5. external EA / GitHub EDGE harvest candidates
6. 6 symbols x M1/M5/M15 Nautilus batch output

Once those adapters emit `amos.arena.v1`, the same leaderboard and feedback loop works without strategy-specific scoring code.
