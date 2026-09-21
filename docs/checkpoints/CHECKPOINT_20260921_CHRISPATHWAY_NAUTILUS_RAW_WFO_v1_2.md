# CHECKPOINT — Chris Pathway → AMOS Nautilus Raw BidAsk WFO v1.2

Date: 2026-09-21

## Status
COMPLETE for the research-validation wiring checkpoint.

This checkpoint connects the Chris Pathway five-stage AMOS research upgrade to the canonical `Sai310421/xauusd-data` Raw Bid/Ask catalog lane without OHLC fallback.

## Files
- `research/chris_quant_raw_bidask_wfo_v1_2.py`
- `tests/test_chris_quant_raw_bidask_wfo_v1_2.py`
- `.github/workflows/chrispathway-nautilus-raw-wfo-v1.2.yml`

## Canonical data/runtime
- NautilusTrader: 1.230.0
- Data: Raw Bid/Ask QuoteTick only
- OHLC resample: false
- Symbols: XAUUSD, EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF
- TF research views: M1, M5, M15
- Alignment: common active range; a bucket with no quote update uses carry-forward price semantics => zero return. Leading/trailing gaps are excluded.
- Walk-forward: chronological train -> frozen OOS
- Default train/test: 240 / 80 observations
- Max folds: 12 per TF

## Five-stage AMOS route
Raw QuoteTick
→ Raw-derived return buckets
→ Bayesian basket search
→ Regime + Student-t fat-tail stress
→ First Passage
→ CVaR
→ Wasserstein diagnostic stress
→ Crystal Ball research context
→ AE DD tightening cap

## Evidence
- Git SHA: `aa6c18537515c0cdb3d3d1bc4f28060f45fee30b`
- Workflow: `Chris Pathway AMOS Raw BidAsk WFO v1.2`
- Run ID: `35550971897`
- Run result: SUCCESS
- Unit gate: 5 passed in 1.30s
- Artifact ID: `10618481713`
- Artifact: `chris-amos-raw-wfo-chris-wfo-v12-35550971897`

Raw QuoteTick counts:
- XAUUSD: 3,933,767
- EURUSD: 798,612
- GBPUSD: 1,138,183
- USDJPY: 1,863,124
- AUDUSD: 573,155
- USDCHF: 1,029,514

## OOS research metrics
| TF | aligned points | folds | mean OOS Sharpe-like | worst OOS MaxDD % | mean OOS CVaR % | mean tail share | mean recovery share | mean AE cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 24,712 | 12 | 0.035667 | 0.133895 | 0.085729 | 0.000000 | 0.000000 | 0.000000 |
| M5 | 4,943 | 12 | 0.085508 | 0.261329 | 0.174526 | 0.000000 | 0.000000 | 0.000000 |
| M15 | 1,647 | 12 | 0.015189 | 0.420183 | 0.246351 | 0.000000 | 0.0015625 | 0.0015625 |

## Interpretation boundary
These are research-transform OOS metrics, not trade-system WR/PF/Net/Monthly21 metrics. This runner does not place simulated orders.

The current fixed First Passage levels (+1% recovery / -3% tail in the research packet) are too wide for most 40-step M1/M5/M15 OOS windows in this dataset, producing near-zero hit shares. This is evidence that the next checkpoint should calibrate first-passage boundaries to empirical volatility/horizon scale rather than treating the current shares as useful probabilities.

Scenario shares are not calibrated future probabilities. The Wasserstein term is the AMOS diagnostic proxy `CVaR + radius`, not a complete Wasserstein DRO solver.

## Safety
- `production_weighting_allowed=false`
- `execution_allowed=false`
- Crystal Ball route is context-only.
- AE DD route is tightening-only and cannot grant approval.

## Next checkpoint
v1.3: volatility/horizon-scaled First Passage boundary calibration + baseline-vs-upgrade OOS delta report, while preserving fixed-boundary results as the v1.2 reference.
