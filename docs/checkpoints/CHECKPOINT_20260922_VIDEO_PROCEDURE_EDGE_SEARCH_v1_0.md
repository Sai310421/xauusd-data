# CHECKPOINT — Video Procedure Statistical EDGE Search v1.0

Date: 2026-09-22 JST

## Purpose
Reproduce the uploaded video's supported procedure: sweep candidate lanes first, measure distributions, filter statistically, then run Monte Carlo risk comparison. Unknown P0/P1/P2/P3 semantics are not invented.

## Discovery evidence
- Workflow: Video Procedure Statistical EDGE Search v1
- Run ID: 35603417081
- Result: SUCCESS
- SHA: be34947f20914aede2beb99b892523b9126428e6
- XAUUSD Raw Bid/Ask QuoteTicks: 3,933,767
- Lanes: 540
- Gate: N >= 100, PF >= 1.20, EV_R > 0
- Eligible: 1

## Frozen candidate
- TF: M15
- EDGE: T1_EMA21
- Direction: LONG
- score threshold: 0.50
- TP: 1.50 ATR
- SL: 1.00 ATR
- horizon: 60 minutes
- N: 245
- WR: 48.9796%
- PF: 1.274735
- EV: +0.090865 R/trade
- median trade: -0.028941 R

## Monte Carlo — 2,000 paths x 400 trades
| Risk/trade | Median terminal return | Median MaxDD | P90 MaxDD |
|---:|---:|---:|---:|
| 0.5% | 19.6464% | 5.3047% | 8.3679% |
| 1.0% | 41.9905% | 10.4047% | 16.2161% |
| 2.0% | 95.3925% | 20.0669% | 30.3056% |

## Interpretation
This is a discovery candidate, not a promoted production EDGE. Only one of 540 lanes passed the initial gate, and the candidate was selected using the discovery period. It therefore requires a truly unseen period before promotion.

The negative median trade despite positive EV/PF means the edge is right-tail dependent; robustness and cost sensitivity matter.

## Next gate already launched
Frozen candidate confirmation uses an unseen Raw Bid/Ask period:
- start: 2026-08-27
- days: 21
- candidate parameters frozen before OOS
- 1.5x and 2.0x spread stress
- 27 neighboring parameter lanes for robustness diagnostics only; OOS does not retune the candidate
- workflow run: 35620376811

## Safety
production_weighting_allowed=false
execution_allowed=false
