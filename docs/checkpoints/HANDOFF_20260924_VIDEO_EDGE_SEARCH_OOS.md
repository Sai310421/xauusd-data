# HANDOFF — Video Procedure Statistical EDGE Search → Frozen OOS Confirmation

Updated: 2026-09-24 05:44 JST

## 1. Goal

Reproduce the uploaded video's strategy-development procedure inside AMOS without inventing unknown video parameters:

1. sweep many explicit candidate lanes;
2. measure trade-result distributions;
3. filter statistically positive candidates;
4. run Monte Carlo risk comparisons;
5. freeze the selected candidate before OOS;
6. confirm it on a later unseen Raw Bid/Ask period;
7. only after statistical selection, interpret the chart pattern/market structure.

The exact meanings of the video's P0/P1/P2/P3 are not visible in the source material. They must remain UNKNOWN until source evidence is available.

## 2. Repository / branch

Repository: `Sai310421/xauusd-data`

Active branch:
`feature/video-statistical-edge-search-v1.0-20260921`

Current branch head observed at handoff:
`2a27ec257894cea3bd943880b4aec860504912e2`

Do not overwrite the canonical Raw Bid/Ask lane with OHLC approximations.

## 3. Discovery implementation

Files:
- `research/video_procedure_edge_search_v1.py`
- `.github/workflows/video-procedure-edge-search-v1.yml`

Discovery data:
- XAUUSD
- Raw Bid/Ask QuoteTick
- M1 / M5 / M15
- 10 explicit AMOS EDGE definitions
- LONG / SHORT
- threshold 0.25 / 0.50 / 0.75
- TP/SL ATR combinations: 0.75/0.75, 1.0/0.75, 1.5/1.0
- total lanes: 540

Discovery filter:
- N >= 100
- PF >= 1.20
- EV_R > 0

Discovery run:
`35603417081`

Frozen candidate selected from discovery:
- TF: M15
- EDGE: `T1_EMA21`
- direction: LONG
- threshold: 0.50
- TP: 1.5 ATR
- SL: 1.0 ATR
- horizon: 4 M15 bars
- discovery N: 245
- discovery PF: 1.2747350094799224
- discovery EV_R: 0.09086505499711309

Important: this is a discovery candidate, not yet a validated production EDGE.

## 4. Frozen OOS confirmation

Files:
- `research/video_procedure_edge_confirmation_v1_1.py`
- `.github/workflows/video-edge-frozen-oos-v1.1.yml`

OOS period:
- start: 2026-08-27
- 21 days
- XAUUSD only
- Raw Bid/Ask only
- separate catalog: `catalog/video_edge_oos`
- cache key currently named `raw-bidask-duka-6sym-2026-08-27-21d-v1`; despite the legacy name, the current workflow explicitly builds/guards XAUUSD only.

Frozen-candidate rule:
`FROZEN_BEFORE_OOS_NO_RETUNING_FROM_OOS`

OOS base gate:
- N >= 100
- PF >= 1.20
- EV_R > 0
- 1.5x spread-stress PF >= 1.0
- 1.5x spread-stress EV_R > 0

Additional diagnostics:
- spread stress: 1.5x and 2.0x
- neighbor grid:
  - threshold 0.4 / 0.5 / 0.6
  - TP 1.25 / 1.5 / 1.75 ATR
  - SL 0.75 / 1.0 / 1.25 ATR
- Monte Carlo:
  - 2,000 paths
  - 400 trades
  - risk 0.5% / 1% / 2%

## 5. Current runtime status

Previous OOS run:
`35620376811`
- result: CANCELLED
- elapsed to cancellation: about 3 hours
- it did not produce a confirmed OOS result.

The workflow was subsequently changed so the unseen-period builder explicitly requests XAUUSD only:
`build_raw_bidask_catalog_duka.py ... --symbols XAUUSD --fresh`

Current run:
`35917899412`
- status at 2026-09-24 05:44 JST: IN_PROGRESS
- checkout/setup/install/cache restore: SUCCESS
- current step: `Build unseen XAUUSD Raw BidAsk period`
- confirmation and artifact upload have NOT run yet.

Therefore: system wiring is intact, but the OOS EDGE result is still pending. Do not report PASS/FAIL, PF, WR, EV or Monte Carlo OOS values until this run reaches the confirmation step successfully.

## 6. What the next operator should do

First inspect run `35917899412`.

If SUCCESS:
- read the uploaded `video-edge-oos-...` artifact / summary;
- report exact OOS N / WR / PF / EV_R / median_R;
- report 1.5x and 2.0x spread-stress results;
- report neighbor robustness counts;
- report Monte Carlo median terminal return, median MaxDD and P90 MaxDD for 0.5%, 1%, 2% risk;
- preserve the frozen candidate; do not retune using the OOS result;
- fileize a v1.1 checkpoint.

If it fails/cancels during Raw catalog build:
- inspect the build logs first;
- do not change the strategy;
- fix only data acquisition/cache/runtime;
- keep the same unseen period and frozen candidate;
- rerun with identical strategy parameters.

If it reaches confirmation but FAILs the gate:
- record the failure as valid OOS evidence;
- do not optimize the frozen candidate against this OOS set;
- return to discovery only in a new experiment/version with a new future holdout.

## 7. Safety / scientific boundaries

- Raw Bid/Ask only; no OHLC fallback.
- Discovery and OOS must remain separated.
- Do not turn OOS into another optimization set.
- Video-unknown P0/P1/P2/P3 semantics remain unknown.
- Monte Carlo resamples observed trade outcomes; it does not create independent market evidence.
- `production_weighting_allowed=false`
- `execution_allowed=false`
- No broker/live authority is granted by this research path.

## 8. Related Nautilus baseline

Canonical Raw Bid/Ask repository:
`Sai310421/xauusd-data`

Earlier Chris/AMOS Raw WFO checkpoint:
`checkpoint/chrispathway-nautilus-raw-wfo-v1.2-20260921`

The Video EDGE search is a separate statistical-discovery lane but must reuse the same Raw Bid/Ask evidence discipline.
