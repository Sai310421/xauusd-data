# CHECKPOINT 2026-09-21 — Video Method Real Raw EDGE Finder v1.0

## Source-grounded procedure
The uploaded video explicitly shows:
- numerical sweep first, chart pattern later;
- lane table with Median/P90/RF/No-Loss;
- filters MedRF >= 1.5, NoLoss% <= 3.0, MedProfit >= $300;
- representative lane distribution -> win rate / avg win / avg loss -> EV;
- Monte Carlo 2,000 paths x 400 trades;
- risk sweep 0.5%, 1%, 2%, 5%, 10%;
- the video's own footer labels the shown sweep SYNTHETIC.

The video does not disclose what P0/P1/P2/P3 mean or the sizing/contract basis behind dollar MedProfit.

## Real Raw implementation
Data: canonical Dukascopy Raw Bid/Ask QuoteTick catalog, no OHLC fallback.
Symbols: XAUUSD, EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF.
Timeframe: M5 numeric research view.
AMOS hypothesis family v1: post-impulse continuation.
P0/P1/P2/P3 are explicitly AMOS-defined train-only absolute-return quantiles: 50/65/80/90%.
Hold: 3 bars.
Selection: first chronological 70% discovery; final 30% untouched OOS.
Monte Carlo: 2,000 x 400, seed family 11+.

## Corrected final run
Run: 35602387761
SHA: 8760f85e16271fadb13c4b142d8025486e141f4e
Artifact: video-method-edge-video-edge-real-v1-35602387761
Artifact ID: 10639492736

Raw ticks:
- XAUUSD 3,933,767
- EURUSD 798,612
- GBPUSD 1,138,183
- USDJPY 1,863,124
- AUDUSD 573,155
- USDCHF 1,029,514

48 lanes scanned.
Discovery candidates: 4.
Strict OOS confirmed: **0**.

All four discovery candidates were EURUSD Long:
| P | IS MedRF | IS NoLoss% | IS EV(normalized) | IS N | OOS MedRF | OOS NoLoss% | OOS EV | OOS N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.381 | 1.80 | +0.04438 | 694 | -0.874 | 98.75 | -0.02931 | 260 |
| 1 | 3.989 | 0.70 | +0.05711 | 489 | -0.733 | 89.50 | -0.01845 | 183 |
| 2 | 3.233 | 2.30 | +0.05491 | 284 | -0.879 | 98.70 | -0.03340 | 100 |
| 3 | 5.060 | 0.35 | +0.08865 | 149 | -0.993 | 100.00 | -0.11628 | 38 |

## Finding
The first numerical hypothesis produced an apparently strong discovery bias on EURUSD Long, but every selected lane reversed sign in untouched OOS. Therefore **no EDGE is promoted from v1.0**.

An earlier implementation accidentally allowed OOS-only lanes to label themselves confirmed; this was corrected fail-closed before this checkpoint. The corrected result is zero confirmed lanes.

Interesting but non-promotable observation: XAUUSD Long became positive in OOS despite not passing discovery. It is recorded only as regime-shift evidence and is not selected as EDGE.

## Next
Broaden the same video-first methodology without chart-pattern hindsight:
- impulse continuation;
- impulse reversal;
- multi-bar streak continuation;
- rolling mean-reversion;
each with four train-only numerical thresholds, followed by discovery -> validation -> untouched final OOS.

No live authority. No production weighting.
