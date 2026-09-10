# DexG Price-State Observation Lab

Official multi-model window for DexG only.

Repo: Sai310421/xauusd-data
Path: research/dexg_price_state/

## Hard separation

G75 is fully detached from this lab.

Out of scope here:
- G75 / G75 TSUGI / G75 Negative Memory
- Hydra66 feature contract as a G75 input
- G75 AE Math Supervisor and EV-G75 surfaces
- any G75 trigger / add / reversal geometry

Those stay in existing G75 research paths and PRs. Do not import, reuse, or A/B-mix them into DexG Baseline.

DexG Frozen Baseline is DAIGO-second Entry/TP Module v1.0 only:
Timing → Direction → Tick Pressure → Destination → Entry → Hold/Add → TP/Exit → Dynamic ATR.

New X features are never mixed into Baseline. Promote only after independent A/B on Raw Tick.

## Pipeline (fixed)

1. X observation notes
2. Entry / Add / Hold / Exit labels
3. Hypothesis
4. Independent Grok audit (PR)
5. Raw Tick / Nautilus BT (no D1 OHLC path, explicit spread)
6. Ablation
7. Adopted EDGE only if promotion gates pass

## Truth boundaries

- Raw Bid/Ask QuoteTick execution only. No OHLC resample fallback.
- KPI table required: N, WR, PF, RF, Net, DD, expectancy, break-even rate.
- No fabricated PASS. Missing raw quotes => BLOCKED_NO_RAW_QUOTES.
- Grok audit PRs stay independent; GPT/Codex integrate after review.
- G75 code, artifacts, workflows, and feature packs are not evidence for DexG.

## Layout

- baseline/      frozen DAIGO Entry/TP v1.0 only
- observations/  X notes and price-state snapshots
- labels/        Entry/Add/Hold/Exit annotations
- hypotheses/    DexG-only candidate hypotheses
- audits/        Grok audit notes referencing DexG PRs
- bt/            DexG runner specs and artifact pointers
- ablation/      DexG factor isolation records
- adopted/       promoted DexG EDGE only
