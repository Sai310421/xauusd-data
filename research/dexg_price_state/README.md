# DexG Price-State Observation Lab

Official multi-model window for this repository.

Repo: Sai310421/xauusd-data
Path: research/dexg_price_state/

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

## Layout

- observations/  X notes and price-state snapshots
- labels/        Entry/Add/Hold/Exit annotations
- hypotheses/    candidate hypotheses before BT
- audits/        Grok audit notes referencing PRs
- bt/            runner specs and artifact pointers
- ablation/      factor isolation records
- adopted/       promoted EDGE only
