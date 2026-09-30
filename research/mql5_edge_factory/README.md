# MQL5 EDGE Mining Factory — PR #9

PR #9 is repurposed from the stopped AMOS Math-ICT Raw Tick validation into a staging lane for MQL5 EDGE mining.

## Scope

Two permitted input lanes:

1. **White-box public/free** — MQL5 CodeBase/articles or other legitimately available source.
2. **Black-box demo observations** — Strategy Tester reports/trade histories from commercially distributed demos, analyzed only through observable behavior. No EX5 protection bypass, decompilation, or recovery of private source.

## Pipeline

`candidate -> observation/import -> feature extraction -> behavioral hypothesis -> EDGE isolation -> numeric validation -> OOS/cost validation -> S/A/B/REJECT -> promotion candidate`

Nothing is promoted to the permanent EDGE Library from this PR automatically.

## Numeric gate

Each candidate EDGE should report when data permits:

- N / trades per day
- WR
- expectancy (R and money)
- PF
- average R
- MaxDD
- MFE / MAE
- cost-adjusted EV and cost elasticity
- regime-conditional EV/PF
- OOS stability
- delta versus matched control

Core test:

`DeltaEV = E[R | EDGE] - E[R | matched control]`

Reconstruction is scored separately from profitability. Suggested behavioral match fields are direction, entry timing, add events, exit timing, holding time, and PnL/MFE/MAE distributions. A high match score is not itself an EDGE claim.

## Promotion rule

- **S**: strong positive delta after costs, adequate N, stable OOS/regimes, acceptable tail/DD.
- **A**: positive and useful, but one robustness dimension needs more evidence.
- **B**: hypothesis worth retaining for more testing; not library-ready.
- **REJECT**: no independent positive EDGE or fails cost/tail/OOS gate.

Only S/A candidates that pass the required Raw Tick/OOS validation become eligible for the existing EDGE Library. Promotion is a separate explicit step.

## Initial components

- `schema.py` — normalized observation/trade schema.
- `metrics.py` — standalone EDGE metrics and matched-control delta.
- `reconstruction.py` — behavioral event-match scoring.
- `classify.py` — S/A/B/REJECT decision scaffold.
- `run.py` — CLI entry point for CSV/JSON observations.

The first milestone is an end-to-end run on one legitimate public/demo observation set before scaling candidate count.