# G75 Multi-Horizon Natural Recovery / Positive Exit — Verified Evidence

Status: DISCOVERY EVIDENCE / WR5 INVALID
Repository: Sai310421/xauusd-data
Branch: nautilus-parallel-lanes-v1
Workflow: G75 Multi-Horizon Fixed Debt Probe v1
Run ID: 33206662093
Head SHA: 0ac85b5179ca90272a55ea370d296130d67c1953
Artifact ID: 9699937571
Artifact: nautilus-MODULE_RECOVERY-g75-fixed-debt-24h-30d-v3-xauonly
Artifact SHA256: f272eb3e5815b5ea54bb7d011cb1f5cfd903e0471edbee7ea651b8784c11a2f4

## Evidence contract

- XAUUSD
- 30-day Raw Bid/Ask Nautilus catalog
- 2,478,315 QuoteTicks
- NautilusTrader 1.230.0
- OHLC resample: false
- Tick thinning: false
- Raw Bid/Ask quality gate: PASS
- Catalog cache reused successfully
- L10 and L20 evaluated independently

This run measures natural executable-basket first passage after the G75 loss/lock state. It does not yet execute the hedge order or independent rescue strategy. Spread is present in Bid/Ask paths. Commission, explicit slippage, execution delay, swap/hedge carry, broker margin mechanics, cashback and actual rescue execution are not modeled. Therefore WR5 remains INVALID and these results are not promotion evidence.

## Economic BE recovery rate

| Horizon | L10 | L20 |
|---:|---:|---:|
| 300s | 72.38% | ~same regime |
| 600s | 80.06% | ~same regime |
| 1800s | 87.70% | ~same regime |
| 3600s | 90.45% | 90.28% |
| 4000s | 90.98% | 90.81% |
| 4500s | 91.35% | ~same regime |
| 5400s | 91.92% | ~same regime |
| 6h | 95.51% | 95.45% |
| 12h | 96.84% | ~same regime |
| 24h | 97.25% | 97.20% |

Note: values marked `~same regime` were not separately quoted in the preserved human summary and must be read from the immutable JSON artifact before use as exact numeric evidence. Do not infer an exact value from this table.

## Post-3600s tail behavior

For L10, among cases still unresolved at 3600s:

- recovered by 4000s: 5.48%
- recovered by 4500s: 9.37%
- recovered by 5400s: 15.32%

Interpretation: 4000s is not a 100% recovery boundary. A meaningful fraction of the 1-hour tail continues to recover naturally, and the strongest observed improvement in the measured horizons occurs by 6h and then 24h.

## Natural positive-exit opportunity

L10 at 6h:

- Economic BE: 95.51%
- +10% of locked debt: 94.94%
- +25% of locked debt: 94.49%
- +50% of locked debt: 93.51%

L10 at 24h:

- Economic BE: 97.25%
- +10% of locked debt: 96.94%
- +25% of locked debt: 96.69%
- +50% of locked debt: 96.15%

L20 at 6h:

- Economic BE: 95.45%
- +50% of locked debt: 93.40%

These are natural price-path opportunity rates, not realized rescue-system PnL.

## Strategy implications to test, not assumptions

The evidence supports testing a multi-horizon Debt-to-Profit router rather than a fixed 300s failure cutoff:

`Loss State -> Natural Hold -> Tail Classification -> Hedge/Fixed-Debt Ledger -> Independent Rescue -> BE Protect -> Profit Extend`

Candidate timing policy for validation:

1. 0–3600s: Natural Hold when DD/margin/tail-risk constraints permit.
2. 3600–5400s: classify the residual tail rather than forcing immediate rescue; the measured residual continues to recover.
3. 5400s–6h: selectively activate independent rescue only when expected net rescue velocity exceeds debt/cost growth and margin/DD guards pass.
4. At BE: transition to BE_PROTECT.
5. Beyond BE: allow PROFIT_EXTEND only when predicted MFE/expected net edge supports continuation.
6. At 6h: unresolved population is approximately 4.5%; treat this population as the primary Tail/Rescue research target.
7. At 24h: unresolved population is approximately 2.7–2.8%; this is the hard-tail research target, not proof of unavoidable loss.

## Required next validation

Run actual Nautilus execution variants under identical Raw Bid/Ask data and total rescue-risk budget:

- Natural/no-control baseline
- HedgeLock only
- HedgeLock + independent Rescue 3
- HedgeLock + independent Rescue 5
- HedgeLock + independent Rescue 10
- Multi-Horizon Debt-to-Profit policy

Primary outputs: SystemPnL>0 first passage at 300/600/1800/3600/4000/4500/5400/6h/12h/24h, MaxFloatingDD, minimum MarginLevel, Rescue PF/expectancy, DebtPaydownTime, CostPerRecoveredDollar, unresolved inventory age, aggregate exposure, tail loss, profit/debt velocity ratio.

For promotion-quality evidence, add all WR5 costs/reality fields and robustness/OOS gates. Missing reality data remains INVALID.
