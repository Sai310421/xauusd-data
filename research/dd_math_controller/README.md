# AMOS DD Math Controller

Purpose: reduce drawdown without changing the TickScalper entry/exit signal logic.

Architecture:

```
TickScalper Core (signal frozen)
        |
        v
DD Math Controller
        |
        v
Execution / sizing
```

## Frozen Core
- Raw BidAsk only
- TickSmoother candidate: 5-tick, MA 3/5/8/13, cross-only
- Session: 07-17 UTC
- Basket exit: 0.8575
- Max layers: 10
- Direction/entry/exit rules are not changed by DD-control experiments.

## Controller families
1. DD-Adaptive Exposure: scale actual order size from closed-equity DD.
2. Floating-DD Governor: keep ENTRY signal unchanged and scale only ADD exposure from mark-to-market DD.
3. Combined DD + Floating-DD: use the stricter of closed-DD and floating-DD exposure budgets.
4. First-Passage / Ruin Governor: deep-layer intervention from adverse boundary-hit probability.
5. Barrier / CVaR / EVaR / Azema-Yor: research extensions, kept outside Core.

## Primary objective
Minimize MaxFloatingDD subject to:
- Return_control >= 0.90 * Return_baseline
- PF_control >= PF_baseline - 0.05
- N_control >= 0.98 * N_baseline
- WR_control >= WR_baseline - 0.5 percentage points

## DD Compression Ratio
DCR = relative_DD_reduction / relative_Return_reduction

A candidate is not promoted on DCR alone. It must also clear a minimum material DD reduction gate.

## Promotion labels
- CORE-FROZEN: entry/exit unchanged.
- DD-ONLY-CANDIDATE: sizing/exposure only.
- DD-ONLY-PASS: all primary gates passed and material DD reduction achieved.
- REJECTED: fails preservation gates.
