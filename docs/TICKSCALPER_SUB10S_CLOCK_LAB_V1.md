# TickScalper Sub-10s Clock Lab v1
Research-only, fail-closed Raw BidAsk clock ablation.

Clocks: 50/100/250/500ms, 1/2/3/5/10s, TRUE_TICK.
Frozen prototype signal: 10-tick momentum + 3 consecutive tick direction + spread gate.
Output: N, WR, PF, EV, Net, Return%, MaxDD%.

## Data gate
Input MUST contain raw timestamp,bid,ask. Mid OHLC/M1/M5 is rejected for this experiment.
Canonical production-grade source remains xauusd-duka-feed / Raw BidAsk catalog.

## Interpretation
This v1 isolates clock sensitivity. It is NOT yet the final Nautilus execution/fill model and does not claim profitability.
Next gate: wire catalog QuoteTick + realistic commissions/slippage and add adaptive N-tick/time windows.
