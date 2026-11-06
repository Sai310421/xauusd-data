# TickScalper behavioral-research lane

This lane is a clean-room behavioral analysis of a user-supplied Myfxbook CSV.
It does not recover, copy, or require the source EA code.

## Components

- `tickscalper_csv_profile.py`: converts the CSV into a reproducible observed-behavior JSON.
- `tickscalper_g1_policy.py`: a pure, testable risk controller that replaces
  unconditional 1.5x / 10-level recovery with a capped, signal-confirmed policy.

## AE Math integration

G1 adopts only risk-shrinking controls from the AE Math research lane:

1. **Dynamic Drawdown Reserve**: add capacity falls continuously as portfolio
   drawdown approaches its cap.
2. **DD velocity/acceleration guard**: a worsening drawdown can only reduce
   add capacity.
3. **First-Passage gate**: an add requires an externally estimated, validated
   recovery probability above a fixed threshold.

The supervisor output is always constrained to `0 <= multiplier <= 1`.
It can halt, reduce, or reject an add; it cannot increase a lot, revive the
10-level grid, or override a hard stop. HJB/QVI, MPC, DRO, and conformal
methods remain research candidates until they pass independent Raw Bid/Ask
native-fill OOS validation.

## Run the profile

```bash
python research/tickscalper/tickscalper_csv_profile.py \
  --csv /secure/path/statement.csv \
  --output results/tickscalper/profile.json
```

Do not commit the source CSV: it is a user-provided trading record.

## Nautilus integration gate

Before a native Raw Bid/Ask replay:

1. Run the CSV profile and retain its output hash.
2. Materialize the exact raw QuoteTick catalog for the CSV date range.
3. Resolve the CSV timezone offset on the in-sample partition only.
4. Keep fills on raw bid/ask; never use OHLC execution.
5. Run three ablations on a fixed OOS split: observed 10-level envelope,
   no-add baseline, and G1 capped-add policy with the AE supervisor.

No output is a trading recommendation or a production-ready strategy without
native-fill evidence, calibrated costs, OOS, and stress tests.
