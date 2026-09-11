# Pipeline

observation → label → hypothesis → Grok review → Raw Tick BT → ablation → adopt

Grok is a station, not a signal source and not an ARMED gate.

## Grok review contract

Input: observation md or hypothesis md + current frozen CORE + candidate module flags.
Output must classify each item as one of:
- KEEP_CORE
- CANDIDATE (name the layer: DEX_G / NERCIA / v2.1-GAP)
- REJECT (reason)
- NEEDS_AB

Forbidden:
- writing a new AND-gate onto ARMED
- mixing G75 / Hydra66
- promoting from public X P/L screenshots
- D1 OHLC path

After review, append a snapshot comment to Issue #19. Do not edit the frozen v2.1 issue body.
