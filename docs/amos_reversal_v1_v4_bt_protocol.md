# AMOS Reversal V1–V4 BT Protocol

## Matrix
- Versions: V1 / V2 / V3 / V4
- Timeframes: M1 / M5 / M15 / H1
- Modes: V1=BASE; V2–V4=AGGRESSIVE/STANDARD/CONSERVATIVE
- Total cells: 40

## Baseline comparability
- Symbol: XAUUSD
- Execution source: raw bid/ask ticks only
- Signal state: completed bars derived from raw ticks
- Stop: structural invalidation
- Target: fixed 2R baseline
- Same fill/cost logic across generations

## Primary KPIs
N, WR, AvgR, PF, EV/trade, NetR, MaxDD(R), RF.

## Acceptance rule
Do not select V4 on WR alone. Prefer the generation/mode/TF cell that improves PF and EV while preserving useful N and containing DD. V3/V4 diagnostics should be used to explain the improvement rather than merely rank it.

## Next validation layers
1. 40-cell in-sample grid
2. V3→V4 context ablation
3. out-of-sample confirmation
4. walk-forward
5. multi-symbol portability test
