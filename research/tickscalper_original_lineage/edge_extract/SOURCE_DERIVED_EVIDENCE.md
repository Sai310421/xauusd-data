# Source-Derived Reconstruction Evidence

## TickSmoother / direction engine
- Igor states `TS_TimeFrame=-1` calculates tick MAs via TickSmoother.
- TickSmoother v2.1 is therefore a hard dependency for original tick-mode parity.
- A later forum inspection exposes internal arrays `Hi`, `Lo`, `tOpen`, `tClose`, `Fast`, `Slow`, `price` and a `ShiftArray(int mode)` routine.
- The old shift loop was reported as `for(int cnt=barCounter; cnt>=0; cnt--)`; later advice changed the lower bound to `cnt > 0`.
- TickSmoother v2.2 later added timer-driven behavior, explicitly warned to differ at short intervals. Therefore **v2.1, not v2.2, is the parity target**.

## v3.44 entry confirmation clues
Forum discussion around v3.44 documents modes where:
- MACD is compared against its signal,
- TickSmoother Fast/Slow MAs are compared with tick close and each other,
- confirm MAs are also included.

A shown short-side relation is structurally:
`tClose < fastMA < slowMA < confMA1 < confMA2`.

This is source-adjacent evidence for a multi-stage trend/confirmation entry component and is stronger than inferring direction only from statement data.

## v3.44 recovery / exit clues
Documented:
- Martingale is active for ExitMode 11/12.
- ExitMode 11: last martingale order profit is tied to `MGPips`.
- ExitMode 12: flexible basket-style exit via `ProfitTarget`.
- `ProfitMode`: 0=Pips, 1=Profit(USD), 2=Total Pips, 3=Total Profit(USD).

This directly supports splitting Core Direction/Timing from Recovery/Sizing and Basket Exit.

## Later safety evolution
Public lineage descriptions add:
- no-hedge and time-condition controls (v3.44.7),
- EquityTrailing (v3.45),
- expanded time filter and magic-number controls (v3.45.1),
- MaxOrders close safety (v3.45.6),
- Salvage/martingale recovery (v3.45.7a family).

These should be diffed as later layers, not assumed to belong to the earliest core.

## Parity consequence
Reconstruction order remains:
1. TickSmoother v2.1 state/output parity
2. Core entry/confirmation parity
3. spread/timing filters
4. exit without recovery
5. add/sizing/recovery
6. later safety layers
7. reofx 2026 delta
