# NOTE-HFT Frozen Restore v2

## Purpose
Preserve the published NOTE-HFT execution core and repair only deleted structural plumbing. Never invent the redacted decision formula.

## Verified from the author's follow-up
- `sp_limit_cnt=0` is intentional: new entries are allowed only when broker spread is exactly zero.
- Decision inputs arrive through UDP from a separate process.
- `if 00000 and 00000` is explicitly a redacted decision-logic block.
- Execution core consumes only `a_cond_num` / `b_cond_num`.
- ~3.4 second decision/entry cadence is intentional.
- Position close path is based on broker position snapshot, not market-price decision logic.

## Frozen fingerprint candidate
- N: 176,483 (reference screenshot supplied in chat)
- WR: 72.71%
- PF: 1.74
- MaxDD: 3.97%
- BUY: 88,223
- SELL: 88,260

A separate public author record also states 352,966 executions in about two weeks for the HFT bot family. Do not conflate the two report windows.

## Architecture
`UDP decision feed -> SignalProvider -> a_cond_num/b_cond_num -> FROZEN execution core -> MT4/MT5 adapter`

The default SignalProvider is FAIL-CLOSED. It never generates synthetic BUY/SELL decisions.

## Repaired structural losses
1. `ask_list` / `bid_list` are actually updated and bounded.
2. `a_cond_num` / `b_cond_num` are explicitly initialized.
3. Signal provider interface is added outside the Frozen core.
4. A replay provider can consume recovered historical signal traces.
5. Parity gate checks N, BUY/SELL balance, WR, PF and DD.
6. Reality-noise evaluation remains mandatory after structural parity.

## Not reconstructed
The redacted formula that transforms UDP prices into `a_cond_num` / `b_cond_num`. It must be supplied from an authentic source or recovered signal trace. Any guessed EMA/ATR/spike/momentum replacement is forbidden in the Frozen baseline.
