# NOTE-HFT Frozen Final v2

## Completion contract

This package preserves the supplied NOTE-HFT execution behavior and refuses to invent the redacted signal generator.

### Frozen invariants
- static_qty = 0.01
- zero-spread entry eligibility remains a hard gate
- 3-second CT permit remains unchanged
- close-first behavior remains unchanged
- BUY/SELL mapping remains unchanged
- a_cond_num / b_cond_num must come from authentic replay/live input

### Baseline fingerprint
- N = 176,483
- BUY = 88,223
- SELL = 88,260
- WR = 72.71%
- PF = 1.74
- MaxDD = 3.97%

### Required replay file
CSV fields at minimum:
`timestamp,a_cond_num,b_cond_num,side,pnl`
Optional: `dd_pct,bid,ask,entry_price,exit_price,latency_ms`

### Reality input
CSV fields at minimum:
`timestamp,bid,ask`

The reality gate measures actual zero-spread eligibility and applies the original 3-second re-entry lock to parseable tick timestamps. It does not infer latency survival from OHLC bars.

### Acceptance order
1. Structural parity: N, BUY, SELL.
2. Execution/reality eligibility.
3. Economic parity: WR, PF, DD.
4. Supervisor/ONNX/Vision only in shadow mode first.
5. Intervention is allowed only after proving improvement without destroying N.

### Fail-closed states
- BLOCKED_SIGNAL: authentic signal replay absent.
- BLOCKED_DATA_ACCESS: Bid/Ask input absent.
- BLOCKED_DATA_FORMAT: input is not real Bid/Ask tick-like data.
- FAIL: structural parity outside tolerance.
- PASS: structural parity within 1% on N/BUY/SELL; economics are reported separately.

No proxy signal generator is included.