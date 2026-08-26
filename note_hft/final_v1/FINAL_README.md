# NOTE-HFT Frozen Final v1

## Purpose
Preserve the public NOTE-HFT execution architecture as a Frozen Baseline and prevent accidental strategy substitution.

## Frozen invariants
- static_qty = 0.01
- 3 second create permit
- close-first behavior
- spread == 0 entry eligibility
- original BUY/SELL side mapping
- no EMA/ATR/Spike proxy inserted for the missing signal fragment

## Authentic signal boundary
The supplied public source intentionally masks the signal generator that produces `a_cond_num` and `b_cond_num`. This pack therefore exposes an authentic signal replay boundary rather than inventing replacement alpha.

Accepted replay schema:

```csv
ts_ns,a_cond_num,b_cond_num
```

A valid BUY-side state must satisfy the original condition family `b_cond_num > c_n_of && a_cond_num < 0`; SELL-side is the symmetric original condition.

## Parity fingerprint
- N: 176,483
- BUY: 88,223
- SELL: 88,260
- WR: 72.71%
- PF: 1.74
- MaxDD: 3.97%

Structural parity is checked before economic parity. N retention below 99% is FAIL.

## Reality model
Do not use ideal-only BT for acceptance. The NOTE-HFT reality layer uses broker Bid/Ask eligibility plus latency, latency-induced slippage, reject, stale/missed UDP, and position/order acknowledgement delay. Spread is not blindly added after the fact because the Frozen entry gate itself requires zero spread.

## Status semantics
- `READY_FOR_AUTHENTIC_SIGNAL`: execution/reality/parity harness complete.
- `BLOCKED_SIGNAL`: no original signal replay is present; do not report WR/PF/DD as NOTE-HFT.
- `BLOCKED_DATA_ACCESS`: broker Bid/Ask dataset cannot be read; do not fabricate zero-spread statistics.
- `PARITY_FAIL`: signal is present but structural fingerprint fails.
- `REALITY_BT_PASS`: structural parity and reality BT both pass.
