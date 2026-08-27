# NOTE-HFT Four-Slot Recovered BOT v5

This is the recovery-first BOT package. The user's original Python NOTE-HFT source remains the source of truth. The package does not replace the bot with a simplified proxy.

## Four recovered masks

```python
if len(self.ask_list) >= 5 and len(self.bid_list) >= 5:
    self.a_cond_num = self.ask_list[0] - self.ask_list[-1]
    self.b_cond_num = self.bid_list[-1] - self.bid_list[0]
else:
    self.a_cond_num = 0
    self.b_cond_num = 0
```

The public source also exposes a five-sample startup gate but omits the lines that maintain `ask_list` and `bid_list`. The patch therefore restores only the implied five-sample rolling history immediately after UDP ask/bid reception.

## Why this is the minimum consistent reconstruction

For coherent upward quote motion, `old_ask-new_ask < 0` and `new_bid-old_bid > 0`, exactly satisfying the visible BUY predicate. Downward motion gives the exact SELL signs. Pure spread widening or narrowing gives equal signs and is rejected by the visible downstream predicates. This uses the fewest operators consistent with the source structure and preserves ask/bid symmetry.

## Recovery order

1. SLOT1 = ask-history readiness.
2. SLOT2 = bid-history readiness.
3. SLOT3 = signed ask move with orientation required by visible BUY logic.
4. SLOT4 = signed bid move with orientation required by visible SELL logic.

N is not used to choose these expressions. `N / BUY / SELL / WR / PF / DD` are downstream fingerprints only.

## What remains unchanged

- original async/ZMQ architecture
- UDP input path
- MT execution path
- `static_qty=0.01`
- `c_n_of=0`
- zero-spread entry gate
- 3-second create permit
- 0.1-second sleeps after create/close
- balance/margin gates
- position-snapshot close-first behavior
- fast-order path

## Status

`CODE_RECOVERY_COMPLETE_CANDIDATE` means the four masks and their required rolling history have been restored in a source-preserving way. It does **not** mean economic parity is already proven. WR=72.71%, PF=1.74 and DD=3.97% remain validation fingerprints, not hard-coded targets.

Use `apply_four_slot_recovery.py ORIGINAL.py COMPLETE.py` to produce the recovered bot from the exact original file while preserving a SHA-256 audit trail.
