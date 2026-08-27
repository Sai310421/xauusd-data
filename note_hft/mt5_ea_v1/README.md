# NOTE-HFT Reconstructed MT5 EA v1

Operational MT5 conversion of the reconstructed NOTE-HFT bot while preserving the observed Frozen execution structure.

## Frozen defaults
- Fixed lot: 0.01
- Alpha window: 5 samples
- BUY: `b_cond_num > 0 && a_cond_num < 0`
- SELL: `a_cond_num > 0 && b_cond_num < 0`
- Reconstructed alpha: `a_cond_num = old_ask - new_ask`, `b_cond_num = new_bid - old_bid`
- Entry spread gate: 0 points by default
- Create cooldown: 3000 ms
- Close-first: position observation triggers exit without opposite signal
- Minimum hold: 100 ms, matching the public Python `await asyncio.sleep(0.1)` after entry send
- Free-margin ratio gate: 0.40
- Async MT5 order mode

## Baseline fingerprint
- N: 176,483
- BUY: 88,223
- SELL: 88,260
- WR: 72.71%
- PF: 1.74
- MaxDD: 3.97%

## Audit
The EA writes `NOTE_HFT_AUDIT_<SYMBOL>.csv` to MT5 Common Files with entry/exit events, bid/ask, spread, reconstructed alpha values, profit, and holding milliseconds.

## Validation rule
Do not add filters to raise WR by reducing N. First preserve trade-count/BUY-SELL structure, then validate WR/PF/DD using Every tick based on real ticks.

## Status
This is RECONSTRUCTED, not a claim that the author's hidden original `a_cond_num / b_cond_num` formula has been recovered verbatim.
