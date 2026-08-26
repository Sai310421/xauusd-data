# NOTE-HFT Frozen Reconstructed Final v3

## Goal
Reconstruct the missing public NOTE-HFT decision fragment without modifying the frozen execution core.

## Frozen invariants
- static_qty = 0.01 (public sample)
- c_n_of = 0
- spread gate: new entry only when broker spread == 0
- 3-second create permit after entry
- close-first: any detected position is closed immediately by the existing public logic
- BUY condition: b_cond_num > 0 AND a_cond_num < 0
- SELL condition: a_cond_num > 0 AND b_cond_num < 0
- UDP is lossy/latest-only by design

## Reconstructed missing alpha
Public code waits for 5 ask/bid samples but removes the lines that populate/use them. The most parsimonious reconstruction consistent with the sign conditions and price-only UDP payload is:

```
a_cond_num = old_ask - new_ask
b_cond_num = new_bid - old_bid
```

over a rolling 5-sample window.

An upward quote move gives a<0,b>0 => BUY. A downward quote move gives a>0,b<0 => SELL. This restores the missing decision fragment while leaving execution unchanged.

`window=5` is the reconstruction default because the public source explicitly blocks startup until five ask and five bid samples exist. `parity_calibrate.py` may compare windows 2..9, but it is prohibited from changing execution logic or target event counts directly.

## Baseline fingerprint
- N: 176,483 trades
- BUY: 88,223
- SELL: 88,260
- WR: 72.71%
- PF: 1.74
- MaxDD: 3.97%

Structural parity is evaluated first. Economic parity (WR/PF/DD) is evaluated only after structural parity.

## Reality policy
Formal NOTE-HFT evaluation must preserve the zero-spread eligibility gate. Noise is execution-side: latency/latency-slippage, stale or missed UDP updates, broker position/ACK delay, and rejection/retry behavior. Artificial normal spread must not be added to eligible entries because that changes the original premise.

## Status language
- `RECONSTRUCTED`: missing public fragment rebuilt from observable structure.
- `AUTHENTIC`: author's hidden original formula recovered verbatim.
- Never label RECONSTRUCTED as AUTHENTIC.

The reconstructed v3 is the operational completion target. If the hidden formula is later recovered, it can replace only `reconstructed_alpha.py`; the Frozen Core and all parity/reality gates remain unchanged.
