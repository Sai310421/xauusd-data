# NOTE-HFT 4-slot reconstruction — evidence ledger

Target fragment:

```python
if SLOT1 and SLOT2:
    self.a_cond_num = SLOT3
    self.b_cond_num = SLOT4
else:
    self.a_cond_num = 0
    self.b_cond_num = 0
```

## Hard source constraints

1. `udp_ask`, `udp_bid`, `udp_sp`, `udp_mid` are created immediately before the masked block.
2. The execution process does not pass `mtstore` into `LocalDataList`; therefore the four masked expressions cannot directly reference the broker/MT price unless hidden global coupling existed. The public structure instead treats UDP as the decision feed and MT as execution feed.
3. Downstream entry conditions are fixed:
   - BUY: `b_cond_num > 0 and a_cond_num < 0`
   - SELL: `a_cond_num > 0 and b_cond_num < 0`
4. The author states that a frozen/degenerate decision-side value must not trigger because opposite signs are required.
5. `ask_list` and `bid_list` exist and `main()` blocks until both contain at least five samples. The published fragment contains no visible append operation, so at least a history-maintenance line is missing/redacted in addition to the four displayed expressions.
6. The author states that this is the execution code used after finding a venue/feed condition where the win was already established. Therefore the hidden formulas should be preferred in the class of short, low-complexity transformations of a leading UDP price feed, not complex indicators.
7. The bot's edge architecture is therefore most consistent with:
   `leading/reference UDP quote movement -> directional condition -> zero-spread execution venue -> immediate close`.

## Strong mathematical invariant from the downstream signs

A coherent upward quote shift should create `(a_cond < 0, b_cond > 0)`, while a coherent downward shift should create `(a_cond > 0, b_cond < 0)`.

The minimal symmetric transform satisfying this is:

```python
a_cond = ask_old - ask_new
b_cond = bid_new - bid_old
```

This has an important microstructure property:

- both sides rise -> BUY
- both sides fall -> SELL
- spread widens (`ask up`, `bid down`) -> both negative -> NO TRADE
- spread narrows (`ask down`, `bid up`) -> both positive -> NO TRADE
- frozen quote -> both zero -> NO TRADE

That behavior matches the author's explanation of the opposite-sign requirement and is materially stronger evidence than fitting N alone.

## Reconstruction order

1. SLOT1 — validity/history condition on ask side.
2. SLOT2 — symmetric validity/history condition on bid side.
3. SLOT3 — ask-side signed movement.
4. SLOT4 — bid-side signed movement.

Economic fingerprints (`WR=72.71%`, `PF=1.74`, `DD=3.97%`) are validation targets, not direct fitting targets. `N=176,483` is used only after directional/economic evidence.
