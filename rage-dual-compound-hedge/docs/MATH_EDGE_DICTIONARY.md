# Formula dictionary

## v1-original

```
D_t = max((K - S_t) / S_t, 0)          # long only; S<=0 returns 0
w_t = 1 / (1 + exp(-κ1 (σ_t/σ_ref - 1) - κ2 D_t))
R_t = λ1 D_t + λ2 max((σ_t-σ_ref)/σ_ref, 0) + λ3 σ_t √τ
H_t = clip( w_t R_t exp(-δ Δt) / (1 + ρ Risk_t) , 0, H_max )
```

σ is an annualized fraction. Δt and τ are in years. H is dimensionless.

Properties that the unit tests lock in:

- H rises with D and falls with Risk_t.
- If λ3 > 0, H stays positive at D = 0 and σ = σ_ref. The baseline spends hedge intensity when nothing is impaired.
- Risk in the denominator cuts size during stress. That can reduce protection in the state that needs it.

## v2-budget-split (candidate, not a replacement)

Adverse move is side-aware and rejects non-positive prices.
`budget_risk` stays in the denominator. `market_stress` enters the weight with the opposite sign, so a market shock increases w instead of shrinking H.
v2 is not evidence of a better hedge until the same tick log is scored for both.

## Quantity

`q = H * protected_notional / (contract_notional * (1 + spread))`

This line is not in the original formula. Leaving it out confuses an intensity with a position.

## Capital lock

If a fraction `f` of a book is locked and that lock loses `c` over the month, the free sleeve must earn

`(1 + target - f (1 - c)) / (1 - f) - 1`

to leave the whole book at `target`. At f = 0.30, c = 0.01, target = 0.10, the free sleeve needs about 14.7% a month, not 10%.
