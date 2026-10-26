# Failure modes already visible without ticks

1. **Idle intensity.** λ3 σ √τ keeps H > 0 when drawdown is zero. A literal implementation buys protection it does not need.
2. **Risk shrinks the hedge.** v1's denominator falls as Risk_t rises. Protection can be smallest in the shock.
3. **H is not a size.** Without the quantity map, the formula cannot be compared to a loss bound.
4. **Bad ticks look calm.** v1 drawdown returns 0 when S ≤ 0.
5. **Pure overlay cannot be H1-B.** If the hedge book is only the opposite of the main book minus cost, its expectancy is the negative of the main expectancy minus cost. A 10% hedge compound then requires a second alpha, on capital the hedge did not lock.
6. **Small losses are shared.** When the main loss is inside the spread and fee, both books can print a loss in the same month. Opposite beta does not remove that band.
7. **Locking capital raises the required edge** of the free sleeve above the headline 10%.
8. **Independent shuffle** understates simultaneous loss whenever the books share a shock. It is rejected.
9. **OHLC proxy** can pass a backtest shape and still fail the quote gate. It is rejected.
10. **Infeasible stage 1** must stay infeasible. Lowering the loss bound after seeing the result is not an optimization, it is a different problem.
