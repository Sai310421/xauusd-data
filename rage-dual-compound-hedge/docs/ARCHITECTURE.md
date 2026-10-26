# Architecture

Two cash books plus a pool.

- Main engine seam: `NullMainEngine` places no orders. Candidate script paths are listed and not imported.
- Hedge engine: reads v1 intensity, converts it to ounces, and faces the main position. A flat main book forces the protective sleeve off. That policy is outside the formula.
- Insurance pool: payoff is not trading pnl. A claim larger than the pool is a recorded shortfall, not new cash.
- Governor: zero equity or drawdown at the limit sets quantity to zero.
- Stage-1 optimizer: smallest hedge quantity meeting a named scenario set. Infeasible stays infeasible.
- Stage-2 optimizer: log-growth on capital the lock did not take, using only sleeves the caller passed in.

Nautilus native fills are a separate process. `adapters/nautilus/fills.py` returns zero and raises if something asks it to trade.
