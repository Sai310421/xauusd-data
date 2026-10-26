# Open source choice

| Piece | Choice | Why |
|-------|--------|-----|
| Backtest engine | NautilusTrader, already in this repo's research scripts | Raw QuoteTick path exists. Not re-run here. |
| Catalog | `research/build_raw_bidask_catalog_duka.py` and `research/raw_bidask_catalog_guard.py` | Dukascopy bid/ask. Reuse, do not rewrite the feed. |
| Optimizer | none yet | Optuna is not a dependency. `adapters/optuna/objective.py` is a grid so the call shape exists. |
| vectorbt | not used | Bar research would violate the raw-tick rule. |
| Statistics | stdlib paired block bootstrap | `arch` is unnecessary until a real return series exists. |

No alternative stack was adopted, because none of them can be validated without the catalog this package refuses to fake.
