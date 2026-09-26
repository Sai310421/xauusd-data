# AMOS_Vivek_Rao_Quant_Library

Independent research line for extracting reusable quantitative components from vivek-v-rao public repositories.

## Mission
Extract, classify, validate, and re-implement reusable EDGE components for AMOS/AE without coupling this work to TickScalper reconstruction.

## Seven extraction lanes
1. Math / Statistics
2. Distribution / Tail
3. Volatility / Regime
4. Data / Quality
5. Risk / Portfolio
6. HFT / Execution
7. Backtest / Validation

## Promotion pipeline
SOURCE -> INVENTORY -> LICENSE GATE -> EDGE CARD -> UNIT TEST -> SYNTHETIC TEST -> MARKET DATA TEST -> OOS / STRESS -> AMOS CANDIDATE -> PROMOTED

No repository is promoted merely because its backtest is profitable. Every candidate must identify the actual reusable mechanism, assumptions, failure modes, data requirements, leakage risks, and license.

## Priority Wave 1
- ReturnDistributions
- Return-Mixtures
- Return-Mixtures-of-Normals
- Time-Series-Distributions
- Conditional-Skew
- intraday-prices
- moving-average-systems
- Sharpe-out-of-sample
- OHLC-Vol
- Intraday-Vol
- GARCH-variants
- High-Frequency-Trading-Software

## Output contract
- registry/repo_inventory.csv
- registry/edge_registry.csv
- dictionary/math_edge_dictionary.md
- modules/
- validation/
- reports/

TickScalper is explicitly out of scope. Cross-use into TickScalper requires a separate, labeled external-enhancement gate.
