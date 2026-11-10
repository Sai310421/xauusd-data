# USDJPY Institutional Risk AI — Phase 2

This directory implements the **data and feature pipeline only**. It is monitoring research infrastructure, not an execution system.

## Fixed contracts

- Futures: Databento continuous-front H1 contract identifiers, configured in `config.yaml`.
- USDJPY: OANDA H1 Mid only, never Bid/Ask or broker execution prices.
- All joins use completed bar-end UTC timestamps.
- Fixture mode is the only mode supported in this phase. Missing production credentials must fail closed; no synthetic fallback is allowed.

## Fixture run

```bash
cd usdjpy-risk-ai
python tests/fixtures/make_fixture.py --output /tmp/raw_fixture.parquet
python -m src.collect --input /tmp/raw_fixture.parquet --output data/raw
python -m src.features --input data/raw --output data/features
pytest -q
ruff check .
```

## Phase 2 evidence

The pipeline validates raw H1 inputs, rejects duplicates and invalid prices, writes monthly Parquet partitions, calculates only backward-looking features, calculates 24-hour USDJPY labels, and records feature source timestamps for leakage tests.

Model fitting, calibration, OCI storage, Discord, live provider calls, and order routing are intentionally outside this phase.
