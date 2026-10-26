# AMOS RAGE Dual Compound Hedge

Research package for a dual book: a main engine and a hedge engine that is supposed to compound on its own while covering the main book.

**Live trading: off.**

The monthly 10% figures in the hypothesis are targets. This tree does not contain a raw bid/ask fill log, so those targets are **BLOCKED**, not achieved.

## What is implemented

- `v1-original` ProtectiveHedge, unchanged, plus a separate `v2-budget-split` candidate
- Independent main / hedge cash books and an insurance pool
- Two-stage optimizer that will not relax the protection constraint to force a 10% result
- Raw quote gate (`RAW_BIDASK_CATALOG_OK` only for real bid/ask rows)
- Walk-forward index splitter and paired block bootstrap
- A structural overlay model explicitly labeled synthetic

## What is not implemented

- A Nautilus run against `research/build_raw_bidask_catalog_duka.py`
- Execution of `ae_allweather_*`, `armada_*`, or `bigplayer_*`
- Optuna studies (a deterministic grid stands in; Optuna is not imported)
- Any order a broker could receive

## Checks

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
PYTHONPATH=src:. python scripts/build_report.py
python scripts/run_backtest.py
```

`run_backtest.py` exits 2 until a real catalog is passed, and even then it does not invent native fills.

GitHub Actions workflow that only runs the unit tests belongs at the repository root `.github/workflows/rage-dch-unit.yml`, because a workflow nested in this directory does not run.
