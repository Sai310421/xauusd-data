# Nautilus BT Entry Point

## Canonical repository

- Repository: `Sai310421/xauusd-data`
- Default branch after merge: `main`
- Current implementation branch: `feat/reusable-nautilus-bt`
- Pull Request: `#2 Add reusable manual Nautilus BT workflow`

## Canonical workflow

- Workflow name: `Nautilus BT Manual`
- File: `.github/workflows/nautilus-bt-manual.yml`
- Trigger: `workflow_dispatch`
- Python: `3.12`
- NautilusTrader default pin: `1.230.0`
- Artifact retention: `30 days`

This workflow is the permanent manual entry point for Nautilus backtests in this repository. New backtest targets should be added as selectable `target` options here instead of creating a new workflow for every experiment.

## Current target

- `minimumspike-reality-noise`
- Runner: `research/minimumspike_reality_noise_bt_v1.py`

## Planned target family

- `g75-pure`
- `g75-volume`
- `g75-volume-economic-be`
- `minimumspike`
- `minimumspike-volume`
- `g75-volume-minimumspike`

## Standard KPI contract

Every promoted target should emit, where applicable:

- `summary.json`
- `kpi.csv`
- `trades.csv`
- `baskets.csv`
- `cost_recovery.csv`
- `environment manifest`

Primary comparison metrics:

- WR
- PF
- Net Profit
- Max DD
- N
- N retention
- Cost Recovery Rate
- Net Edge / Trade
- Spread sensitivity
- Slippage sensitivity

## Operating rule

1. Use the same dataset ID and broker/execution assumptions for comparison runs.
2. Record Git SHA, Python version, and NautilusTrader version for every run.
3. Do not accept a high WR result if net edge after spread/commission/slippage is negative.
4. Keep the G75 core frozen when evaluating external Volume/OrderFlow/MinimumSpike edges unless the run is explicitly designated as a core-change experiment.
5. Preserve failed-run artifacts where possible for diagnosis.

## Where to run it

GitHub repository → **Actions** → **Nautilus BT Manual** → **Run workflow** → choose `target` → run.

This file is the location pointer for future ChatGPT, Grok, Claude, Gemini, Codex, and human operators.
