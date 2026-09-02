# QuantConnect Cloud Bridge

Saved: 2026-09-03

Purpose: make QuantConnect cloud usable from the same GitHub Actions / multi-chat contract as Nautilus, without mixing verification labels.

## What this is
A second engine for XAUUSD Hour CFD QuoteBars on QuantConnect Cloud.

## What this is not
- Not NautilusTrader
- Not Dukascopy BI5 QuoteTick
- Not `NAUTILUS_BT`, `RAW_BIDASK_PASS`, `BROKER_REALITY_BT`, or `VERIFIED`

Label every QC result exactly `QC_CLOUD_BT`.

## Fastest path (no CLI)

1. Open https://www.quantconnect.com/terminal
2. New Python project
3. Paste `research/qc/xauusd_hour_cloud.py` into `main.py`
4. Backtest
5. Copy the log line starting with `QC_KPI_JSON=`
6. Store it under `results/qc-bt/<experiment_id>/kpi.json`

Parameters in the QC UI:
- `start_date` default `2025-09-01`
- `end_date` default `2026-08-28` (matches `xauusd-duka-feed` books window)
- `experiment_id` use `YYYYMMDD-HHMM-qc-agent-shortid`
- `cash` `1000`

## GitHub Actions path

Repo: `Sai310421/xauusd-data`

Actions:
- `QC Cloud XAUUSD Hour Smoke`
- `QC Cloud BT Hub`

Secrets (required for CLI cloud run):
- `QC_USER_ID`
- `QC_API_TOKEN`

Get them from https://www.quantconnect.com/account
Lean CLI cloud commands also need a paid Organization seat.

If secrets are absent the hub still writes an immutable request envelope and fails closed. It does not invent KPIs.

## Navigation

`Sai310421/xauusd-data -> Actions -> QC Cloud XAUUSD Hour Smoke`
`Sai310421/xauusd-data -> Actions -> QC Cloud BT Hub`
