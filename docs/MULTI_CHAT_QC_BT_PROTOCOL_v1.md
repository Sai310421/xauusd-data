# Multi-Chat QuantConnect BT Protocol v1

Purpose: any chat can request a QC cloud BT the same way it requests a Nautilus BT, without confusing the two.

## Canonical files
- algorithm: `research/qc/xauusd_hour_cloud.py`
- runner: `research/qc/qc_cloud_runner.py`
- config: `research/qc_bt_standard_config.json`
- bridge: `docs/QUANTCONNECT_CLOUD_BRIDGE.md`
- shared data: `docs/SHARED_BIDASK_CATALOG_BRIDGE.md`
- smoke workflow: `.github/workflows/qc-cloud-xau-smoke.yml`
- hub workflow: `.github/workflows/qc-cloud-bt-hub.yml`
- results: `results/qc-bt/<experiment_id>/`

## Invocation
When the user says `QC BTして` or `QuantConnect クラウド`:

1. Create `experiment_id` = `YYYYMMDD-HHMM-qc-<agent>-<shortid>`
2. Dispatch `QC Cloud BT Hub` or paste the algorithm into the QC web IDE
3. Keep `verification_level=QC_CLOUD_BT`
4. Return run_id + SHA + `manifest.json` + `QC_KPI_JSON`

Do not run Nautilus workflows for this request.

When the user says `Lean local` or `同じ BidAsk で QC`:

1. Materialize `lean_data/` via `Shared BidAsk Materialize`
2. Point Lean `data-folder` at `lean_data`
3. Label `LEAN_LOCAL_BT`
4. Do not call it `QC_CLOUD_BT` or `NAUTILUS_BT`

## Fail closed
Missing `QC_USER_ID` / `QC_API_TOKEN` is not a proxy license.
Write `FAIL_CLOSED_NO_CREDENTIALS` and point to the web-IDE paste path.
Missing `lean_data` is not a license to use OANDA Cloud ticks as Dukascopy.

## Evidence rule
Never call a QC Cloud result `NAUTILUS_VERIFIED` or `RAW_BIDASK_PASS`.
Never call a Lean-local Dukascopy replay `QC_CLOUD_BT`.
