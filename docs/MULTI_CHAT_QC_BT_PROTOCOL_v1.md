# Multi-Chat QuantConnect BT Protocol v1

Purpose: any chat can request a QC cloud BT the same way it requests a Nautilus BT, without confusing the two.

## Canonical files
- algorithm: `research/qc/xauusd_hour_cloud.py`
- runner: `research/qc/qc_cloud_runner.py`
- config: `research/qc_bt_standard_config.json`
- bridge: `docs/QUANTCONNECT_CLOUD_BRIDGE.md`
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

## Fail closed
Missing `QC_USER_ID` / `QC_API_TOKEN` is not a proxy license.
Write `FAIL_CLOSED_NO_CREDENTIALS` and point to the web-IDE paste path.

## Evidence rule
Never call a QC result `NAUTILUS_VERIFIED` or `RAW_BIDASK_PASS`.
