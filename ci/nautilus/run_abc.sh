#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export RUST_BACKTRACE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
trap 'code=$?; echo "AMOS_BT_FAILED step=${STEP:-unknown} exit=$code" >&2' ERR
mkdir -p results/amos-abc
BT_MODE="${BT_MODE:-SMOKE_1D}"
case "$BT_MODE" in
  SMOKE_1D) BT_DAYS="${BT_DAYS:-1}" ;;
  RESEARCH_90D) BT_DAYS="${BT_DAYS:-90}"; test "$BT_DAYS" -eq 90 || exit 2 ;;
  *) echo "Unknown BT_MODE: $BT_MODE" >&2; exit 2 ;;
esac
export BT_MODE BT_DAYS
python - <<'PY'
import json, os
from pathlib import Path
manifest={"schema_version":1,"mode":os.environ["BT_MODE"],"start":os.getenv("BT_START","2026-10-19"),"days":int(os.environ["BT_DAYS"]),"symbol":"XAUUSD","source":"RAW_BIDASK_QUOTETICK","live_orders":False,"native_kpi":"NOT_MEASURED","oos_wfo_mc":"NOT_IMPLEMENTED"}
Path("results/amos-abc/run-manifest.json").write_text(json.dumps(manifest,indent=2))
PY
STEP=build_catalog; echo "AMOS_STEP_START $STEP"; python -u research/build_raw_bidask_catalog_duka.py --start "${BT_START:-2026-10-19}" --days "$BT_DAYS" --catalog catalog/raw_bidask --symbols XAUUSD --workers 8 --fresh
STEP=guard; echo "AMOS_STEP_START $STEP"; python -u research/raw_bidask_catalog_guard.py --catalog catalog/raw_bidask --symbols XAUUSD
STEP=virtual_bt; echo "AMOS_STEP_START $STEP"; python -u research/amos_abc_nautilus_rawtick_bt.py --catalog catalog/raw_bidask --out results/amos-abc/rawtick-kpi.json
test -s results/amos-abc/rawtick-kpi.json

STEP=validate_virtual; echo "AMOS_STEP_START $STEP"; python -u research/validate_abc_rawtick_kpi.py --input results/amos-abc/rawtick-kpi.json

# Separate formal matching gate; does not replace virtual baseline.
for edge in A01 B01 C01; do
  STEP="native_matching_${edge}"; echo "AMOS_STEP_START $STEP"
  python -u research/amos_abc_nautilus_native_matching.py --catalog catalog/raw_bidask --edge "$edge" --out "results/amos-abc/native-matching-${edge}.json"
done

STEP=validate_native; echo "AMOS_STEP_START $STEP"
python - <<'PY'
import json
from pathlib import Path
root=Path("results/amos-abc")
summary={}
for edge in ("A01","B01","C01"):
    data=json.loads((root/f"native-matching-{edge}.json").read_text())
    assert data["verification"]=="NAUTILUS_NATIVE_MATCHING_RAW_QUOTETICK"
    assert data["raw_ticks"]>=1000
    entry=data["edges"][edge]
    assert entry["native_fills"]>=1,edge
    summary[edge]=entry
(root/"native-gate-summary.json").write_text(json.dumps({"gate":"PASS","native_kpi":"NOT_MEASURED","edges":summary},indent=2))
manifest_path=root/"run-manifest.json"
manifest=json.loads(manifest_path.read_text())
manifest["raw_ticks"]=json.loads((root/"rawtick-kpi.json").read_text())["raw_ticks"]
manifest["virtual_gate"]="PASS"
manifest["native_fill_gate"]="PASS"
manifest["research_kpi_gate"]="NOT_EVALUATED"
manifest_path.write_text(json.dumps(manifest,indent=2))
PY
