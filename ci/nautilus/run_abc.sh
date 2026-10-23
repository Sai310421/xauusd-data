#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export RUST_BACKTRACE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
trap 'code=$?; echo "AMOS_BT_FAILED step=${STEP:-unknown} exit=$code" >&2' ERR
mkdir -p results/amos-abc
STEP=build_catalog; echo "AMOS_STEP_START $STEP"; python -u research/build_raw_bidask_catalog_duka.py --start "${BT_START:-2026-10-19}" --days "${BT_DAYS:-1}" --catalog catalog/raw_bidask --symbols XAUUSD --workers 8 --fresh
STEP=guard; echo "AMOS_STEP_START $STEP"; python -u research/raw_bidask_catalog_guard.py --catalog catalog/raw_bidask --symbols XAUUSD
STEP=virtual_bt; echo "AMOS_STEP_START $STEP"; python -u research/amos_abc_nautilus_rawtick_bt.py --catalog catalog/raw_bidask --out results/amos-abc/rawtick-kpi.json
test -s results/amos-abc/rawtick-kpi.json

STEP=validate_virtual; echo "AMOS_STEP_START $STEP"; python -u research/validate_abc_rawtick_kpi.py --input results/amos-abc/rawtick-kpi.json

# Separate formal matching gate; does not replace virtual baseline.
STEP=native_matching; echo "AMOS_STEP_START $STEP"; python -u research/amos_abc_nautilus_native_matching.py --catalog catalog/raw_bidask --out results/amos-abc/native-matching.json
