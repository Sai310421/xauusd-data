#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p results/amos-abc
python research/build_raw_bidask_catalog_duka.py --start "${BT_START:-2026-10-19}" --days "${BT_DAYS:-1}" --catalog catalog/raw_bidask --symbols XAUUSD --workers 8 --fresh
python research/raw_bidask_catalog_guard.py --catalog catalog/raw_bidask --symbols XAUUSD
python research/amos_abc_nautilus_rawtick_bt.py --catalog catalog/raw_bidask --out results/amos-abc/rawtick-kpi.json
test -s results/amos-abc/rawtick-kpi.json

python research/validate_abc_rawtick_kpi.py --input results/amos-abc/rawtick-kpi.json

# Separate formal matching gate; does not replace virtual baseline.
python research/amos_abc_nautilus_native_matching.py --catalog catalog/raw_bidask --out results/amos-abc/native-matching.json
