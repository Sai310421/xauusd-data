"""Readiness gate tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integration import assert_integration_ready, integration_readiness


def test_readiness_lists_missing_final_integration_dependencies() -> None:
    items = integration_readiness({}, set(), {})
    names = {item.name for item in items if item.status == "BLOCKED"}
    assert {"DATABENTO_API_KEY", "DATA_SOURCE_GC", "src/predict.py"}.issubset(names)
    with pytest.raises(RuntimeError, match="INTEGRATION_NOT_READY"):
        assert_integration_ready(items)


def test_readiness_passes_when_all_required_contracts_are_present() -> None:
    config = {"multiasset": {"symbols": {name.lower(): name for name in ("RTY", "GC", "DXY", "US10Y")}}}
    files = {"src/predict.py", "src/report.py", "models/production/gold.manifest.json", "models/production/stock.manifest.json"}
    items = integration_readiness(config, files, {"DATABENTO_API_KEY":"x", "OANDA_API_TOKEN":"x", "OCI_OBJECT_STORAGE_BUCKET":"x"})
    assert_integration_ready(items)
