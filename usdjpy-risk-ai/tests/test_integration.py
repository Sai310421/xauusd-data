"""Final integration gate regression test."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integration import assert_integration_ready, integration_readiness


def test_missing_operational_components_block_final_integration() -> None:
    items = integration_readiness({}, {"src/predict.py", "src/report.py"}, {})
    blocked = {item.name for item in items if item.status == "BLOCKED"}
    assert {"DISCORD_WEBHOOK_URL", "src/discord.py", "src/shap_analysis.py", ".github/workflows/usdjpy-risk-ai-train.yml", "models/production/target_05.manifest.json"}.issubset(blocked)
    with pytest.raises(RuntimeError, match="INTEGRATION_NOT_READY"):
        assert_integration_ready(items)
