"""Final integration readiness gate with complete required-artifact inventory."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReadinessItem:
    name: str
    status: str
    reason: str


def _file_item(path: str, files: set[str], reason: str) -> ReadinessItem:
    return ReadinessItem(path, "PASS" if path in files else "BLOCKED", "present" if path in files else reason)


def integration_readiness(config: Mapping[str, Any], files: set[str], environment: Mapping[str, str]) -> list[ReadinessItem]:
    """Enumerate all conditions that must pass before final integration begins."""
    items: list[ReadinessItem] = []
    for secret in ("DATABENTO_API_KEY", "OANDA_API_TOKEN", "OCI_OBJECT_STORAGE_BUCKET", "DISCORD_WEBHOOK_URL"):
        items.append(ReadinessItem(secret, "PASS" if environment.get(secret) else "BLOCKED", "configured" if environment.get(secret) else "missing"))
    symbols = config.get("multiasset", {}).get("symbols", {})
    for source in ("RTY", "GC", "DXY", "US10Y"):
        status = "PASS" if source.lower() in symbols else "BLOCKED"
        items.append(ReadinessItem(f"DATA_SOURCE_{source}", status, "configured" if status == "PASS" else "not configured"))
    for path, reason in (
        ("src/predict.py", "inference entry point absent"),
        ("src/report.py", "report entry point absent"),
        ("src/discord.py", "Discord notifier absent"),
        ("src/shap_analysis.py", "SHAP analyzer absent"),
        (".github/workflows/usdjpy-risk-ai-train.yml", "training workflow absent"),
        (".github/workflows/usdjpy-risk-ai-predict.yml", "prediction workflow absent"),
    ):
        items.append(_file_item(path, files, reason))
    for name in ("target_05", "target_10", "target_20", "gold", "stock"):
        path = f"models/production/{name}.manifest.json"
        items.append(_file_item(path, files, "production manifest absent"))
    return items


def assert_integration_ready(items: list[ReadinessItem]) -> None:
    blocked = [item.name for item in items if item.status != "PASS"]
    if blocked:
        raise RuntimeError("INTEGRATION_NOT_READY: " + ", ".join(blocked))
