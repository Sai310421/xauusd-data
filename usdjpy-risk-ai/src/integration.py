"""Explicit final-integration readiness gate; no silent partial deployment."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReadinessItem:
    """One prerequisite and its deterministic integration status."""

    name: str
    status: str
    reason: str


def integration_readiness(
    config: Mapping[str, Any],
    files: set[str],
    environment: Mapping[str, str],
) -> list[ReadinessItem]:
    """List every blocking prerequisite before final end-to-end verification."""
    items: list[ReadinessItem] = []
    required_secrets = ("DATABENTO_API_KEY", "OANDA_API_TOKEN", "OCI_OBJECT_STORAGE_BUCKET")
    for secret in required_secrets:
        items.append(ReadinessItem(secret, "PASS" if environment.get(secret) else "BLOCKED", "configured" if environment.get(secret) else "missing"))
    for source in ("RTY", "GC", "DXY", "US10Y"):
        status = "PASS" if source.lower() in config.get("multiasset", {}).get("symbols", {}) else "BLOCKED"
        items.append(ReadinessItem(f"DATA_SOURCE_{source}", status, "configured" if status == "PASS" else "not configured"))
    for path in ("src/predict.py", "src/report.py"):
        status = "PASS" if path in files else "BLOCKED"
        items.append(ReadinessItem(path, status, "present" if status == "PASS" else "not implemented"))
    for name in ("gold", "stock"):
        path = f"models/production/{name}.manifest.json"
        status = "PASS" if path in files else "BLOCKED"
        items.append(ReadinessItem(path, status, "present" if status == "PASS" else "production manifest absent"))
    return items


def assert_integration_ready(items: list[ReadinessItem]) -> None:
    """Prevent final integration from proceeding while any blocker remains."""
    blocked = [item.name for item in items if item.status != "PASS"]
    if blocked:
        raise RuntimeError("INTEGRATION_NOT_READY: " + ", ".join(blocked))
