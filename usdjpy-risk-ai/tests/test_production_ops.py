"""Production operation contract tests without external credentials or network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.discord import notification_payload, send_discord, stop_message


def test_stop_message_has_required_operational_fields() -> None:
    content = stop_message("STALE_DATA", "2026-01-01T00:00:00Z", "Review latest data")
    assert "STATUS" in content
    assert "STOPPED" in content
    assert "STALE_DATA" in content
    assert notification_payload("STALE_DATA", "2026-01-01T00:00:00Z", "Review latest data")["content"] == content


def test_discord_rejects_non_https_endpoint_before_network() -> None:
    with pytest.raises(ValueError, match="DISCORD_WEBHOOK_URL_INVALID"):
        send_discord("http://example.invalid", "ignored")
