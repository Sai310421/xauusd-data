"""Discord notification adapter for advisory and fail-closed production events."""
from __future__ import annotations

from typing import Any

import requests


def stop_message(reason: str, as_of: str, action_required: str) -> str:
    """Render the fixed operational stop notification."""
    return (
        "USDJPY Crash Model\n\nSTATUS\n\nSTOPPED\n\nReason\n\n"
        f"{reason}\n\nAs Of\n\n{as_of}\n\nAction Required\n\n{action_required}"
    )


def send_discord(webhook_url: str, content: str, timeout_seconds: float = 10.0) -> None:
    """Deliver one non-trading notification or raise on an observable failure."""
    if not webhook_url.startswith("https://"):
        raise ValueError("DISCORD_WEBHOOK_URL_INVALID")
    response = requests.post(webhook_url, json={"content": content}, timeout=timeout_seconds)
    response.raise_for_status()


def notification_payload(reason: str, as_of: str, action_required: str) -> dict[str, Any]:
    """Build an inspectable payload before any HTTP side effect."""
    return {"content": stop_message(reason, as_of, action_required)}
