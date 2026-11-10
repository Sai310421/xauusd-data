"""Safe H1 sensor snapshot adapter for tick-driven RAGE research."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class SignalValidationError(ValueError):
    """Raised when an H1 macro signal cannot safely enter tick state."""


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise SignalValidationError(f"{field} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SignalValidationError(f"{field} must include UTC offset")
    return parsed.astimezone(UTC)


def _number(value: Any, field: str) -> float:
    if not isinstance(value, (float, int)):
        raise SignalValidationError(f"{field} must be numeric")
    return float(value)


def _source_hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SignalValidationError("invalid source_hash")
    if any(character not in "0123456789abcdef" for character in value):
        raise SignalValidationError("invalid source_hash")
    return value


@dataclass(frozen=True)
class H1MacroSignal:
    """Validated immutable H1 signal snapshot from the monitoring system."""

    signal_id: str
    as_of_utc: datetime
    valid_until_utc: datetime
    model_version: str
    portfolio_risk: float
    portfolio_regime: str
    psi_status: str
    source_hash: str

    @property
    def macro_crisis_score(self) -> float:
        """Normalize the 0..100 portfolio score once at the responsibility boundary."""
        return min(max(self.portfolio_risk / 100.0, 0.0), 1.0)


def parse_h1_signal(payload: dict[str, Any], now: datetime) -> H1MacroSignal:
    """Reject STOPPED, stale, unordered, or malformed sensor snapshots."""
    required = {
        "schema_version",
        "signal_id",
        "as_of_utc",
        "valid_until_utc",
        "signal_status",
        "model_version",
        "portfolio_risk",
        "portfolio_regime",
        "psi_status",
        "source_hash",
        "probability_ordering_pass",
        "feature_schema_pass",
        "manifest_integrity_pass",
    }
    missing = required.difference(payload)
    if missing:
        raise SignalValidationError(f"missing signal fields: {sorted(missing)}")
    if payload["schema_version"] != "h1-macro-signal/v1":
        raise SignalValidationError("unsupported schema_version")
    if payload["signal_status"] != "READY":
        raise SignalValidationError("SIGNAL_STOPPED")
    if not (
        payload["probability_ordering_pass"]
        and payload["feature_schema_pass"]
        and payload["manifest_integrity_pass"]
    ):
        raise SignalValidationError("SIGNAL_QUALITY_GATE_FAILED")
    as_of = _parse_time(payload["as_of_utc"], "as_of_utc")
    valid_until = _parse_time(payload["valid_until_utc"], "valid_until_utc")
    current = now.astimezone(UTC)
    if valid_until <= as_of or current > valid_until:
        raise SignalValidationError("HEDGE_INPUT_STALE")
    risk = _number(payload["portfolio_risk"], "portfolio_risk")
    if not 0.0 <= risk <= 100.0:
        raise SignalValidationError("portfolio_risk must be in [0, 100]")
    return H1MacroSignal(
        signal_id=str(payload["signal_id"]),
        as_of_utc=as_of,
        valid_until_utc=valid_until,
        model_version=str(payload["model_version"]),
        portfolio_risk=risk,
        portfolio_regime=str(payload["portfolio_regime"]),
        psi_status=str(payload["psi_status"]),
        source_hash=_source_hash(payload["source_hash"]),
    )


@dataclass
class SignalStateSynchronizer:
    """Hold each accepted H1 state across ticks without recalculating the sensor."""

    downward_smoothing_alpha: float = 0.25
    current: H1MacroSignal | None = None
    held_macro_score: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.downward_smoothing_alpha <= 1.0:
            raise SignalValidationError("downward_smoothing_alpha must be in (0, 1]")

    def accept(self, payload: dict[str, Any], now: datetime) -> H1MacroSignal:
        """Accept a new H1 snapshot; crisis escalation applies immediately."""
        signal = parse_h1_signal(payload, now)
        target = signal.macro_crisis_score
        if self.current is None or target >= self.held_macro_score:
            self.held_macro_score = target
        else:
            alpha = self.downward_smoothing_alpha
            self.held_macro_score += alpha * (target - self.held_macro_score)
        self.current = signal
        return signal

    def state_for_tick(self, now: datetime) -> H1MacroSignal:
        """Return held state for a tick or fail once the H1 snapshot expires."""
        if self.current is None:
            raise SignalValidationError("HEDGE_SIGNAL_UNAVAILABLE")
        if now.astimezone(UTC) > self.current.valid_until_utc:
            raise SignalValidationError("HEDGE_INPUT_STALE")
        return self.current
