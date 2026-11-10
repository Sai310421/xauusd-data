"""Protective hedge math with separated macro and execution-risk semantics."""
from __future__ import annotations

from dataclasses import dataclass


class HedgeInputError(ValueError):
    """Raised when an input violates the protective hedge safety contract."""


def _bounded(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise HedgeInputError(f"{name} must be in [0, 1]")
    return float(value)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def smoothstep(start: float, end: float, value: float) -> float:
    """Map a bounded macro score smoothly from zero to one."""
    if not 0.0 <= start < end <= 1.0:
        raise HedgeInputError("invalid macro thresholds")
    score = _bounded(value, "macro_crisis_score")
    scaled = _clip((score - start) / (end - start), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


@dataclass(frozen=True)
class ProtectiveHedgeParams:
    """Fixed non-execution configuration for macro-aware hedge sizing."""

    h_max: float
    lambda_macro: float
    rho_execution: float = 0.10
    macro_start_threshold: float = 0.30
    macro_full_threshold: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 < self.h_max <= 1.0:
            raise HedgeInputError("h_max must be in (0, 1]")
        if self.lambda_macro < 0.0 or self.rho_execution < 0.0:
            raise HedgeInputError("lambda_macro and rho_execution must be non-negative")
        if not 0.0 <= self.macro_start_threshold < self.macro_full_threshold <= 1.0:
            raise HedgeInputError("invalid macro threshold range")


@dataclass(frozen=True)
class HedgeIntensity:
    """Auditable decomposition of requested and executable hedge intensity."""

    macro_intensity: float
    requested_hedge_ratio: float
    executable_hedge_ratio: float
    final_hedge_ratio: float


def compute_hedge_intensity(
    base_hedge_ratio: float,
    macro_crisis_score: float,
    hedge_execution_risk: float,
    params: ProtectiveHedgeParams,
) -> HedgeIntensity:
    """Compute hedge intensity without using macro risk as execution cost."""
    if base_hedge_ratio < 0.0:
        raise HedgeInputError("base_hedge_ratio must be non-negative")
    macro = _bounded(macro_crisis_score, "macro_crisis_score")
    execution = _bounded(hedge_execution_risk, "hedge_execution_risk")
    macro_intensity = smoothstep(
        params.macro_start_threshold,
        params.macro_full_threshold,
        macro,
    )
    requested = min(
        params.h_max,
        base_hedge_ratio * (1.0 + params.lambda_macro * macro_intensity),
    )
    executable = requested / (1.0 + params.rho_execution * execution)
    final = _clip(executable, 0.0, params.h_max)
    return HedgeIntensity(macro_intensity, requested, executable, final)


def drawdown_proxy(peak_equity: float, current_equity: float) -> float:
    """Return a bounded drawdown proxy while rejecting invalid equity."""
    if peak_equity <= 0.0 or current_equity <= 0.0:
        raise HedgeInputError("equity inputs must be positive")
    return max((peak_equity - current_equity) / current_equity, 0.0)
