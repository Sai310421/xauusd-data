from __future__ import annotations

"""AE multi-filtration e-process evidence governor (DERIVED).

This module is intentionally small and auditable. It does not claim that an e-process
creates trading expectancy. It converts bounded per-stream evidence into nonnegative
multiplicative e-processes and fuses coarse/fine streams conservatively.
"""

from dataclasses import dataclass, field
import math
from typing import Dict


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


def bounded_signal(score: float, center: float = 0.5, scale: float = 0.5) -> float:
    """Map a probability/quality score to Z in [-1,1]. AE DERIVED transform."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return _clip((float(score) - center) / scale, -1.0, 1.0)


@dataclass
class EProcess:
    bet_fraction: float = 0.25
    value: float = 1.0
    running_max: float = 1.0

    def update(self, z: float) -> float:
        lam = _clip(self.bet_fraction, 0.0, 0.999)
        zz = _clip(z, -1.0, 1.0)
        self.value *= max(1e-12, 1.0 + lam * zz)
        self.running_max = max(self.running_max, self.value)
        return self.value


def log_adjuster(e_max: float, c: float = 0.5) -> float:
    """Conservative monotone coarse-filtration adjuster proxy.

    AE DERIVED. The implementation deliberately grows sublinearly with e_max to avoid
    naive over-counting of coarse evidence. This is not presented as the unique
    source-paper adjuster.
    """
    x = max(float(e_max), 1.0)
    return 1.0 + c * math.log(x)


@dataclass
class MultiFiltrationGovernor:
    streams: Dict[str, EProcess] = field(default_factory=dict)
    fine_stream: str = "M1"
    fusion_weight: float = 0.5
    min_evidence: float = 1.05

    def ensure(self, name: str, bet_fraction: float = 0.25) -> EProcess:
        if name not in self.streams:
            self.streams[name] = EProcess(bet_fraction=bet_fraction)
        return self.streams[name]

    def update(self, name: str, score: float, center: float = 0.5, scale: float = 0.5) -> float:
        return self.ensure(name).update(bounded_signal(score, center, scale))

    def fused_value(self) -> float:
        if not self.streams:
            return 1.0
        w = _clip(self.fusion_weight, 0.0, 1.0)
        fine = self.streams.get(self.fine_stream)
        fine_value = fine.value if fine is not None else 1.0
        coarse_vals = [log_adjuster(ep.running_max) for k, ep in self.streams.items() if k != self.fine_stream]
        coarse = sum(coarse_vals) / len(coarse_vals) if coarse_vals else 1.0
        # Convex fusion keeps the governor interpretable and prevents raw products.
        return w * fine_value + (1.0 - w) * coarse

    def allow_entry(self) -> bool:
        return self.fused_value() >= self.min_evidence

    def snapshot(self) -> dict:
        return {
            "fused_e": self.fused_value(),
            "min_evidence": self.min_evidence,
            "allow_entry": self.allow_entry(),
            "streams": {k: {"e": v.value, "e_max": v.running_max} for k, v in self.streams.items()},
        }
