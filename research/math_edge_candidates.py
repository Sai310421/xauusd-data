from __future__ import annotations

"""Reusable AE Math EDGE primitives.

IMPORTANT:
- Functions marked DERIVED are AE compositions, not verbatim source equations.
- They are deterministic utilities for A/B experiments; they are not evidence of edge.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class JointBounds:
    mfe_lo: float
    mfe_hi: float
    mae_lo: float
    mae_hi: float
    tau_ebe_lo: float
    tau_ebe_hi: float
    tail_lo: float
    tail_hi: float


def empirical_joint_box(
    residual_vectors: Sequence[Sequence[float]],
    point_forecast: Sequence[float],
    alpha: float = 0.10,
) -> JointBounds:
    """DERIVED proxy for a multivariate conformal reachable set.

    Uses coordinate-wise empirical residual quantiles as an implementation gate.
    This is NOT the general multi-variable conformal construction from the source
    literature; it is an intentionally simple baseline to test whether joint
    uncertainty information has economic value before a more exact method is added.
    """
    r = np.asarray(residual_vectors, dtype=float)
    p = np.asarray(point_forecast, dtype=float)
    if r.ndim != 2 or r.shape[1] != 4 or p.shape != (4,):
        raise ValueError("expected residual matrix Nx4 and point_forecast length 4")
    if len(r) < 20:
        raise ValueError("at least 20 calibration residuals required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    lo = p + np.quantile(r, alpha / 2.0, axis=0)
    hi = p + np.quantile(r, 1.0 - alpha / 2.0, axis=0)
    return JointBounds(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3])


def ae_safe_edge(
    b: JointBounds,
    lambda_mae: float = 1.0,
    lambda_tau: float = 0.01,
    lambda_tail: float = 1.0,
) -> float:
    """DERIVED AE action score from conservative joint bounds."""
    return float(
        b.mfe_lo
        - lambda_mae * max(0.0, b.mae_hi)
        - lambda_tau * max(0.0, b.tau_ebe_hi)
        - lambda_tail * max(0.0, b.tail_hi)
    )


def conformal_kelly_governor(
    kelly_fraction: float,
    width_norm: float,
    observed_miscoverage: float,
    target_alpha: float = 0.10,
    beta: float = 1.0,
    gamma: float = 4.0,
    f_min: float = 0.0,
    f_max: float = 1.0,
) -> float:
    """DERIVED AE uncertainty-scaled Kelly governor."""
    w = max(0.0, float(width_norm))
    excess = max(0.0, float(observed_miscoverage) - float(target_alpha))
    g = 1.0 / (1.0 + beta * w)
    h = math.exp(-gamma * excess)
    x = float(kelly_fraction) * g * h
    return float(min(max(x, f_min), f_max))


def wasserstein_like_lower_utility(
    empirical_utility: float,
    epsilon: float,
    lipschitz_bound: float,
) -> float:
    """DERIVED conservative screening bound.

    For 1-Wasserstein DRO, a Lipschitz utility admits a natural epsilon*L
    robustness penalty. This helper is only a screening proxy; it is NOT the
    certified LP approximation from the research paper.
    """
    if epsilon < 0 or lipschitz_bound < 0:
        raise ValueError("epsilon and lipschitz_bound must be non-negative")
    return float(empirical_utility - epsilon * lipschitz_bound)


def hurst_dd_multiplier(hurst: float, horizon: float) -> float:
    """DERIVED multiplicative correction T^(H-1/2)."""
    h = float(hurst)
    t = float(horizon)
    if not 0.0 < h < 1.0:
        raise ValueError("hurst must be in (0,1)")
    if t <= 0:
        raise ValueError("horizon must be positive")
    return float(t ** (h - 0.5))


def robust_sizing_from_scores(
    base_size: float,
    safe_edge_score: float,
    lower_utility: float,
    kelly_governor: float,
    dd_multiplier: float = 1.0,
    min_scale: float = 0.0,
    max_scale: float = 1.25,
) -> float:
    """DERIVED composition used only for explicit COMPOSED experiments."""
    if dd_multiplier <= 0:
        raise ValueError("dd_multiplier must be positive")
    signal_gate = 1.0 / (1.0 + math.exp(-safe_edge_score))
    utility_gate = 1.0 / (1.0 + math.exp(-lower_utility))
    scale = signal_gate * utility_gate * max(0.0, kelly_governor) / dd_multiplier
    scale = min(max(scale, min_scale), max_scale)
    return float(base_size * scale)
