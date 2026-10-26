"""ProtectiveHedge baseline (original formula preserved).

Do not assume this formula guarantees protection.
Validate under Raw Tick, costs, and real market conditions.
"""
from typing import Dict
import math

def compute_hedge_intensity(
    sigma_t: float,
    sigma_ref: float,
    D_t: float,
    Risk_t: float,
    delta: float,
    dt: float,
    kappa1: float,
    kappa2: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    tau: float,
    rho: float,
    H_max: float,
) -> float:
    """Original ProtectiveHedge formula."""
    w_t = 1.0 / (1.0 + math.exp(
        -kappa1 * (sigma_t / sigma_ref - 1.0) - kappa2 * D_t
    ))
    R_t = (
        lambda1 * D_t
        + lambda2 * max((sigma_t - sigma_ref) / sigma_ref, 0.0)
        + lambda3 * sigma_t * math.sqrt(tau)
    )
    H_raw = (w_t * R_t * math.exp(-delta * dt)) / (1.0 + rho * Risk_t)
    return max(0.0, min(H_raw, H_max))

def drawdown_proxy(K: float, S_t: float) -> float:
    return max((K - S_t) / S_t, 0.0) if S_t > 0 else 0.0
