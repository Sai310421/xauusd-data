from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class OverlayEligibility:
    eligible: bool
    reasons: list[str]
    monthly_n: float
    pf: float
    wr_pct: float
    base_expectancy: float | None = None

    def to_dict(self): return asdict(self)


def assess(monthly_n: float, pf: float, wr_pct: float, base_expectancy: float | None = None,
           min_n: float = 100.0, min_pf: float = 1.0, min_wr: float = 65.0) -> OverlayEligibility:
    """Screen only: qualifies a base EDGE for a separate 3-split + G75 overlay experiment.

    This does NOT claim the overlay is profitable. The overlay must be independently
    Raw-Tick/OOS tested against the unsplit base and a 3-split-only control.
    """
    reasons=[]
    if monthly_n < min_n: reasons.append('monthly_n_below_gate')
    if pf < min_pf: reasons.append('pf_below_gate')
    if wr_pct < min_wr: reasons.append('wr_below_gate')
    if base_expectancy is not None and base_expectancy <= 0: reasons.append('nonpositive_base_expectancy')
    return OverlayEligibility(not reasons,reasons,monthly_n,pf,wr_pct,base_expectancy)


OVERLAY_ABLATION = (
    'BASE',
    'BASE_SPLIT3',
    'BASE_G75',
    'BASE_SPLIT3_G75',
)

# Required comparison before any high-return/stability claim:
# BASE vs BASE_SPLIT3 vs BASE_G75 vs BASE_SPLIT3_G75.
# Measure N, WR, PF, expectancy, MaxDD, tail loss, cost elasticity, OOS stability,
# max simultaneous exposure/layers and margin stress. Splitting one signal into
# three child orders is not counted as tripling independent N.
