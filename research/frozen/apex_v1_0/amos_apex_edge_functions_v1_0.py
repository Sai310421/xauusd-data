"""
AMOS Apex Recognition Edge v1.0
Pure Python reference functions.
No broker/MT5 dependency.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ApexInputs:
    rci_short: float
    rci_mid: float
    rci_long: float
    cci: float
    rsi: float
    cci_reversal: bool
    price_turn: bool
    deviation_confirmed: bool = False
    deviation_enabled: bool = False

@dataclass
class ApexConfig:
    rci_level: float = 90.0
    rci_min_count: int = 2
    cci_level: float = 200.0
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    min_score: float = 4.0
    require_price_turn: bool = True
    nanpin_min_score: float = 3.0
    max_risk_stage: int = 7
    grid_expansion: float = 0.06

@dataclass
class ApexResult:
    side: str
    score: float
    rci_count: int
    candidate: bool
    high_confidence: bool

def rci_extreme_count(x: ApexInputs, side: str, cfg: ApexConfig) -> int:
    r = (x.rci_short, x.rci_mid, x.rci_long)
    if side.upper() == "BUY":
        return sum(v <= -cfg.rci_level for v in r)
    return sum(v >= cfg.rci_level for v in r)

def apex_score(x: ApexInputs, side: str, cfg: ApexConfig) -> ApexResult:
    side = side.upper()
    n = rci_extreme_count(x, side, cfg)
    score = 0.0
    if n >= cfg.rci_min_count:
        score += 1.0
    if n >= 3:
        score += 1.0

    cci_extreme = x.cci <= -cfg.cci_level if side == "BUY" else x.cci >= cfg.cci_level
    rsi_extreme = x.rsi <= cfg.rsi_low if side == "BUY" else x.rsi >= cfg.rsi_high

    score += float(cci_extreme)
    score += float(rsi_extreme)
    score += float(x.cci_reversal)
    score += float(x.price_turn)
    if x.deviation_enabled:
        score += float(x.deviation_confirmed)

    candidate = (
        score >= cfg.min_score and
        (x.price_turn if cfg.require_price_turn else True)
    )
    high = (
        score >= max(cfg.min_score + 1.0, 5.0)
        and n >= cfg.rci_min_count
        and x.cci_reversal
        and x.price_turn
    )
    return ApexResult(side, score, n, candidate, high)

def allow_nanpin(result: ApexResult, stage: int, cfg: ApexConfig) -> bool:
    return (
        stage < cfg.max_risk_stage
        and result.candidate
        and result.score >= cfg.nanpin_min_score
    )

def dynamic_grid(base_grid_pips: float, current_positions: int, expansion: float = 0.06) -> float:
    n = max(0, current_positions - 1)
    return base_grid_pips * ((1.0 + max(0.0, expansion)) ** n)

def martingale_lot(base_lot: float, multiplier: float, n: int) -> float:
    return base_lot * (multiplier ** n)

def adjusted_lot(base_lot: float,
                 logic_b: bool = False,
                 deviation_met: bool = True,
                 logic_b_multiplier: float = 0.5,
                 deviation_miss_multiplier: float = 0.7) -> float:
    lot = base_lot
    if logic_b:
        lot *= logic_b_multiplier
    if not deviation_met:
        lot *= deviation_miss_multiplier
    return lot

def peak_profit_exit(current_profit_pips: float,
                     max_profit_pips: float,
                     trigger: float,
                     trail_width: float):
    max_profit_pips = max(max_profit_pips, current_profit_pips)
    exit_now = (
        max_profit_pips >= trigger
        and current_profit_pips <= max_profit_pips - trail_width
    )
    return max_profit_pips, exit_now
