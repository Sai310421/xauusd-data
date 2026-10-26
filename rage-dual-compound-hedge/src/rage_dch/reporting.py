"""Research report. Statuses are only PASS, CONDITIONAL, FAIL, BLOCKED.

PASS is not available for the 10% hypothesis in this module: no raw-tick
fill log is produced here.
"""

from __future__ import annotations

from rage_dch.hedge_optimizer import required_monthly_edge

GATES = (
    "MAIN_10_PERCENT_COMPOUND",
    "HEDGE_10_PERCENT_COMPOUND",
    "SIMULTANEOUS_LOSS_CONTROL",
    "PROTECTION_COVERAGE",
    "PORTFOLIO_POSITIVE_GROWTH",
    "OOS_REPRODUCIBILITY",
    "MONTE_CARLO_ROBUSTNESS",
)


def structural_requirements(cost_rate: float = 0.01) -> list[dict[str, float | None]]:
    rows = []
    for step in range(0, 17):
        locked = step / 20.0
        rows.append(
            {
                "locked_fraction": locked,
                "cost_rate": cost_rate,
                "required_monthly_edge_on_free_sleeve": required_monthly_edge(locked, cost_rate, 0.10),
            }
        )
    return rows


def empty_market_report() -> dict:
    """What is allowed to be said before a raw bid/ask catalog is read."""
    edge_30 = required_monthly_edge(0.30, 0.01, 0.10)
    return {
        "title": "AMOS RAGE DCH RESEARCH REPORT",
        "live_trading": False,
        "formula_version_baseline": "v1-original",
        "formula_version_candidate": "v2-budget-split",
        "raw_tick_count": None,
        "raw_tick_span": None,
        "nautilus_native_fills": 0,
        "main_monthly_net": None,
        "hedge_monthly_net": None,
        "total_monthly_net": None,
        "max_floating_dd": None,
        "simultaneous_loss_rate_market": None,
        "protection_shortfall_market": None,
        "oos_wfo": None,
        "monte_carlo_market": None,
        "structural_note": (
            "If 30% of the hedge book is locked and that lock costs 1% a month, "
            "the free 70% must earn more than 10% for the whole hedge book to compound at 10%. "
            f"Required free-sleeve monthly edge at that point: {edge_30:.6f}. "
            "ProtectiveHedge does not supply this edge."
        ),
        "structural_required_edge": structural_requirements(0.01),
        "hypothesis": {
            "MAIN_10_PERCENT_COMPOUND": "BLOCKED",
            "HEDGE_10_PERCENT_COMPOUND": "BLOCKED",
            "SIMULTANEOUS_LOSS_CONTROL": "CONDITIONAL",
            "PROTECTION_COVERAGE": "CONDITIONAL",
            "PORTFOLIO_POSITIVE_GROWTH": "BLOCKED",
            "OOS_REPRODUCIBILITY": "BLOCKED",
            "MONTE_CARLO_ROBUSTNESS": "CONDITIONAL",
        },
        "hypothesis_why": {
            "MAIN_10_PERCENT_COMPOUND": "No raw bid/ask fill log. Existing ae_/armada_/bigplayer_ scripts were not executed.",
            "HEDGE_10_PERCENT_COMPOUND": "No hedge alpha is implemented. A pure offset sleeve pays costs and cannot be the 10% engine.",
            "SIMULTANEOUS_LOSS_CONTROL": "Paired-block method exists and keeps the common shock. Market paths were not resampled.",
            "PROTECTION_COVERAGE": "Scenario optimizer and shortfall accounting exist. Coverage on XAUUSD ticks was not measured.",
            "PORTFOLIO_POSITIVE_GROWTH": "Blocked with the two compound gates. Identity P_total = P_A + P_R + P_I - C is implemented only.",
            "OOS_REPRODUCIBILITY": "Walk-forward splitter is tested on index ranges. No unused market window was scored.",
            "MONTE_CARLO_ROBUSTNESS": "Independent shuffling is rejected in code. Robustness on real paired returns is blocked.",
        },
    }
