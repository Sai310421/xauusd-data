from __future__ import annotations


def classify(n: int, expectancy: float, pf: float, max_dd: float, delta_expectancy: float | None = None, oos_positive: bool | None = None) -> dict:
    reasons = []
    if n < 30:
        return {"grade": "B", "reasons": ["sample_too_small"], "library_ready": False}
    if expectancy <= 0 or pf <= 1.0:
        return {"grade": "REJECT", "reasons": ["no_positive_standalone_edge"], "library_ready": False}
    if delta_expectancy is not None and delta_expectancy <= 0:
        return {"grade": "REJECT", "reasons": ["no_positive_delta_vs_control"], "library_ready": False}
    if oos_positive is False:
        return {"grade": "REJECT", "reasons": ["oos_failed"], "library_ready": False}
    if n >= 200 and pf >= 1.8 and oos_positive is True:
        reasons.append("strong_numeric_candidate")
        return {"grade": "S", "reasons": reasons, "library_ready": False}
    reasons.append("positive_but_more_robustness_needed")
    return {"grade": "A", "reasons": reasons, "library_ready": False}
