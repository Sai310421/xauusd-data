"""Optuna is not imported.

A deterministic grid is the stand-in so the seam exists without pretending
a study ran. If optuna is added later, it must not retune on the OOS slice.
"""

from __future__ import annotations


def grid_minimize(space: dict[str, list[float]], score) -> dict[str, float]:
    if not space:
        raise ValueError("empty space")
    keys = list(space)
    best_params: dict[str, float] | None = None
    best_score = float("inf")

    def walk(index: int, current: dict[str, float]) -> None:
        nonlocal best_params, best_score
        if index == len(keys):
            value = score(current)
            if value < best_score:
                best_score = value
                best_params = dict(current)
            return
        key = keys[index]
        for value in space[key]:
            current[key] = value
            walk(index + 1, current)

    walk(0, {})
    if best_params is None:
        raise ValueError("space produced no point")
    return best_params
