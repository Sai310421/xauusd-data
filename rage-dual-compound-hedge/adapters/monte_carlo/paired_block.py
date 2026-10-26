"""Paired block bootstrap.

Main and hedge are cut on the same blocks so a common shock stays common.
independent_shuffle exists only as a negative example. Official paths must
not call it.
"""

from __future__ import annotations

import random


def paired_block_paths(
    main: list[float],
    hedge: list[float],
    block: int,
    n_paths: int,
    horizon: int,
    seed: int,
) -> list[tuple[list[float], list[float]]]:
    if len(main) != len(hedge) or not main:
        raise ValueError("aligned non-empty series required")
    if block < 1 or block > len(main) or horizon < 1 or n_paths < 1:
        raise ValueError("invalid block, horizon, or path count")
    rng = random.Random(seed)
    n = len(main)
    paths = []
    for _ in range(n_paths):
        pm: list[float] = []
        ph: list[float] = []
        while len(pm) < horizon:
            start = rng.randrange(0, n - block + 1)
            pm.extend(main[start : start + block])
            ph.extend(hedge[start : start + block])
        paths.append((pm[:horizon], ph[:horizon]))
    return paths


def independent_shuffle(
    main: list[float],
    hedge: list[float],
    seed: int,
) -> tuple[list[float], list[float]]:
    """Breaks the common shock. Do not use for the research claim."""
    rng = random.Random(seed)
    shuffled = list(hedge)
    rng.shuffle(shuffled)
    return list(main), shuffled
