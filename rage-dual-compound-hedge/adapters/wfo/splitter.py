"""Walk-forward index splitter. Test rows are never also train rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train: range
    test: range


def walk_forward(n: int, train: int, test: int, step: int, embargo: int = 0) -> list[Fold]:
    if min(train, test, step) < 1 or embargo < 0:
        raise ValueError("train, test, and step must be positive")
    if train + embargo + test > n:
        return []
    folds: list[Fold] = []
    start = 0
    while start + train + embargo + test <= n:
        train_range = range(start, start + train)
        test_range = range(start + train + embargo, start + train + embargo + test)
        if not set(train_range).isdisjoint(test_range):
            raise RuntimeError("leakage")
        folds.append(Fold(train_range, test_range))
        start += step
    return folds
