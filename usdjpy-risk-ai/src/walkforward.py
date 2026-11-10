"""Walk-forward CLI; unavailable folds are reported, never silently skipped."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .modeling import walk_forward


def main() -> None:
    """Write deterministic walk-forward records from feature Parquet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture-n-estimators", type=int, default=20)
    args = parser.parse_args()
    paths = sorted(args.input.glob("features_*.parquet")) if args.input.is_dir() else [args.input]
    features = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    report = walk_forward(features, n_estimators=args.fixture_n_estimators)
    args.output.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output / "walkforward.csv", index=False)


if __name__ == "__main__":
    main()
