"""Fixture collection CLI for Phase 2."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_provider import normalize_raw, write_monthly_raw


def main() -> None:
    """Normalize a fixture file and write monthly raw Parquet partitions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = (
        pd.read_parquet(args.input)
        if args.input.suffix == ".parquet"
        else pd.read_csv(args.input)
    )
    frame = normalize_raw(source)
    paths = write_monthly_raw(frame, args.output)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
