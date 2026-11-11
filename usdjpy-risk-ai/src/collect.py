"""Collect fixture or production H1 raw bars into monthly Parquet partitions."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data_provider import (
    DatabentoProvider,
    OandaProvider,
    normalize_raw,
    write_monthly_raw,
)


def _parse_utc(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamps must include UTC offset, e.g. 2026-01-01T00:00:00Z")
    return timestamp.to_pydatetime()


def _config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    """Run one explicit collection mode; production can never use fixture input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "production"), default="fixture")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--start", type=_parse_utc)
    parser.add_argument("--end", type=_parse_utc)
    args = parser.parse_args()
    if args.mode == "fixture":
        if args.input is None:
            parser.error("--input is required in fixture mode")
        raw = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    else:
        if args.input is not None:
            parser.error("--input is forbidden in production mode")
        if args.start is None or args.end is None:
            parser.error("--start and --end are required in production mode")
        config = _config(args.config)
        futures = DatabentoProvider.from_environment(config)
        fx = OandaProvider.from_environment(config)
        frames = [futures.fetch_bars(symbol, args.start, args.end, "1h") for symbol in ("6J", "VX", "ES", "NQ", "ZN")]
        frames.append(fx.fetch_bars("USDJPY", args.start, args.end, "1h"))
        raw = pd.concat(frames, ignore_index=True)
    paths = write_monthly_raw(normalize_raw(raw), args.output)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
