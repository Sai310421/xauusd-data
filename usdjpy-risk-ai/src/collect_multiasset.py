"""Collect the fixed multi-asset H1 universe into Parquet partitions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .multiasset_provider import (
    MULTIASSET_ASSETS,
    MultiAssetDatabentoProvider,
    write_monthly_multiasset,
)


def main() -> None:
    """Run a bounded, explicit multi-asset collection request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    provider = MultiAssetDatabentoProvider.from_environment(config)
    start, end = pd.Timestamp(args.start).to_pydatetime(), pd.Timestamp(args.end).to_pydatetime()
    frame = pd.concat([provider.fetch_bars(asset, start, end) for asset in MULTIASSET_ASSETS], ignore_index=True)
    print("\n".join(str(path) for path in write_monthly_multiasset(frame, args.output)))
