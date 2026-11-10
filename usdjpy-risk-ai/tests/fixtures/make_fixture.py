"""Create deterministic H1 fixture raw data for CI."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ("6J", "VX", "ES", "NQ", "ZN", "USDJPY")


def make_fixture(periods: int = 850) -> pd.DataFrame:
    """Return complete deterministic raw bars for all required symbols."""
    rng = np.random.default_rng(42)
    starts = pd.date_range("2023-01-01", periods=periods, freq="h", tz="UTC")
    base = {"6J": 0.009, "VX": 20.0, "ES": 4000.0, "NQ": 12000.0, "ZN": 110.0, "USDJPY": 130.0}
    rows: list[dict[str, object]] = []
    for offset, symbol in enumerate(SYMBOLS):
        innovations = rng.normal(0, 0.002 + offset * 0.0001, periods)
        prices = base[symbol] * np.exp(np.cumsum(innovations))
        for index, timestamp in enumerate(starts):
            rows.append({
                "symbol": symbol,
                "source_bar_start_utc": timestamp.isoformat(),
                "close": float(prices[index]),
                "volume": float(1000 + 20 * offset + (index % 31)),
                "provider": "fixture",
                "dataset": "fixture",
                "actual_contract": symbol,
                "instrument_id": f"{symbol}.FIXTURE",
                "roll_event": False,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_fixture().to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
