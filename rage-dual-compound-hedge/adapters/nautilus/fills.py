"""Native fill log.

This module does not emit fills. A fill count of zero means the engine
did not run, not that a strategy was flat.
"""

from __future__ import annotations


def native_fill_count() -> int:
    return 0


def require_engine() -> None:
    raise RuntimeError(
        "NautilusTrader is not attached in this package. "
        "Reuse research/build_raw_bidask_catalog_duka.py and the existing "
        "raw-bidask workflow. Do not substitute OHLC."
    )
