"""Show the splitter on a length, not on a market. OOS gate stays blocked."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.wfo.splitter import walk_forward  # noqa: E402


def main() -> int:
    folds = walk_forward(120, train=60, test=20, step=20, embargo=5)
    print(
        json.dumps(
            {
                "status": "BLOCKED",
                "reason": "index splitter only; no unused market window scored",
                "folds": [
                    {"train": [fold.train.start, fold.train.stop], "test": [fold.test.start, fold.test.stop]}
                    for fold in folds
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
