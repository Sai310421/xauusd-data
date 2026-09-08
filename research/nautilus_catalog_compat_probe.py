from __future__ import annotations

import argparse
import json
from pathlib import Path

from nautilus_trader.persistence.catalog import ParquetDataCatalog

from nautilus_catalog_compat import validate_raw_quote_ticks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--output", default="results/nautilus-compat/catalog_probe.json")
    args = ap.parse_args()

    catalog = ParquetDataCatalog(str(Path(args.catalog)))
    report = validate_raw_quote_ticks(catalog, symbol=args.symbol, min_ticks=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
