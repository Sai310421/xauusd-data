"""Generate an advisory-only portfolio hedge signal from calibrated asset models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .asset_features import FEATURES_GOLD, FEATURES_STOCK
from .asset_modeling import FittedAssetTarget
from .portfolio_engine import (
    PortfolioAllocation,
    PortfolioEngineError,
    portfolio_advisory_from_models,
)


def _json(path: Path) -> dict[str, Any]:
    """Read one explicit JSON input contract."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fitted(
    model_path: Path,
    calibrator_path: Path,
    features: list[str],
) -> FittedAssetTarget:
    """Rebuild a schema-pinned fitted target from two serialized artifacts."""
    return FittedAssetTarget(
        model=joblib.load(model_path),
        calibrator=joblib.load(calibrator_path),
        feature_order=features,
    )


def main() -> None:
    """Write a portfolio advisory JSON; no order-routing capability exists here."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--gold-model", required=True, type=Path)
    parser.add_argument("--gold-calibrator", required=True, type=Path)
    parser.add_argument("--gold-version", required=True)
    parser.add_argument("--stock-model", required=True, type=Path)
    parser.add_argument("--stock-calibrator", required=True, type=Path)
    parser.add_argument("--stock-version", required=True)
    parser.add_argument("--psi", required=True, type=Path)
    parser.add_argument("--allocation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        row = pd.read_parquet(args.features).sort_values("datetime_utc").iloc[[-1]]
        allocation_data = _json(args.allocation)
        allocation = PortfolioAllocation(
            gold_weight=float(allocation_data["gold_weight"]),
            stock_weight=float(allocation_data["stock_weight"]),
            cash_weight=float(allocation_data["cash_weight"]),
        )
        advisory = portfolio_advisory_from_models(
            _load_fitted(args.gold_model, args.gold_calibrator, FEATURES_GOLD),
            _load_fitted(args.stock_model, args.stock_calibrator, FEATURES_STOCK),
            row,
            row,
            allocation,
            _json(args.psi),
            args.gold_version,
            args.stock_version,
        )
        output: dict[str, object] = {"status": "OK", **advisory.as_dict()}
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    except (FileNotFoundError, KeyError, TypeError, ValueError, PortfolioEngineError) as error:
        args.output.write_text(
            json.dumps({"status": "STOPPED", "reason": str(error)}),
            encoding="utf-8",
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
