"""Train non-production Gold and US-stock candidate LightGBM bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from .asset_features import (
    FEATURES_GOLD,
    FEATURES_STOCK,
    GOLD_TARGETS,
    STOCK_TARGETS,
    prepare_asset_training_rows,
)
from .asset_modeling import fit_asset_target


def _hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _train_group(frame: pd.DataFrame, name: str, features: list[str], targets: tuple[str, ...], directory: Path, minimum: int) -> None:
    """Train all thresholds for one asset group and write candidate manifests."""
    rows = prepare_asset_training_rows(frame, features, targets)
    for target in targets:
        positive = int(rows[target].sum())
        version = f"candidate-{name}-{target}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        manifest: dict[str, object] = {"version": version, "model_group": name, "target": target, "feature_schema": features, "feature_order": features, "positive_count": positive, "data_hash": _hash(rows[features + [target]]), "train_start": str(rows["datetime_utc"].min()), "train_end": str(rows["datetime_utc"].max()), "production_eligible": False}
        if positive >= minimum and rows[target].nunique() == 2:
            fitted = fit_asset_target(rows, target, features)
            joblib.dump(fitted.model, directory / f"{name}_{target}.model.pkl")
            joblib.dump(fitted.calibrator, directory / f"{name}_{target}.calibrator.pkl")
            manifest["status"] = "CANDIDATE_PENDING_WALKFORWARD"
            manifest["calibration_method"] = fitted.calibrator.method
        else:
            manifest["status"] = "BLOCKED_POSITIVE_COUNT"
        (directory / f"{name}_{target}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Train candidate-only asset models; promotion remains quality-gated."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--min-positive-samples", type=int, default=50)
    args = parser.parse_args()
    paths = sorted(args.input.glob("asset_features_*.parquet")) if args.input.is_dir() else [args.input]
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    args.models.mkdir(parents=True, exist_ok=True)
    _train_group(frame, "gold", FEATURES_GOLD, GOLD_TARGETS, args.models, args.min_positive_samples)
    _train_group(frame, "stock", FEATURES_STOCK, STOCK_TARGETS, args.models, args.min_positive_samples)


if __name__ == "__main__":
    main()
