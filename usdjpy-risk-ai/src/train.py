"""Candidate-only training with immutable-storage compatible manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURES
from .modeling import fit_target


def _hash_features(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--min-positive-samples", type=int, default=50)
    args = parser.parse_args()
    paths = sorted(args.input.glob("features_*.parquet")) if args.input.is_dir() else [args.input]
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    args.models.mkdir(parents=True, exist_ok=True)
    for target in ("target_05", "target_10", "target_20"):
        positive = int(frame[target].sum())
        version = f"candidate-{target}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        manifest: dict[str, object] = {
            "version": version, "target": target, "created_at": datetime.now(UTC).isoformat(),
            "feature_schema": FEATURES, "feature_order": FEATURES, "positive_count": positive,
            "data_hash": _hash_features(frame[FEATURES + [target]]),
            "train_start": str(frame["datetime_utc"].min()), "train_end": str(frame["datetime_utc"].max()),
            "metrics": {}, "psi_reference_uri": "oci://pending/psi/reference.parquet",
            "config_version": "v1.3.2-phase5", "production_eligible": False,
        }
        if positive >= args.min_positive_samples and frame[target].nunique() == 2:
            fitted = fit_target(frame, target)
            model_path = args.models / f"{target}.model.pkl"
            calibrator_path = args.models / f"{target}.calibrator.pkl"
            joblib.dump(fitted.model, model_path)
            joblib.dump(fitted.calibrator, calibrator_path)
            manifest.update({
                "model_uri": f"oci://pending/models/{version}/{target}.model.pkl",
                "calibrator_uri": f"oci://pending/models/{version}/{target}.calibrator.pkl",
                "model_sha256": _hash_file(model_path), "calibrator_sha256": _hash_file(calibrator_path),
                "calibration_method": fitted.calibrator.method,
                "status": "CANDIDATE_PENDING_OCI_UPLOAD",
            })
        else:
            manifest["status"] = "BLOCKED_POSITIVE_COUNT"
        (args.models / f"{target}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
