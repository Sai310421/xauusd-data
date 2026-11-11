"""Fail-closed local inference entry point for promoted USDJPY model bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import FEATURES
from .modeling import CalibrationArtifact
from .safety import (
    PredictionStopped,
    regime,
    validate_prediction_input,
    validate_probability_order,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Predict only from three explicitly promoted local model manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    row = pd.read_parquet(args.features).sort_values("datetime_utc").iloc[-1]
    probabilities: dict[str, float] = {}
    manifests: dict[str, dict[str, Any]] = {}
    try:
        for target in ("target_05", "target_10", "target_20"):
            manifest = _load(args.models / f"{target}.manifest.json")
            if manifest.get("status") != "PRODUCTION":
                raise PredictionStopped("MANIFEST_INVALID")
            validate_prediction_input(row, manifest)
            model = joblib.load(args.models / f"{target}.model.pkl")
            calibrator: CalibrationArtifact = joblib.load(args.models / f"{target}.calibrator.pkl")
            raw = model.predict_proba(pd.DataFrame([row[FEATURES]]))[:, 1]
            probabilities[target] = float(calibrator.transform(raw)[0])
            manifests[target] = manifest
        validate_probability_order(probabilities["target_05"], probabilities["target_10"], probabilities["target_20"])
    except (PredictionStopped, FileNotFoundError, ValueError, KeyError) as error:
        args.output.write_text(json.dumps({"status":"STOPPED", "reason":str(error)}), encoding="utf-8")
        raise SystemExit(2) from error
    output = {"status":"OK", "as_of":str(row["datetime_utc"]), "prob_05":probabilities["target_05"], "prob_10":probabilities["target_10"], "prob_20":probabilities["target_20"], "regime":regime(probabilities["target_05"]), "model_version":manifests["target_05"]["version"]}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
