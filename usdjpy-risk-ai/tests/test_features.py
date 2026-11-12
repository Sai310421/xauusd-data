"""Causal feature and label tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_provider import normalize_raw
from src.features import (
    FEATURES,
    FeatureQualityError,
    build_features,
    prepare_training_rows,
    validate_feature_integrity,
)
from tests.fixtures.make_fixture import make_fixture


def _features() -> pd.DataFrame:
    raw = normalize_raw(make_fixture())
    return build_features(raw)


def test_training_schema_is_fixed_and_causal() -> None:
    frame = _features()
    prepared = prepare_training_rows(frame)
    assert list(prepared[FEATURES].columns) == FEATURES
    assert (prepared["max_feature_source_timestamp"] <= prepared["datetime_utc"]).all()
    assert len(prepared) > 0


def test_future_price_change_does_not_alter_prior_feature_rows() -> None:
    raw = normalize_raw(make_fixture())
    baseline = build_features(raw)
    changed = raw.copy()
    late = changed["datetime_utc"].max()
    changed.loc[
        (changed["datetime_utc"] == late) & (changed["symbol"] == "USDJPY"),
        "close",
    ] *= 1.5
    altered = build_features(changed)
    cutoff = late - pd.Timedelta(hours=25)
    columns = FEATURES + ["risk_score", "future_return"]
    left = baseline.loc[
        baseline["datetime_utc"] < cutoff, columns
    ].reset_index(drop=True)
    right = altered.loc[
        altered["datetime_utc"] < cutoff, columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_futures_roll_is_excluded_from_features_not_oanda_labels() -> None:
    raw = normalize_raw(make_fixture())
    event_time = raw["datetime_utc"].iloc[-40]
    raw.loc[
        (raw["symbol"] == "ES") & (raw["datetime_utc"] == event_time),
        "roll_event",
    ] = True
    frame = build_features(raw)
    event_row = frame.loc[frame["datetime_utc"] == event_time].iloc[0]
    assert pd.isna(event_row["r_es"])
    assert pd.isna(event_row["flow_vx"])
    assert event_row[["target_05", "target_10", "target_20"]].notna().all()


def test_future_feature_timestamp_is_rejected() -> None:
    frame = _features()
    frame.loc[500, "max_feature_source_timestamp"] = (
        frame.loc[500, "datetime_utc"] + pd.Timedelta(hours=1)
    )
    with pytest.raises(FeatureQualityError, match="FEATURE_INTEGRITY_FAILURE"):
        validate_feature_integrity(frame)