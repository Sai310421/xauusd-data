"""Tests for normalized raw H1 contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_provider import DataQualityError, normalize_raw
from tests.fixtures.make_fixture import make_fixture


def test_normalize_uses_completed_bar_end() -> None:
    raw = make_fixture(periods=3)
    normalized = normalize_raw(raw)
    assert (normalized["datetime_utc"] - normalized["source_bar_start_utc"] == pd.Timedelta(hours=1)).all()
    assert normalized["checksum"].str.len().eq(64).all()


def test_duplicate_is_rejected() -> None:
    raw = make_fixture(periods=3)
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataQualityError, match="duplicate"):
        normalize_raw(duplicate)

def test_monthly_writer_merges_incremental_windows(tmp_path: Path) -> None:
    from src.data_provider import write_monthly_raw

    full = normalize_raw(make_fixture(periods=7))
    initial = full.loc[full["datetime_utc"] <= full["datetime_utc"].iloc[4]].copy()
    later = full.loc[full["datetime_utc"] >= full["datetime_utc"].iloc[3]].copy()
    write_monthly_raw(initial, tmp_path)
    paths = write_monthly_raw(later, tmp_path)
    stored = pd.read_parquet(paths[0])
    assert len(stored) == len(full)