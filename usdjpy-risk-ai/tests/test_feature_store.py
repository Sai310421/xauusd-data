"""Immutable state-snapshot tests for the OCI Feature Store contract."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_store import hydrate_latest, publish_directory
from src.storage import LocalObjectStore


def test_publish_then_hydrate_restores_all_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw_2026_01.parquet").write_bytes(b"first")
    nested = source / "nested"
    nested.mkdir()
    (nested / "manifest.json").write_text("{}", encoding="utf-8")
    store = LocalObjectStore({})
    state = publish_directory(store, "raw", source)
    restored = tmp_path / "restored"
    assert hydrate_latest(store, "raw", restored) == state
    assert (restored / "raw_2026_01.parquet").read_bytes() == b"first"
    assert (restored / "nested" / "manifest.json").read_text(encoding="utf-8") == "{}"


def test_second_snapshot_replaces_logical_file_and_retains_state_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "raw_2026_01.parquet"
    path.write_bytes(b"first")
    store = LocalObjectStore({})
    publish_directory(store, "raw", source)
    path.write_bytes(b"second")
    publish_directory(store, "raw", source)
    restored = tmp_path / "restored"
    hydrate_latest(store, "raw", restored)
    assert (restored / "raw_2026_01.parquet").read_bytes() == b"second"
