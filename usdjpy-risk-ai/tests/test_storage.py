"""Offline tests for immutable OCI artifact governance."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.storage import (
    LocalObjectStore,
    ObjectIntegrityError,
    StorageError,
    publish_candidate_bundle,
    validate_manifest,
)


def _manifest() -> dict[str, object]:
    return {
        "version": "candidate-v001",
        "created_at": "2026-11-11T00:00:00Z",
        "target": "target_05",
        "feature_schema": ["flow_vx"],
        "feature_order": ["flow_vx"],
        "model_uri": "oci://bucket/models/candidate-v001/target_05.model.pkl",
        "calibrator_uri": "oci://bucket/models/candidate-v001/target_05.calibrator.pkl",
        "model_sha256": "0" * 64,
        "calibrator_sha256": "0" * 64,
        "train_start": "2019-01-01T00:00:00Z",
        "train_end": "2024-01-01T00:00:00Z",
        "metrics": {},
        "psi_reference_uri": "oci://bucket/psi/reference.parquet",
        "data_hash": "1" * 64,
        "config_version": "v1.3.2",
    }


def test_manifest_rejects_schema_or_uri_tampering() -> None:
    manifest = _manifest()
    manifest["feature_order"] = ["tampered"]
    with pytest.raises(ObjectIntegrityError, match="schema/order"):
        validate_manifest(manifest)


def test_immutable_store_rejects_overwrite_and_hash_tampering() -> None:
    store = LocalObjectStore({})
    digest = store.put_immutable("models/a.pkl", b"model")
    assert digest == hashlib.sha256(b"model").hexdigest()
    with pytest.raises(StorageError, match="EXISTS"):
        store.put_immutable("models/a.pkl", b"replacement")
    with pytest.raises(ObjectIntegrityError, match="HASH"):
        store.get_verified("models/a.pkl", "0" * 64)


def test_candidate_bundle_is_immutable_and_not_production(tmp_path: Path) -> None:
    model = tmp_path / "model.pkl"
    calibrator = tmp_path / "calibrator.pkl"
    model.write_bytes(b"model")
    calibrator.write_bytes(b"calibrator")
    manifest = _manifest()
    store = LocalObjectStore({})
    published = publish_candidate_bundle(store, model, calibrator, manifest)
    assert published["model_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert "production" not in store.objects
    with pytest.raises(StorageError, match="EXISTS"):
        publish_candidate_bundle(store, model, calibrator, manifest)
