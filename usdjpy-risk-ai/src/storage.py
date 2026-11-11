"""Immutable OCI Object Storage contract for model and report artifacts.

No caller may use local storage as a production fallback.  Candidate model
artifacts are content-addressed and verified after retrieval; promotion is a
separate quality-gated operation.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


class StorageError(RuntimeError):
    """Base error for an object-storage operation."""


class StorageCredentialsError(StorageError):
    """Raised when OCI production configuration is not available."""


class ObjectIntegrityError(StorageError):
    """Raised when an object or manifest fails its SHA-256 contract."""


REQUIRED_MANIFEST_FIELDS = {
    "version",
    "created_at",
    "target",
    "feature_schema",
    "feature_order",
    "model_uri",
    "calibrator_uri",
    "model_sha256",
    "calibrator_sha256",
    "train_start",
    "train_end",
    "metrics",
    "psi_reference_uri",
    "data_hash",
    "config_version",
}


class ObjectStore(Protocol):
    """Minimal immutable object contract shared by OCI and offline tests."""

    def put_immutable(self, name: str, payload: bytes) -> str:
        """Create one object and return its sha256 without overwriting."""

    def get_verified(self, name: str, expected_sha256: str) -> bytes:
        """Read exactly one object and validate its content hash."""


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of payload."""
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject incomplete or internally inconsistent manifests."""
    missing = REQUIRED_MANIFEST_FIELDS.difference(manifest)
    if missing:
        raise ObjectIntegrityError(f"MANIFEST_INVALID missing: {sorted(missing)}")
    if manifest["feature_schema"] != manifest["feature_order"]:
        raise ObjectIntegrityError("MANIFEST_INVALID feature schema/order mismatch")
    for field in ("model_sha256", "calibrator_sha256", "data_hash"):
        value = str(manifest[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ObjectIntegrityError(f"MANIFEST_INVALID malformed {field}")
    for field in ("model_uri", "calibrator_uri", "psi_reference_uri"):
        if not str(manifest[field]).startswith("oci://"):
            raise ObjectIntegrityError(f"MANIFEST_INVALID non-OCI {field}")


def _safe_name(name: str) -> str:
    """Reject absolute paths and traversal before constructing an OCI key."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or name == "":
        raise StorageError("unsafe object name")
    return str(path)


@dataclass
class LocalObjectStore:
    """In-memory offline test store; it is never constructed by production code."""

    objects: dict[str, bytes]

    def put_immutable(self, name: str, payload: bytes) -> str:
        """Create without overwrite to model OCI conditional PUT semantics."""
        key = _safe_name(name)
        if key in self.objects:
            raise StorageError("OBJECT_ALREADY_EXISTS")
        self.objects[key] = payload
        return sha256_bytes(payload)

    def get_verified(self, name: str, expected_sha256: str) -> bytes:
        """Read and validate test data exactly as production downloads do."""
        payload = self.objects[_safe_name(name)]
        if sha256_bytes(payload) != expected_sha256:
            raise ObjectIntegrityError("OBJECT_HASH_MISMATCH")
        return payload


@dataclass
class OciObjectStore:
    """OCI Object Storage adapter using conditional immutable writes."""

    client: Any
    namespace: str
    bucket: str
    prefix: str = "usdjpy-risk-ai"

    @classmethod
    def from_environment(cls, config: Mapping[str, Any]) -> OciObjectStore:
        """Create OCI client from explicit config; absence is a hard stop."""
        oci_config = config["oci"]
        bucket_env = str(oci_config.get("bucket_env", "OCI_OBJECT_STORAGE_BUCKET"))
        bucket = os.environ.get(bucket_env)
        if not bucket:
            raise StorageCredentialsError(f"missing required production secret: {bucket_env}")
        try:
            import oci
        except ImportError as exc:
            raise StorageCredentialsError("oci package is required in production") from exc
        config_file = os.environ.get("OCI_CONFIG_FILE", str(oci_config.get("config_file", "~/.oci/config")))
        profile = os.environ.get("OCI_CONFIG_PROFILE", str(oci_config.get("profile", "DEFAULT")))
        try:
            sdk_config = oci.config.from_file(config_file, profile)
            oci.config.validate_config(sdk_config)
            client = oci.object_storage.ObjectStorageClient(sdk_config)
            namespace = os.environ.get(str(oci_config.get("namespace_env", "OCI_OBJECT_STORAGE_NAMESPACE")))
            namespace = namespace or client.get_namespace().data
        except Exception as exc:  # SDK exceptions differ across auth modes.
            raise StorageCredentialsError("OCI_OBJECT_STORAGE_UNAVAILABLE") from exc
        return cls(client=client, namespace=namespace, bucket=bucket, prefix=str(oci_config.get("prefix", "usdjpy-risk-ai")))

    def _object_name(self, name: str) -> str:
        """Build an isolated key below the configured project prefix."""
        return _safe_name(str(PurePosixPath(self.prefix) / _safe_name(name)))

    def uri(self, name: str) -> str:
        """Return the canonical OCI URI recorded inside a manifest."""
        return f"oci://{self.bucket}/{self._object_name(name)}"

    def put_immutable(self, name: str, payload: bytes) -> str:
        """Write once using OCI's conditional-create header."""
        key = self._object_name(name)
        digest = sha256_bytes(payload)
        try:
            self.client.put_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                object_name=key,
                put_object_body=payload,
                content_length=len(payload),
                content_type="application/octet-stream",
                if_none_match="*",
                opc_meta={"sha256": digest},
            )
        except Exception as exc:
            raise StorageError("OCI_PUT_FAILED") from exc
        return digest

    def get_verified(self, name: str, expected_sha256: str) -> bytes:
        """Download and verify content rather than trusting object metadata."""
        try:
            response = self.client.get_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                object_name=self._object_name(name),
            )
            payload = response.data.content
        except Exception as exc:
            raise StorageError("OCI_GET_FAILED") from exc
        if sha256_bytes(payload) != expected_sha256:
            raise ObjectIntegrityError("OBJECT_HASH_MISMATCH")
        return payload


def publish_candidate_bundle(
    store: ObjectStore,
    model_path: Path,
    calibrator_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Store candidate assets immutably; this function cannot promote production."""
    candidate = dict(manifest)
    version = str(candidate["version"])
    target = str(candidate["target"])
    model_name = f"models/{version}/{target}.model.pkl"
    calibrator_name = f"models/{version}/{target}.calibrator.pkl"
    model_payload = model_path.read_bytes()
    calibrator_payload = calibrator_path.read_bytes()
    candidate["model_sha256"] = sha256_bytes(model_payload)
    candidate["calibrator_sha256"] = sha256_bytes(calibrator_payload)
    if isinstance(store, OciObjectStore):
        candidate["model_uri"] = store.uri(model_name)
        candidate["calibrator_uri"] = store.uri(calibrator_name)
    validate_manifest(candidate)
    store.put_immutable(model_name, model_payload)
    store.put_immutable(calibrator_name, calibrator_payload)
    manifest_name = f"manifests/candidate-{version}-{target}.json"
    manifest_payload = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    store.put_immutable(manifest_name, manifest_payload)
    return candidate
