"""Content-addressed OCI persistence for ephemeral GitHub Actions runners."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .storage import ObjectStore, OciObjectStore, StorageError


class FeatureStoreError(RuntimeError):
    """Raised when a persisted state cannot be safely restored or published."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_name(category: str, payload: bytes) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"state/{category}/{stamp}-{_sha256(payload)}.json"


def _artifact_name(category: str, logical_name: str, payload: bytes) -> str:
    suffix = Path(logical_name).suffix
    return f"data/{category}/{_sha256(payload)}{suffix}"


def _manifest_hash(name: str) -> str:
    digest = Path(name).stem.rsplit("-", maxsplit=1)[-1]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise FeatureStoreError("FEATURE_STORE_MANIFEST_NAME_INVALID")
    return digest


def publish_directory(store: ObjectStore, category: str, directory: Path) -> str:
    """Persist one complete local directory as immutable content and a state snapshot."""
    if not directory.is_dir():
        raise FeatureStoreError("FEATURE_STORE_DIRECTORY_MISSING")
    artifacts: dict[str, dict[str, str]] = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        logical_name = path.relative_to(directory).as_posix()
        payload = path.read_bytes()
        name = _artifact_name(category, logical_name, payload)
        digest = _sha256(payload)
        try:
            stored = store.put_immutable(name, payload)
            if stored != digest:
                raise FeatureStoreError("FEATURE_STORE_HASH_MISMATCH")
        except StorageError as error:
            if "OBJECT_ALREADY_EXISTS" not in str(error):
                raise FeatureStoreError("FEATURE_STORE_PUT_FAILED") from error
        artifacts[logical_name] = {"name": name, "sha256": digest}
    document = json.dumps(
        {"category": category, "artifacts": artifacts},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    state_name = _snapshot_name(category, document)
    try:
        store.put_immutable(state_name, document)
    except StorageError as error:
        if "OBJECT_ALREADY_EXISTS" not in str(error):
            raise FeatureStoreError("FEATURE_STORE_STATE_PUT_FAILED") from error
    return state_name


def hydrate_latest(store: ObjectStore, category: str, directory: Path, allow_empty: bool = False) -> str | None:
    """Restore the latest verified state snapshot into a runner-local directory."""
    candidates = store.list_names(f"state/{category}/")
    if not candidates:
        if allow_empty:
            directory.mkdir(parents=True, exist_ok=True)
            return None
        raise FeatureStoreError("FEATURE_STORE_STATE_MISSING")
    state_name = max(candidates)
    try:
        document = store.get_verified(state_name, _manifest_hash(state_name))
    except StorageError as error:
        raise FeatureStoreError("FEATURE_STORE_STATE_INVALID") from error
    try:
        state: dict[str, Any] = json.loads(document)
    except json.JSONDecodeError as error:
        raise FeatureStoreError("FEATURE_STORE_STATE_INVALID") from error
    if state.get("category") != category or not isinstance(state.get("artifacts"), dict):
        raise FeatureStoreError("FEATURE_STORE_STATE_INVALID")
    directory.mkdir(parents=True, exist_ok=True)
    for logical_name, reference in state["artifacts"].items():
        if not isinstance(reference, dict) or set(reference) != {"name", "sha256"}:
            raise FeatureStoreError("FEATURE_STORE_STATE_INVALID")
        destination = directory / logical_name
        if (
            destination.resolve().parent != directory.resolve()
            and directory.resolve() not in destination.resolve().parents
        ):
            raise FeatureStoreError("FEATURE_STORE_PATH_INVALID")
        try:
            payload = store.get_verified(str(reference["name"]), str(reference["sha256"]))
        except StorageError as error:
            raise FeatureStoreError("FEATURE_STORE_ARTIFACT_INVALID") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return state_name


def main() -> None:
    """Publish or hydrate a category using only OCI Object Storage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("publish", "hydrate"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    store = OciObjectStore.from_environment(config)
    if args.command == "publish":
        print(publish_directory(store, args.category, args.directory))
    else:
        state = hydrate_latest(store, args.category, args.directory, args.allow_empty)
        print(state or "EMPTY")


if __name__ == "__main__":
    main()
