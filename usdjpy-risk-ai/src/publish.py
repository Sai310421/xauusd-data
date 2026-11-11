"""Publish one candidate model bundle to OCI without changing production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .storage import OciObjectStore, publish_candidate_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "CANDIDATE_PENDING_OCI_UPLOAD":
        parser.error("only CANDIDATE_PENDING_OCI_UPLOAD manifests may be published")
    with args.config.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    target = str(manifest["target"])
    store = OciObjectStore.from_environment(config)
    published = publish_candidate_bundle(store, args.models / f"{target}.model.pkl", args.models / f"{target}.calibrator.pkl", manifest)
    args.manifest.write_text(json.dumps(published, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
