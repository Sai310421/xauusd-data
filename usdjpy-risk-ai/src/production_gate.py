"""Production readiness gate; a pass authorizes monitoring only, never trading."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from .integration import assert_integration_ready, integration_readiness


def main() -> None:
    """Emit a machine-readable inventory and exit nonzero when production is blocked."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--repository-root", type=Path, default=Path(".."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    files = {
        path.relative_to(args.repository_root).as_posix()
        for path in args.repository_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    environment = {name: os.environ.get(name, "") for name in (
        "DATABENTO_API_KEY", "OANDA_API_TOKEN", "OCI_OBJECT_STORAGE_BUCKET", "DISCORD_WEBHOOK_URL"
    )}
    items = integration_readiness(config, files, environment)
    payload = {"items": [item.__dict__ for item in items]}
    try:
        assert_integration_ready(items)
        payload["status"] = "READY_FOR_SHADOW"
    except RuntimeError as error:
        payload["status"] = "BLOCKED"
        payload["reason"] = str(error)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit(2) from error
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
