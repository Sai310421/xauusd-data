"""Render an auditable fixed-layout HTML prediction report."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    """Render required fields without executing a trade or webhook request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(args.prediction.read_text(encoding="utf-8"))
    fields = ("as_of", "prob_05", "prob_10", "prob_20", "regime", "top_drivers", "psi", "model_version", "auc_roc", "pr_auc", "brier_score", "ece", "status")
    rows = "".join(f"<tr><th>{html.escape(field)}</th><td>{html.escape(str(record.get(field, 'N/A')))}</td></tr>" for field in fields)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"<!doctype html><html><body><h1>USDJPY Crash Model</h1><table>{rows}</table></body></html>", encoding="utf-8")


if __name__ == "__main__":
    main()
