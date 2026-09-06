"""Advisory post-backtest review hook for AMOS/ODS.

Reads deterministic BT evidence from a result directory, sends only a bounded
summary to UnoRouter, and writes an advisory JSON/Markdown review beside the
results. It never changes measured KPIs, trading logic, or pass/fail gates.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from unorouter_client import UnoRouterClient, UnoRouterError

MAX_FILE_BYTES = 200_000
MAX_TEXT_CHARS = 24_000
DEFAULT_MODELS = [
    "glm-5.3-flash:free",
    "codestral-latest:free",
    "glm-5.3-flash-thinking:free",
]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": f"could not parse {path.name}: {exc}"}


def _csv_preview(path: Path, rows: int = 12) -> dict[str, Any]:
    out: dict[str, Any] = {"file": path.name, "rows": []}
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            out["columns"] = reader.fieldnames or []
            for i, row in enumerate(reader):
                if i >= rows:
                    break
                out["rows"].append(row)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def collect_evidence(result_dir: Path) -> dict[str, Any]:
    if not result_dir.exists() or not result_dir.is_dir():
        raise FileNotFoundError(result_dir)

    evidence: dict[str, Any] = {
        "result_dir": str(result_dir),
        "json": {},
        "csv_preview": [],
        "files": [],
    }

    for p in sorted(result_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(result_dir))
        size = p.stat().st_size
        evidence["files"].append({"path": rel, "bytes": size})
        if size > MAX_FILE_BYTES:
            continue
        if p.suffix.lower() == ".json":
            evidence["json"][rel] = _read_json(p)
        elif p.suffix.lower() == ".csv" and len(evidence["csv_preview"]) < 6:
            preview = _csv_preview(p)
            preview["path"] = rel
            evidence["csv_preview"].append(preview)

    return evidence


def _models() -> list[str]:
    configured = os.getenv("UNOROUTER_REVIEW_MODELS", "").strip()
    if configured:
        return [x.strip() for x in configured.split(",") if x.strip()]
    model = os.getenv("UNOROUTER_REASONING_MODEL", "").strip()
    return ([model] if model else []) + [m for m in DEFAULT_MODELS if m != model]


def review(result_dir: Path) -> tuple[dict[str, Any], str]:
    evidence = collect_evidence(result_dir)
    evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2)[:MAX_TEXT_CHARS]

    system = (
        "You are the AMOS post-backtest advisory reviewer. The supplied evidence is measured output. "
        "Never alter, invent, normalize, or reinterpret measured values as if verified. "
        "Do not authorize live trading. Do not override Frozen Core, Raw BidAsk provenance, Nautilus gates, "
        "or deterministic pass/fail logic. Return compact valid JSON only with keys: "
        "verdict, measured_findings, risks, next_experiments, stop_conditions, confidence. "
        "Each next_experiments item must be a falsifiable test and must preserve deterministic BT execution."
    )
    user = "Review this deterministic BT evidence and propose the smallest high-information next tests:\n" + evidence_text

    client = UnoRouterClient()
    errors: list[str] = []
    for model in _models():
        try:
            text = client.text(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=900,
            )
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lstrip().startswith("json"):
                    cleaned = cleaned.lstrip()[4:].lstrip()
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("review response is not an object")
            parsed["review_model"] = model
            parsed["advisory_only"] = True
            parsed["measured_results_modified"] = False
            return parsed, text
        except (UnoRouterError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{model}: {exc}")

    raise RuntimeError("All UnoRouter review models failed: " + " | ".join(errors))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: post_bt_unorouter_review.py <result_dir>", file=sys.stderr)
        return 2

    result_dir = Path(sys.argv[1])
    review_json, _ = review(result_dir)
    out_json = result_dir / "unorouter_advisory_review.json"
    out_md = result_dir / "unorouter_advisory_review.md"
    out_json.write_text(json.dumps(review_json, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# UnoRouter Advisory Post-BT Review",
        "",
        "> Advisory only. Measured BT results, provenance, gates, and Frozen Core remain unchanged.",
        "",
        "```json",
        json.dumps(review_json, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
