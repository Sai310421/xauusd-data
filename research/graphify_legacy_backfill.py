from __future__ import annotations

"""Build a conservative legacy-asset registry for AE Graphify.

Rules:
- Discover only repository-tracked code/docs/workflows and committed evidence references.
- Never promote historical chat numbers or filenames to VERIFIED by inference.
- Evidence is classified by what is physically present in the repository.
- Output is deterministic and suitable for Graphify registry ingestion.
"""

import csv
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "graphify" / "legacy"

KEYWORDS = {
    "g75": ["g75", "tsugi"],
    "goririn": ["goririn", "fusion"],
    "minimumspike": ["minimumspike", "minimum spike"],
    "mvc": ["math_edge_mvc", "multi-variable conformal", "multivariable conformal"],
    "vgrsi": ["vgrsi"],
    "fib_harmonic": ["fib", "fibonacci", "harmonic"],
    "ict_smc": ["ict", "smc", "fvg", "ifvg", "bpr"],
    "recovery": ["recovery", "economic be", "hedge lock", "debt"],
    "hft": ["hft", "note-hft", "note_hft"],
    "crystal_x": ["crystal-x", "crystal_x", "crystal"],
    "amos": ["amos"],
}

TRACKED_PREFIXES = ("research/", "docs/", ".github/workflows/", "agents/", "strategies/")
EVIDENCE_PATTERNS = (
    re.compile(r"results/ae-bt/", re.I),
    re.compile(r"reality_evidence\.json", re.I),
    re.compile(r"catalog_manifest\.json", re.I),
    re.compile(r"catalog_sha", re.I),
    re.compile(r"raw[_ -]?bidask", re.I),
    re.compile(r"nautilus", re.I),
)


@dataclass
class LegacyAsset:
    asset_id: str
    name: str
    aliases: str
    status: str
    evidence_level: str
    source_files: str
    evidence_files: str
    workflow_files: str
    code_files: str
    docs_files: str
    sha256: str
    notes: str


def git_files() -> list[str]:
    cp = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [x.strip() for x in cp.stdout.splitlines() if x.strip()]


def read_text(rel: str) -> str:
    p = ROOT / rel
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def matching_files(files: Iterable[str], terms: list[str]) -> list[str]:
    out: list[str] = []
    for rel in files:
        if not rel.startswith(TRACKED_PREFIXES):
            continue
        hay = (rel + "\n" + read_text(rel)).lower()
        if any(term.lower() in hay for term in terms):
            out.append(rel)
    return sorted(set(out))


def evidence_level(matches: list[str]) -> tuple[str, list[str]]:
    evidence: list[str] = []
    raw = False
    nautilus = False
    run_evidence = False
    for rel in matches:
        txt = read_text(rel)
        if any(p.search(txt) for p in EVIDENCE_PATTERNS):
            evidence.append(rel)
        if re.search(r"raw[_ -]?bidask|quote ?ticks|catalog_manifest", txt, re.I):
            raw = True
        if re.search(r"nautilus", txt, re.I):
            nautilus = True
        if re.search(r"reality_evidence\.json|results/ae-bt/", txt, re.I):
            run_evidence = True

    # Conservative: repository references are evidence that a lane existed,
    # not proof that a particular historical KPI is valid.
    if raw and nautilus and run_evidence:
        return "LEGACY_EVIDENCE_FOUND", sorted(set(evidence))
    if nautilus or run_evidence:
        return "LEGACY_PARTIAL_EVIDENCE", sorted(set(evidence))
    return "LEGACY_UNVERIFIED", sorted(set(evidence))


def digest(paths: list[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(paths):
        h.update(rel.encode())
        p = ROOT / rel
        if p.exists() and p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def build() -> list[LegacyAsset]:
    files = git_files()
    assets: list[LegacyAsset] = []
    for asset_id, terms in KEYWORDS.items():
        matches = matching_files(files, terms)
        if not matches:
            continue
        ev_level, evidence = evidence_level(matches)
        status = "legacy_evidence_found" if ev_level == "LEGACY_EVIDENCE_FOUND" else "legacy_unverified"
        workflows = [x for x in matches if x.startswith(".github/workflows/")]
        code = [x for x in matches if x.startswith(("research/", "agents/", "strategies/"))]
        docs = [x for x in matches if x.startswith("docs/")]
        assets.append(
            LegacyAsset(
                asset_id=f"legacy_{asset_id}",
                name=asset_id.replace("_", " ").upper(),
                aliases=";".join(terms),
                status=status,
                evidence_level=ev_level,
                source_files=";".join(matches),
                evidence_files=";".join(evidence),
                workflow_files=";".join(workflows),
                code_files=";".join(code),
                docs_files=";".join(docs),
                sha256=digest(matches),
                notes="Backfilled from tracked repository sources only; historical KPI claims require explicit run/artifact verification.",
            )
        )
    return assets


def write_outputs(assets: list[LegacyAsset]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "legacy_assets.csv"
    fields = list(LegacyAsset.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in assets:
            w.writerow(asdict(a))

    payload = {
        "schema": "ae_graphify_legacy_backfill_v1",
        "policy": {
            "no_chat_number_promotion": True,
            "raw_bidask_required_for_raw_pass": True,
            "run_or_artifact_required_for_kpi_verification": True,
        },
        "assets": [asdict(a) for a in assets],
    }
    (OUT_DIR / "legacy_assets.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# AE Graphify Legacy Backfill",
        "",
        "過去資産を捨てずにGraphifyへ取り込むための保守的Backfillです。",
        "会話上のKPIや記憶値は証拠へ昇格させず、GitHubに物理的に残るコード・文書・workflow・evidence参照だけで分類します。",
        "",
        "| Asset | Status | Evidence | Code | Workflows | Docs |",
        "|---|---|---|---:|---:|---:|",
    ]
    for a in assets:
        lines.append(
            f"| {a.name} | {a.status} | {a.evidence_level} | {len(a.code_files.split(';')) if a.code_files else 0} | {len(a.workflow_files.split(';')) if a.workflow_files else 0} | {len(a.docs_files.split(';')) if a.docs_files else 0} |"
        )
    lines += [
        "",
        "## Promotion rule",
        "",
        "`LEGACY_UNVERIFIED -> LEGACY_EVIDENCE_FOUND -> NAUTILUS_PASS -> RAW_BIDASK_PASS -> VERIFIED`",
        "",
        "各昇格には実Run ID、commit SHA、strategy hash、catalog hash、artifact/KPI evidenceを要求します。",
    ]
    (OUT_DIR / "LEGACY_BACKFILL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = build()
    write_outputs(result)
    print(json.dumps({"assets": len(result), "out": str(OUT_DIR.relative_to(ROOT))}, ensure_ascii=False))
