from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from core import feedback_queue, load_candidates, rank, score_candidate


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="AMOS Dual-Arena scorer")
    p.add_argument("--input", default="results", help="Directory to scan recursively for JSON result files")
    p.add_argument("--output", default="results/amos-dual-arena/latest")
    p.add_argument("--top-n-feedback", type=int, default=5)
    args = p.parse_args()

    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("*.json")) if root.exists() else []
    candidates = load_candidates(files)
    scores = [score_candidate(c) for c in candidates]
    leagues = rank(scores)
    feedback = feedback_queue(scores, args.top_n_feedback)

    (out / "leaderboards.json").write_text(json.dumps(leagues, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "feedback_queue.json").write_text(json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "candidates_ranked": len(scores),
                "feedback_items": len(feedback),
                "leagues": {k: len(v) for k, v in leagues.items()},
                "input_root": str(root),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for name, rows in leagues.items():
        _write_csv(out / f"{name}.csv", rows)

    print(json.dumps({"ranked": len(scores), "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
