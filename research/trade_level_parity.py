from __future__ import annotations

import argparse
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

KEYS = ["symbol", "side", "entry_time", "exit_time"]
NUMERIC = ["entry_price", "exit_price", "qty", "pnl"]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"BLOCKED_MISSING_INPUT: {path}")
    df = pd.read_csv(path)
    missing = [c for c in KEYS + NUMERIC if c not in df.columns]
    if missing:
        raise SystemExit(f"BLOCKED_SCHEMA_MISMATCH {path}: {missing}")
    for c in ["entry_time", "exit_time"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["symbol", "side", "entry_time"], na_position="last").reset_index(drop=True)


def _ms(ts) -> float | None:
    if pd.isna(ts):
        return None
    return float(pd.Timestamp(ts).value) / 1_000_000.0


def _finite_numeric(row: pd.Series) -> bool:
    return all(pd.notna(row[c]) and pd.api.types.is_number(row[c]) for c in NUMERIC)


def _hopcroft_karp(adj: list[list[int]], right_count: int) -> list[int]:
    """Return right match for every left vertex, or -1.

    Candidate edges are pre-bounded by the timestamp tolerance window, so this avoids
    rescanning the full Nautilus export while finding a globally valid one-to-one assignment.
    """
    n = len(adj)
    pair_u = [-1] * n
    pair_v = [-1] * right_count
    dist = [0] * n

    def bfs() -> bool:
        q: deque[int] = deque()
        found = False
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adj[u]:
                mate = pair_v[v]
                if mate == -1:
                    found = True
                elif dist[mate] < 0:
                    dist[mate] = dist[u] + 1
                    q.append(mate)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            mate = pair_v[v]
            if mate == -1 or (dist[mate] == dist[u] + 1 and dfs(mate)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                dfs(u)
    return pair_u


def compare(mt5: pd.DataFrame, nt: pd.DataFrame, *, max_mismatch_pct: float, time_tol_ms: float, sample_limit: int = 50) -> dict:
    details: list[dict] = []
    mismatch_rows = 0

    def note(item: dict) -> None:
        nonlocal mismatch_rows
        mismatch_rows += 1
        if len(details) < sample_limit:
            details.append(item)

    groups_a: dict[tuple[str, str], list[tuple[int, pd.Series]]] = defaultdict(list)
    groups_b: dict[tuple[str, str], list[tuple[int, pd.Series]]] = defaultdict(list)

    for label, df, groups in (("mt5", mt5, groups_a), ("nautilus", nt, groups_b)):
        for idx, row in df.iterrows():
            key = (str(row.get("symbol", "")).strip(), str(row.get("side", "")).strip().upper())
            if not key[0] or not key[1] or _ms(row.get("entry_time")) is None or _ms(row.get("exit_time")) is None or not _finite_numeric(row):
                note({"reason": "invalid_required_field", "source": label, "index": int(idx)})
                continue
            groups[key].append((int(idx), row))

    for key in sorted(set(groups_a) | set(groups_b)):
        aa = groups_a.get(key, [])
        bb = groups_b.get(key, [])
        if not aa:
            for idx, _ in bb:
                note({"reason": "unmatched_nautilus", "index": idx})
            continue
        if not bb:
            for idx, _ in aa:
                note({"reason": "unmatched_mt5", "index": idx})
            continue

        b_entry = [_ms(row["entry_time"]) for _, row in bb]
        assert all(x is not None for x in b_entry)
        b_entry_f = [float(x) for x in b_entry]
        adj: list[list[int]] = []
        for _, a in aa:
            ae = float(_ms(a["entry_time"]))
            ax = float(_ms(a["exit_time"]))
            lo = bisect_left(b_entry_f, ae - time_tol_ms)
            hi = bisect_right(b_entry_f, ae + time_tol_ms)
            candidates = []
            for j in range(lo, hi):
                bx = float(_ms(bb[j][1]["exit_time"]))
                if abs(ax - bx) <= time_tol_ms:
                    candidates.append(j)
            adj.append(candidates)

        match = _hopcroft_karp(adj, len(bb))
        used_b: set[int] = set()
        for u, v in enumerate(match):
            ia, a = aa[u]
            if v == -1:
                note({"reason": "unmatched_mt5", "index": ia})
                continue
            used_b.add(v)
            ib, b = bb[v]
            row_diff = {}
            for c in NUMERIC:
                av, bv = float(a[c]), float(b[c])
                tol = max(1e-9, abs(av) * 1e-6)
                if abs(av - bv) > tol:
                    row_diff[c] = [av, bv]
            if row_diff:
                note({"mt5_index": ia, "nautilus_index": ib, "reason": "field_mismatch", "diff": row_diff})
        for j, (ib, _) in enumerate(bb):
            if j not in used_b:
                note({"reason": "unmatched_nautilus", "index": ib})

    n = max(len(mt5), len(nt), 1)
    mismatch_pct = 100.0 * mismatch_rows / n
    return {
        "status": "PASS" if mismatch_pct <= max_mismatch_pct else "FAIL",
        "mt5_trades": int(len(mt5)),
        "nautilus_trades": int(len(nt)),
        "count_delta": int(abs(len(mt5) - len(nt))),
        "mismatch_rows": int(mismatch_rows),
        "mismatch_pct": float(mismatch_pct),
        "threshold_pct": float(max_mismatch_pct),
        "time_tolerance_ms": float(time_tol_ms),
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--nautilus", required=True)
    ap.add_argument("--out", default="results/parity/trade_level_parity.json")
    ap.add_argument("--max-mismatch-pct", type=float, default=3.0)
    ap.add_argument("--time-tolerance-ms", type=float, default=1000.0)
    args = ap.parse_args()
    result = compare(load(Path(args.mt5)), load(Path(args.nautilus)), max_mismatch_pct=args.max_mismatch_pct, time_tol_ms=args.time_tolerance_ms)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
