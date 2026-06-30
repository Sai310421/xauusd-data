"""Fetch the latest available bars and append to existing CSV / parquet files.

This is a thin wrapper meant to run from a GitHub Action or locally.
It does not invent data: it only delegates to a configured upstream
(HuggingFace dataset or a user-supplied MT5 export).

Usage:
    python scripts/fetch_latest.py [--from YYYY-MM-DD] [--to YYYY-MM-DD]

Environment:
    HF_DATASET_ID   HuggingFace dataset id, e.g. "user/xauusd-extended"
    HF_TOKEN        token if the dataset is gated
"""
import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas pyarrow", file=sys.stderr)
    sys.exit(2)


def fetch_huggingface(start: date, end: date) -> "pd.DataFrame | None":
    dataset_id = os.getenv("HF_DATASET_ID")
    if not dataset_id:
        return None
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=dataset_id,
        filename=f"XAUUSD_M5_{start.isoformat()}_{end.isoformat()}.parquet",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
    )
    return pd.read_parquet(path)


def append_into(target: Path, df_new: "pd.DataFrame") -> int:
    if target.exists():
        df_old = pd.read_parquet(target) if target.suffix == ".parquet" else pd.read_csv(
            target, parse_dates=["datetime"]
        )
        df = pd.concat([df_old, df_new]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    else:
        df = df_new
    if target.suffix == ".parquet":
        df.to_parquet(target, index=False)
    else:
        df.to_csv(target, index=False)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=str(date.today() - timedelta(days=14)))
    ap.add_argument("--to", dest="end", default=str(date.today()))
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    df = fetch_huggingface(start, end)
    if df is None:
        print("HF_DATASET_ID not set — nothing to fetch.")
        return 0

    pq_target = Path("parquet/XAUUSD_M5_latest.parquet")
    csv_target = Path("csv/XAUUSD/XAUUSD_M5_latest.csv")
    pq_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)

    rows_pq = append_into(pq_target, df)
    rows_csv = append_into(csv_target, df)
    print(f"parquet rows: {rows_pq:,}")
    print(f"csv rows:     {rows_csv:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
