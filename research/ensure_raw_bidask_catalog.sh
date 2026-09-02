#!/usr/bin/env bash
set -euo pipefail

CATALOG=${RAW_CATALOG:-catalog/raw_bidask}
START=${RAW_START:-2026-07-27}
DAYS=${RAW_DAYS:-30}
WORKERS=${RAW_WORKERS:-12}

mkdir -p "$CATALOG"

if python research/raw_bidask_catalog_guard.py --catalog "$CATALOG"; then
  echo 'RAW_CATALOG_READY=cache_or_existing'
  exit 0
fi

echo 'RAW_CATALOG_BOOTSTRAP_START'
ok=0
for attempt in 1 2 3; do
  echo "RAW_ENSURE_ATTEMPT=$attempt"
  extra=()
  if [ "$attempt" -gt 1 ]; then
    extra+=(--fresh)
  fi
  if python research/build_raw_bidask_catalog_duka.py \
    --start "$START" \
    --days "$DAYS" \
    --catalog "$CATALOG" \
    --workers "$WORKERS" \
    "${extra[@]}"; then
    if python research/raw_bidask_catalog_guard.py --catalog "$CATALOG"; then
      ok=1
      break
    fi
  fi
  sleep $((attempt * 15))
done

if [ "$ok" -ne 1 ]; then
  echo 'RAW_CATALOG_BOOTSTRAP_FAILED'
  exit 21
fi

echo 'RAW_CATALOG_READY=bootstrapped'
