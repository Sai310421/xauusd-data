# Graphify Legacy Backfill Policy

## Purpose

Graphify導入以前のAE資産を捨てずにKnowledge Graphへ統合する。ただし、過去チャット上の数値・記憶値・名称だけを根拠に検証済みへ昇格させない。

## Standard flow

```text
Legacy code/docs/workflows
        ↓
repository-backed discovery
        ↓
LEGACY_UNVERIFIED
        ↓ evidence exists
LEGACY_EVIDENCE_FOUND
        ↓ executable Nautilus evidence
NAUTILUS_PASS
        ↓ same Raw Bid/Ask catalog + hashes
RAW_BIDASK_PASS
        ↓ broker-reality validation
BROKER_REALITY_PASS
        ↓ full evidence chain
VERIFIED
```

## Evidence rules

### LEGACY_UNVERIFIED

名前、コード断片、仕様文書、過去workflowなどは存在するが、KPIを再現する十分なrun evidenceがない。

### LEGACY_EVIDENCE_FOUND

GitHub上にNautilus lane、Raw Bid/Ask参照、results/evidence pathなどの物理的証拠が存在する。ただし、その存在だけではHistorical KPI値をVerifiedとしない。

### NAUTILUS_PASS

Nautilus実行証拠があり、run_id / commit SHA / strategy hash / artifactが追跡可能。

### RAW_BIDASK_PASS

NAUTILUS_PASSに加え、Raw Bid/Ask Catalog hash、期間、銘柄、execution assumptionsが固定・追跡可能で、OHLC resample fallbackが使われていない。

### VERIFIED

必要な再現証拠を揃え、共有KPIとartifactの整合が確認されている。

## Required identity for promotion

可能な限り以下をGraphify edge / artifact metadataへ残す。

- repository
- run_id
- commit SHA
- strategy hash
- catalog hash
- symbols
- timeframe(s)
- date range
- execution assumptions
- initial equity display basis
- WR
- N
- PF
- RF
- MaxDD
- Monthly21
- Daily compound rate
- NetProfit
- artifact path/name

## Never infer

以下は禁止する。

1. 過去チャットで語られたKPIをartifact無しでVerifiedへ昇格する。
2. 同名戦略の別versionのKPIを流用する。
3. OHLC/resample結果をRaw Bid/Ask結果として扱う。
4. workflowが存在するだけでBT成功とみなす。
5. code pathが存在するだけでEDGEが実証済みとみなす。

## Backfill implementation

- Scanner: `research/graphify_legacy_backfill.py`
- Workflow: `.github/workflows/graphify-legacy-backfill.yml`
- Registry: `graphify/legacy/legacy_assets.csv`
- Generated evidence: `graphify/legacy/legacy_assets.json`
- Generated report: `graphify/legacy/LEGACY_BACKFILL_REPORT.md`

The scanner uses tracked repository sources only and is intentionally conservative.
