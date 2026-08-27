# Multi-Chat Nautilus BT Protocol v1

Purpose: make every ChatGPT/Grok/Claude/Gemini/Codex chat able to identify, run-request, compare, and audit the same NautilusTrader BT without relying on chat memory.

## Canonical repository
- repo: Sai310421/xauusd-data
- NautilusTrader: 1.230.0
- canonical workflow: `.github/workflows/ae-nautilus-bt-hub.yml`
- results root: `results/ae-bt/<experiment_id>/`

## Non-negotiable evidence rule
Never call a result `NAUTILUS_VERIFIED` unless all of these exist:
1. GitHub Actions run_id
2. commit SHA
3. workflow name
4. NautilusTrader version
5. dataset id/hash
6. strategy/config hash
7. result artifact
8. execution model metadata

Otherwise label it exactly one of:
- PROXY_BT
- NAUTILUS_SMOKE
- NAUTILUS_BT
- BROKER_REALITY_BT

`Proxy BT != Nautilus BT != Broker Reality BT`.

## Experiment contract
Each experiment uses a unique `experiment_id`, recommended format:
`YYYYMMDD-HHMM-<edge>-<agent>-<shortid>`.

Request JSON schema:
```json
{
  "experiment_id": "20260828-1200-residual-wasserstein-chatgpt-a1",
  "edge": "baseline",
  "dataset": "public",
  "seed": 42,
  "notes": "independent validation"
}
```

Allowed initial edge values:
- baseline
- residual_wasserstein
- ou_qvi
- sampled_data

## Multi-chat concurrency
Different chats MUST use different experiment_id values. They may run concurrently because each run writes only to its own result directory/artifact. Do not overwrite a shared `latest.csv` as primary evidence.

## Required output manifest
Every runner should emit `manifest.json` containing at minimum:
```json
{
  "experiment_id": "...",
  "verification_level": "NAUTILUS_BT",
  "git_sha": "...",
  "nautilus_version": "1.230.0",
  "dataset_id": "...",
  "dataset_sha256": "...",
  "strategy_sha256": "...",
  "config_sha256": "...",
  "seed": 42,
  "started_at_utc": "...",
  "finished_at_utc": "..."
}
```

## Comparison gate
Compare BASE vs EDGE only when dataset hash, execution assumptions and baseline strategy hash match. Report at least N, WR, PF, MaxDD, RF, Net/Return, Monthly21, costs/slippage where available, and recovery/tail metrics for recovery experiments.

## Promotion states
`DISCOVERED -> FORMALIZED -> PROXY_PASS -> NAUTILUS_PASS -> RAW_BIDASK_PASS -> BROKER_REALITY_PASS -> VERIFIED`

A failure at a higher gate does not inherit a lower-gate PASS.

## Prompt for any new chat
Use this repository as the source of truth. Read `docs/MULTI_CHAT_NAUTILUS_BT_PROTOCOL_v1.md`, `.github/workflows/ae-nautilus-bt-hub.yml`, and the experiment manifest before reporting results. Never describe local Python/proxy calculations as Nautilus BT. Return run_id + SHA + artifact/manifest evidence with every claimed Nautilus result.
