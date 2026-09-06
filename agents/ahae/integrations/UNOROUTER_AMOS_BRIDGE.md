# UnoRouter AMOS/ODS Bridge

Status: integrated (gateway client + route policy + GitHub Actions smoke test)

## Purpose

UnoRouter is an optional multi-LLM gateway for AMOS/ODS analysis workloads. It must not become a hard dependency of deterministic Nautilus/backtest execution or live trading logic.

## Endpoint

- OpenAI-compatible base URL: `https://api.unorouter.com/v1`
- Secret: `UNOROUTER_API_KEY`

## Files

- `agents/ahae/integrations/unorouter_client.py` — dependency-free OpenAI-compatible client.
- `agents/ahae/integrations/unorouter_routes.json` — role/model policy and guardrails.
- `.github/workflows/unorouter-smoke.yml` — manual connectivity test.

## GitHub setup required

Create repository Actions secret:

`UNOROUTER_API_KEY=<your UnoRouter token>`

Optional repository/environment variables can select model IDs:

- `UNOROUTER_WORKER_MODEL`
- `UNOROUTER_CODE_MODEL`
- `UNOROUTER_REASONING_MODEL`

The smoke workflow accepts a model ID manually and defaults to `glm-5.3-flash:free`.

## Architecture rule

```text
AMOS / ODS Supervisor
        |
        v
LLM Router Interface
  |       |        |
Direct  UnoRouter  Local Ollama
          |
          v
OpenAI-compatible providers/models
```

UnoRouter is advisory/shadow-only around BT:

```text
source/strategy -> LLM analysis -> patch/hypothesis candidate
                                  |
                                  v
                         deterministic validation
                                  |
                                  v
                         Raw Tick / Nautilus BT
                                  |
                                  v
                              KPI gate
```

Never allow an LLM response to rewrite measured BT results, bypass KPI gates, override a Frozen Core, or directly place a trade.

## Smoke test

GitHub Actions -> `UnoRouter AMOS Smoke` -> Run workflow.

Success output:

`UnoRouter smoke test OK (<model>)`

## Failure policy

- Missing key: fail closed.
- HTTP/API failure: raise `UnoRouterError`; do not silently fabricate a model response.
- Gateway outage: route orchestration should fall back to a direct provider or local Ollama where configured.
- Model removal: update the model route configuration; do not bind strategy logic to a specific external model ID.
