# HFT Boost / Nautilus Reference Paths

Saved: 2026-09-03

Purpose: persistent source-of-truth pointers for future HFT Boost / NOTE-HFT / Nautilus Raw BidAsk backtests, plus the QuantConnect comparison engine.

## Canonical repository

- `Sai310421/xauusd-data`

### Confirmed Nautilus workflows

- `.github/workflows/note-hft-nautilus-smoke.yml`
  - Workflow name: `NOTE-HFT Nautilus Public Smoke`
  - NautilusTrader: `1.230.0`

- `.github/workflows/minimumspike-reality-noise-bt-v1.yml`
  - Workflow name: `MinimumSpike NOTE-HFT Reality Noise BT v1`
  - NautilusTrader: `1.230.0`

### Confirmed QuantConnect workflows

- `.github/workflows/qc-cloud-xau-smoke.yml`
  - Workflow name: `QC Cloud XAUUSD Hour Smoke`
- `.github/workflows/qc-cloud-bt-hub.yml`
  - Workflow name: `QC Cloud BT Hub`
- Protocol: `docs/MULTI_CHAT_QC_BT_PROTOCOL_v1.md`
- Bridge: `docs/QUANTCONNECT_CLOUD_BRIDGE.md`
- Algorithm: `research/qc/xauusd_hour_cloud.py`

QC results are `QC_CLOUD_BT` only. Never promote them to `NAUTILUS_BT`.

### Actions navigation

`Sai310421/xauusd-data -> Actions -> NOTE-HFT Nautilus Public Smoke`
`Sai310421/xauusd-data -> Actions -> MinimumSpike NOTE-HFT Reality Noise BT v1`
`Sai310421/xauusd-data -> Actions -> QC Cloud XAUUSD Hour Smoke`
`Sai310421/xauusd-data -> Actions -> QC Cloud BT Hub`

## Duka feed repository

- `Sai310421/xauusd-duka-feed`

- `books/duka-sync.json`
- `books/xauusd_mtf.json`
- `scripts/duka_sync.py`
- `Actions`
- `agents/ahae/integrations/NAUTILUS_GITHUB_ACTIONS_BRIDGE.md`
- `agents/ahae/integrations/DUKA_NAUTILUS_DATA_CACHE.md`
- `agents/ahae/integrations/QUANTCONNECT_CLOUD_BRIDGE.md`

## HFT Boost integration policy

Canonical Nautilus direction:

`Duka Raw BidAsk -> Duka cache/sync -> Nautilus QuoteTick -> NOTE-HFT/HFT Base -> Reality Noise Gate -> ICT/Harmonic Boost -> KPI evidence`

QuantConnect comparison direction:

`QC CFD QuoteBar XAUUSD Hour -> QC Cloud BT Hub -> QC_KPI_JSON -> results/qc-bt/<experiment_id>/`

Do not use OHLC-resample fallback for final Nautilus evidence.
Do not treat QC Hour QuoteBars as Dukascopy ticks.

Normalize comparable reports to the shared `$1,000 / 21 business day` view while retaining original raw BT metrics and immutable run evidence.
