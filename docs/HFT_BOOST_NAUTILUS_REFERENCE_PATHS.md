# HFT Boost / Nautilus Reference Paths

Saved: 2026-08-31

Purpose: persistent source-of-truth pointers for future HFT Boost / NOTE-HFT / Nautilus Raw BidAsk backtests.

## Canonical repository

- `Sai310421/xauusd-data`

### Confirmed workflows

- `.github/workflows/note-hft-nautilus-smoke.yml`
  - Workflow name: `NOTE-HFT Nautilus Public Smoke`
  - NautilusTrader: `1.230.0`
  - Purpose: public NOTE-HFT + XAUUSD dataset smoke validation.

- `.github/workflows/minimumspike-reality-noise-bt-v1.yml`
  - Workflow name: `MinimumSpike NOTE-HFT Reality Noise BT v1`
  - NautilusTrader: `1.230.0`
  - Purpose: Ideal / Normal / Stress / Tail reality-noise backtest.

### Actions navigation

`Sai310421/xauusd-data -> Actions -> NOTE-HFT Nautilus Public Smoke`

`Sai310421/xauusd-data -> Actions -> MinimumSpike NOTE-HFT Reality Noise BT v1`

## Duka feed repository

- `Sai310421/xauusd-duka-feed`

User-designated persistent references:

- `books/duka-sync.json`
- `books/xauusd_mtf.json`
- `scripts/duka_sync.py`
- `Actions`
- `agents/ahae/integrations/NAUTILUS_GITHUB_ACTIONS_BRIDGE.md`
- `agents/ahae/integrations/DUKA_NAUTILUS_DATA_CACHE.md`

Note: the two `agents/ahae/integrations/...` paths were user-provided as canonical references but were not resolvable on the repository default branch during the 2026-08-31 check. Keep them as intended bridge/cache references until their current branch/location is confirmed.

## HFT Boost integration policy

Future HFT Boost work should prefer the existing NOTE-HFT / Reality Noise / Duka cache infrastructure over creating isolated duplicate pipelines.

Canonical direction:

`Duka Raw BidAsk -> Duka cache/sync -> Nautilus QuoteTick -> NOTE-HFT/HFT Base -> Reality Noise Gate -> ICT/Harmonic Boost -> KPI evidence`

Do not use OHLC-resample fallback for final Nautilus evidence.

Normalize comparable reports to the shared `$1,000 / 21 business day` view while retaining original raw BT metrics and immutable run evidence.
