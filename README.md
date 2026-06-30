# xauusd-data

Public OHLCV dataset for AI-driven backtesting. Any LLM with file-upload or repo-fetch can run the same backtest on the same bars.

- **Pairs**: XAUUSD, EURUSD, USDJPY, GBPUSD, AUDUSD, XAGUSD, EURGBP, AUDNZD
- **Resolutions**: M1, M5 (parquet) — XAUUSD also has H1 and D1 as CSV
- **Coverage (initial)**: 2026-02-25 → 2026-05-26 (90 days, last full quarter pair)
- **Roadmap**: monthly append via the included `fetch_latest.py`; quarterly multi-year backfill in Releases

> This is **mid-quote OHLCV** — no broker spread, no slippage, no swap. Use it for *strategy comparison* across AIs. For deployment numbers, run on a broker's strategy tester.

## How each tool consumes this

### Python (anywhere)
```python
import pandas as pd
df = pd.read_parquet(
    "https://raw.githubusercontent.com/Sai310421/xauusd-data/main/parquet/XAUUSD_M5_90d_20260526.parquet"
)
df.head()
```

### ChatGPT, Claude.ai, Gemini AI Studio
1. Open the file in this repo's `csv/` directory
2. Tap the **Raw** button, then **Download**
3. Upload the CSV in the chat
4. Paste the BT prompt below

### Codex / GitHub Actions
```yaml
- uses: actions/checkout@v4
  with:
    repository: Sai310421/xauusd-data
    path: data
- run: python your_bt.py data/csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv
```

### `git clone`
```bash
git clone https://github.com/Sai310421/xauusd-data
```

### ai-trader (Backtrader-based CLI, GPL-3.0)
[whchien/ai-trader](https://github.com/whchien/ai-trader) runs 20+ built-in strategies from a YAML config.
```bash
pip install ai-trader
git clone https://github.com/Sai310421/xauusd-data
cat > bt.yaml <<'EOF'
broker:   { cash: 100000, commission: 0.0001 }
data:     { file: "xauusd-data/csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv", start_date: "2026-02-25", end_date: "2026-05-26" }
strategy: { class: "CrossSMAStrategy", params: { fast: 10, slow: 30 } }
sizer:    { type: "percent", params: { percents: 95 } }
EOF
ai-trader run bt.yaml
```
Lists available strategies: `ai-trader list strategies`.

## Standard BT prompt (paste into any AI, identical input → comparable output)

```
This CSV is XAUUSD M5 bars. Run a backtest with the following strategy
and report WR / PF / MaxDD / Sharpe / total trades.

Strategy spec:
- entry: bullish FVG (3-candle fair value gap) detected, then BUY on first
  retracement of 50% into the gap
- killzone filter: London session 08:00–11:00 UTC only
- SL: entry - 1.5 × ATR(14)
- TP: 3R fixed (3 × distance from entry to SL)
- risk: $100 per trade, $10,000 starting account, fixed-fractional sizing

Output:
- WR (win rate)
- PF (profit factor)
- MaxDD (max drawdown, % of equity)
- Sharpe (annualized)
- total trades
- equity curve as a list of (timestamp, equity) tuples

Constraints:
- Treat bar as filled at close
- No spread / slippage modeling
- Document any data quality issue you hit (gaps, NaNs)
```

The strategy above is a generic ICT FVG demo; swap it for any strategy you want to compare across AIs.

## Validation reference

Run the included `validate.py` after fetching:

```bash
python scripts/validate.py csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv
```

Checks:
- monotonic timestamps
- `low <= open, close <= high`
- weekend gaps (Sat–Sun) only — no weekday gaps
- no NaNs

## Layout

```
.
├── csv/
│   └── XAUUSD/
│       ├── XAUUSD_M1_2026Q1Q2.csv   (~5.3 MB,  86k rows)
│       ├── XAUUSD_M5_2026Q1Q2.csv   (~1.1 MB,  17k rows)
│       ├── XAUUSD_H1_2026Q1Q2.csv   (~90 KB,   1.4k rows)
│       └── XAUUSD_D1_2026Q1Q2.csv   (~4  KB,    77 rows)
├── parquet/
│   ├── XAUUSD_M5_90d_20260526.parquet
│   ├── XAUUSD_M1_90d_20260526.parquet
│   ├── EURUSD_M5_90d_20260526.parquet
│   ├── EURUSD_M1_90d_20260526.parquet
│   ├── USDJPY_M5_90d_20260526.parquet
│   ├── USDJPY_M1_90d_20260526.parquet
│   ├── GBPUSD_M5_90d_20260526.parquet
│   ├── GBPUSD_M1_90d_20260526.parquet
│   ├── AUDUSD_M5_90d_20260526.parquet
│   ├── XAGUSD_M5_90d_20260526.parquet
│   ├── EURGBP_M5_90d_20260526.parquet
│   ├── EURGBP_M1_90d_20260526.parquet
│   ├── AUDNZD_M5_90d_20260526.parquet
│   └── AUDNZD_M1_90d_20260526.parquet
└── scripts/
    ├── validate.py
    └── fetch_latest.py
```

## CSV format

```
datetime,open,high,low,close,volume
2026-02-25 00:00:00,5146.39500,5149.40500,5139.48500,5139.98500,0.05823
```

- timezone: UTC
- datetime: ISO 8601
- OHLC: 5 decimal places (XAUUSD broker convention)
- volume: tick volume (normalized)

## Sources

- 2026 Q1–Q2: aggregated from public broker feeds (mid quote)
- For multi-year (2004→) data: see `scripts/fetch_latest.py` — pulls extension data from HuggingFace + the user's MT5 feed locally

## Updating

A monthly GitHub Action runs `scripts/fetch_latest.py` and appends new bars. Open a PR if you have an MT5 export newer than the last commit.

## License

MIT.
