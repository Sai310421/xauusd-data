REQUIRED_TRADE_COLUMNS = [
    "entry_ts", "exit_ts", "side", "entry", "exit", "pnl"
]

OPTIONAL_TRADE_COLUMNS = [
    "symbol", "timeframe", "lot", "add_count", "gross_pnl", "spread",
    "commission", "slippage", "swap", "mfe", "mae", "holding_s",
    "regime", "source_id", "candidate_id"
]

# Times used for behavioral reconstruction should be Unix seconds or another
# consistent numeric clock. Monetary PnL must be net of known costs when the
# purpose is EDGE qualification.
