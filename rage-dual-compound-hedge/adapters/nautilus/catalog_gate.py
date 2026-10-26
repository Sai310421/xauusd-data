"""Raw bid/ask gate. OHLC bars are rejected. No ticks are synthesized here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuoteRow:
    ts: int
    bid: float
    ask: float


@dataclass(frozen=True)
class CatalogVerdict:
    status: str
    count: int
    reasons: tuple[str, ...]


def validate_quote_ticks(rows: list[QuoteRow], minimum: int = 1000) -> CatalogVerdict:
    reasons: list[str] = []
    if len(rows) < minimum:
        reasons.append(f"count {len(rows)} < {minimum}")
    last_ts = None
    for i, row in enumerate(rows):
        if row.bid <= 0 or row.ask <= 0:
            reasons.append(f"non-positive price at {i}")
            break
        if row.ask < row.bid:
            reasons.append(f"ask < bid at {i}")
            break
        if last_ts is not None and row.ts < last_ts:
            reasons.append(f"timestamp not monotonic at {i}")
            break
        last_ts = row.ts
    if reasons:
        return CatalogVerdict("FAIL", len(rows), tuple(reasons))
    return CatalogVerdict("RAW_BIDASK_CATALOG_OK", len(rows), ())


def reject_ohlc(bar: dict) -> CatalogVerdict:
    if "bid" not in bar or "ask" not in bar:
        return CatalogVerdict("FAIL", 0, ("ohlc proxy rejected",))
    return CatalogVerdict("FAIL", 0, ("single bar is not a catalog",))
