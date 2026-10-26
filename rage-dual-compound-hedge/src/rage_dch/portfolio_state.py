"""Independent cash books. Main, hedge, and the insurance pool do not share a wallet."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Book:
    name: str
    cash: float
    position: float = 0.0
    fees: float = 0.0
    realized: float = 0.0
    transfers_in: float = 0.0
    transfers_out: float = 0.0

    def equity(self, bid: float, ask: float) -> float:
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError("need a positive bid/ask with ask >= bid")
        mtm = self.position * bid if self.position >= 0 else self.position * ask
        return self.cash + mtm

    def fill(self, side: str, qty: float, price: float, fee: float) -> None:
        if qty < 0 or fee < 0 or price <= 0:
            raise ValueError("qty, fee must be non-negative and price positive")
        if side == "buy":
            self.cash -= price * qty + fee
            self.position += qty
        elif side == "sell":
            self.cash += price * qty - fee
            self.position -= qty
            self.realized += price * qty
        else:
            raise ValueError("side must be buy or sell")
        self.fees += fee


@dataclass
class Portfolio:
    main: Book
    hedge: Book
    pool: float = 0.0
    ledger: list[str] = field(default_factory=list)

    def total_equity(self, bid: float, ask: float) -> float:
        return self.main.equity(bid, ask) + self.hedge.equity(bid, ask) + self.pool

    def transfer(self, src: Book, dst: Book, amount: float, reason: str) -> None:
        if amount < 0:
            raise ValueError("transfer amount must be non-negative")
        if src.cash + 1e-9 < amount:
            raise ValueError(f"{src.name} cash {src.cash} cannot fund {amount}")
        src.cash -= amount
        src.transfers_out += amount
        dst.cash += amount
        dst.transfers_in += amount
        self.ledger.append(f"{src.name}->{dst.name} {amount:.6f} {reason}")

    def fund_pool(self, src: Book, amount: float, reason: str) -> None:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        if src.cash + 1e-9 < amount:
            raise ValueError("insufficient cash for pool funding")
        src.cash -= amount
        src.transfers_out += amount
        self.pool += amount
        self.ledger.append(f"{src.name}->pool {amount:.6f} {reason}")

    def pay_from_pool(self, dst: Book, amount: float, reason: str) -> float:
        """Pay what the pool can. Returns the unpaid shortfall. Does not invent cash."""
        if amount < 0:
            raise ValueError("amount must be non-negative")
        paid = min(self.pool, amount)
        self.pool -= paid
        dst.cash += paid
        dst.transfers_in += paid
        shortfall = amount - paid
        self.ledger.append(f"pool->{dst.name} paid {paid:.6f} shortfall {shortfall:.6f} {reason}")
        return shortfall
