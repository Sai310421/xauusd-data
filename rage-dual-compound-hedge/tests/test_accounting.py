import unittest

from rage_dch.compound_accounting import (
    geometric_growth,
    max_floating_drawdown,
    months_below,
    period_return,
    simultaneous_loss_rate,
)
from rage_dch.hedge_engine import decide_hedge
from rage_dch.insurance_pool import insurance_payoff, total_pnl
from rage_dch.main_engine import NullMainEngine, Quote
from rage_dch.portfolio_state import Book, Portfolio
from rage_dch.risk_governor_adapter import govern


class AccountingTests(unittest.TestCase):
    def test_fees_and_conservation_with_pool(self):
        main = Book("main", 100_000)
        hedge = Book("hedge", 50_000)
        book = Portfolio(main, hedge)
        start = book.total_equity(2400, 2400.4)
        main.fill("buy", 10, 2400.4, 2)
        hedge.fill("sell", 4, 2400.0, 1)
        book.fund_pool(hedge, 100, "premium")
        shortfall = book.pay_from_pool(main, 40, "claim")
        self.assertEqual(shortfall, 0.0)
        end = book.total_equity(2400, 2400.4)
        spread = 10 * 0.4 + 4 * 0.4
        self.assertAlmostEqual(end, start - main.fees - hedge.fees - spread)

    def test_short_mark_uses_ask(self):
        hedge = Book("hedge", 10_000)
        hedge.fill("sell", 1, 2400, 0)
        self.assertAlmostEqual(hedge.equity(2400, 2410), 9990)
        marked_on_bid = hedge.cash + hedge.position * 2390
        self.assertNotAlmostEqual(hedge.equity(2390, 2400), marked_on_bid)

    def test_geometric_is_not_arithmetic(self):
        self.assertAlmostEqual(geometric_growth([0.5, -0.5]), (math_sqrt := __import__("math").sqrt(1.5 * 0.5)) - 1)
        self.assertLess(geometric_growth([0.5, -0.5]), 0.0)
        self.assertEqual(months_below([0.11, 0.09, 0.10], 0.10), 1)
        self.assertAlmostEqual(period_return(100, 110), 0.10)

    def test_drawdown_and_simultaneous(self):
        self.assertAlmostEqual(max_floating_drawdown([100, 120, 90, 130]), (120 - 90) / 120)
        self.assertAlmostEqual(simultaneous_loss_rate([-1, 2, -3], [-2, -1, 4]), 1 / 3)

    def test_insurance_identity(self):
        payoff = insurance_payoff(-50, coverage=30, deductible=10)
        self.assertAlmostEqual(payoff, 30)
        self.assertAlmostEqual(total_pnl(-50, 5, payoff, 4), -50 + 5 + 30 - 4)

    def test_null_main_and_flat_hedge_policy(self):
        quote = Quote(1, 2400, 2400.4)
        self.assertEqual(NullMainEngine().desired(quote), [])
        decision = decide_hedge(
            quote, 0.0, 2400, 0.2, 0.16, 0.1, 0.0, 2400, 0.0002, 0.0
        )
        self.assertEqual(decision.target_oz, 0.0)
        self.assertEqual(decision.orders, ())

    def test_hedge_opposes_a_long(self):
        quote = Quote(1, 2280, 2280.4)
        decision = decide_hedge(
            quote, 10.0, 2400, 0.22, 0.16, 0.1, 0.0, 2400, 0.0, 0.0
        )
        self.assertLess(decision.target_oz, 0.0)
        self.assertGreater(decision.intensity, 0.0)
        self.assertLess(abs(decision.target_oz), 10.0)

    def test_governor_halt(self):
        self.assertEqual(govern(3, 0, 0.01, 0.2, 5).status, "halt-no-equity")
        self.assertEqual(govern(3, 100, 0.2, 0.2, 5).quantity, 0.0)
        self.assertEqual(govern(9, 100, 0.0, 0.2, 5).status, "capped")


if __name__ == "__main__":
    unittest.main()
