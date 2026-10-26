import unittest

from adapters.monte_carlo.paired_block import independent_shuffle, paired_block_paths
from adapters.monte_carlo.structural import pearson, run_overlay, run_v1_intensity_overlay
from adapters.nautilus.catalog_gate import QuoteRow, reject_ohlc, validate_quote_ticks
from adapters.nautilus.fills import native_fill_count
from adapters.optuna.objective import grid_minimize
from adapters.wfo.splitter import walk_forward
from rage_dch.hedge_optimizer import Scenario, maximize_log_growth, min_protection_quantity, required_monthly_edge
from rage_dch.reporting import empty_market_report


class GateTests(unittest.TestCase):
    def test_stage_one_quantity_and_infeasible(self):
        scenarios = [
            Scenario("crash", main_pnl=-1000, hedge_pnl_per_unit=50, cost_per_unit=1),
            Scenario("calm", main_pnl=100, hedge_pnl_per_unit=-5, cost_per_unit=1),
        ]
        got = min_protection_quantity(scenarios, loss_bound=200, q_max=100, unit_cost_price=1)
        self.assertTrue(got.feasible)
        self.assertAlmostEqual(got.quantity, 800 / 49)
        blocked = min_protection_quantity(scenarios, loss_bound=200, q_max=10, unit_cost_price=1)
        self.assertFalse(blocked.feasible)

    def test_required_edge_rises_when_capital_is_locked(self):
        self.assertAlmostEqual(required_monthly_edge(0.0, 0.01, 0.10), 0.10)
        locked = required_monthly_edge(0.30, 0.01, 0.10)
        self.assertIsNotNone(locked)
        assert locked is not None
        self.assertAlmostEqual(locked, (1.10 - 0.30 * 0.99) / 0.70 - 1.0)
        self.assertGreater(locked, 0.10)
        self.assertIsNone(required_monthly_edge(1.0, 0.01, 0.10))

    def test_stage_two_does_not_invent_a_winner(self):
        result = maximize_log_growth({"flat-alpha": [0.0, 0.0, 0.0]}, 0.30, 0.01, 0.10)
        self.assertFalse(result.meets_target)
        self.assertLess(result.geometric_monthly, 0.10)

    def test_positive_sleeve_can_be_selected_without_being_called_market(self):
        result = maximize_log_growth({"given": [0.2, 0.2, 0.2]}, 0.0, 0.0, 0.10)
        self.assertTrue(result.meets_target)
        self.assertEqual(result.sleeve, "given")

    def test_wfo_has_no_overlap(self):
        folds = walk_forward(100, train=40, test=10, step=10, embargo=5)
        self.assertGreater(len(folds), 1)
        for fold in folds:
            self.assertTrue(set(fold.train).isdisjoint(fold.test))
            self.assertLess(max(fold.train), min(fold.test))

    def test_paired_blocks_keep_negative_correlation(self):
        main = [((i % 17) - 8) / 8 for i in range(180)]
        hedge = [-x for x in main]
        paths = paired_block_paths(main, hedge, block=6, n_paths=30, horizon=60, seed=3)
        corrs = [pearson(a, b) for a, b in paths]
        self.assertLess(sum(corrs) / len(corrs), -0.99)
        shuffled_main, shuffled_hedge = independent_shuffle(main, hedge, seed=3)
        self.assertGreater(pearson(shuffled_main, shuffled_hedge), -0.5)

    def test_catalog_gate(self):
        rows = [QuoteRow(i, 2400.0, 2400.2) for i in range(1000)]
        self.assertEqual(validate_quote_ticks(rows).status, "RAW_BIDASK_CATALOG_OK")
        broken = list(rows)
        broken[10] = QuoteRow(10, 2400.4, 2400.0)
        self.assertEqual(validate_quote_ticks(broken).status, "FAIL")
        rewound = list(rows)
        rewound[50] = QuoteRow(4, 2400.0, 2400.2)
        self.assertEqual(validate_quote_ticks(rewound).status, "FAIL")
        self.assertIn("ohlc", reject_ohlc({"open": 1, "high": 1, "low": 1, "close": 1}).reasons[0])
        self.assertEqual(native_fill_count(), 0)

    def test_grid_search_is_not_optuna(self):
        best = grid_minimize({"x": [0.0, 0.5, 1.0]}, lambda p: (p["x"] - 0.5) ** 2)
        self.assertAlmostEqual(best["x"], 0.5)

    def test_report_refuses_pass_on_the_hypothesis(self):
        report = empty_market_report()
        self.assertEqual(report["nautilus_native_fills"], 0)
        self.assertIsNone(report["raw_tick_count"])
        for key in (
            "MAIN_10_PERCENT_COMPOUND",
            "HEDGE_10_PERCENT_COMPOUND",
            "PORTFOLIO_POSITIVE_GROWTH",
            "OOS_REPRODUCIBILITY",
        ):
            self.assertNotEqual(report["hypothesis"][key], "PASS")
            self.assertEqual(report["hypothesis"][key], "BLOCKED")

    def test_pure_overlay_does_not_hit_hedge_target(self):
        summary = run_overlay(paths=80, months=18, seed=11)
        self.assertEqual(summary.label, "SYNTHETIC_STRUCTURAL_NOT_MARKET")
        self.assertLess(summary.mean_geom_hedge, 0.0)
        self.assertEqual(summary.paths_hedge_ge_target, 0)
        self.assertGreater(summary.simultaneous_loss_rate, 0.0)

    def test_v1_intensity_leaves_a_shortfall(self):
        summary = run_v1_intensity_overlay(paths=60, months=12, seed=11)
        self.assertEqual(summary.paths_hedge_ge_target, 0)
        self.assertGreater(summary.mean_shortfall_when_main_below_bound, 0.0)


if __name__ == "__main__":
    unittest.main()
