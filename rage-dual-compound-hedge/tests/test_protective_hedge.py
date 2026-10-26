import math
import unittest

from rage_dch.protective_hedge import (
    HedgeParams,
    compute_hedge_intensity,
    drawdown_proxy,
    drawdown_proxy_v2,
    protective_hedge_v1,
    protective_hedge_v2,
    quantity_from_intensity,
)


class ProtectiveHedgeTests(unittest.TestCase):
    def params(self) -> HedgeParams:
        return HedgeParams()

    def test_v1_matches_expanded_formula(self):
        sigma_t, sigma_ref, d, risk, dt = 0.20, 0.16, 0.05, 0.20, 1.0 / 252.0
        p = self.params()
        got = protective_hedge_v1(sigma_t, sigma_ref, d, risk, dt, p)
        logit = -p.kappa1 * (sigma_t / sigma_ref - 1.0) - p.kappa2 * d
        w = 1.0 / (1.0 + math.exp(logit))
        r = (
            p.lambda1 * d
            + p.lambda2 * ((sigma_t - sigma_ref) / sigma_ref)
            + p.lambda3 * sigma_t * math.sqrt(p.tau)
        )
        h = (w * r * math.exp(-p.delta * dt)) / (1.0 + p.rho * risk)
        self.assertAlmostEqual(got.w, w, places=12)
        self.assertAlmostEqual(got.r, r, places=12)
        self.assertAlmostEqual(got.h, h, places=12)

    def test_skeleton_signature_returns_same_h(self):
        p = self.params()
        h = compute_hedge_intensity(0.2, 0.16, 0.05, 0.2, p.delta, 0.0, p.kappa1, p.kappa2, p.lambda1, p.lambda2, p.lambda3, p.tau, p.rho, p.h_max)
        self.assertAlmostEqual(h, protective_hedge_v1(0.2, 0.16, 0.05, 0.2, 0.0, p).h)

    def test_bounds_and_monotonicity(self):
        p = self.params()
        low = protective_hedge_v1(0.16, 0.16, 0.0, 0.1, 0.0, p).h
        high = protective_hedge_v1(0.16, 0.16, 0.2, 0.1, 0.0, p).h
        self.assertGreater(high, low)
        risky = protective_hedge_v1(0.2, 0.16, 0.05, 2.0, 0.0, p).h
        calm_risk = protective_hedge_v1(0.2, 0.16, 0.05, 0.0, 0.0, p).h
        self.assertLess(risky, calm_risk)
        clipped = protective_hedge_v1(1.0, 0.16, 2.0, 0.0, 0.0, HedgeParams(h_max=0.2, lambda1=5, lambda3=5)).h
        self.assertAlmostEqual(clipped, 0.2)

    def test_lambda3_hedges_with_no_drawdown(self):
        h = protective_hedge_v1(0.16, 0.16, 0.0, 0.0, 0.0, self.params()).h
        self.assertGreater(h, 0.0)

    def test_v1_drawdown_long_only_and_hides_bad_price(self):
        self.assertAlmostEqual(drawdown_proxy(100, 90), 10 / 90)
        self.assertEqual(drawdown_proxy(100, 110), 0.0)
        self.assertEqual(drawdown_proxy(100, 0), 0.0)

    def test_v2_side_and_risk_split(self):
        self.assertAlmostEqual(drawdown_proxy_v2(100, 110, "short"), 10 / 110)
        with self.assertRaises(ValueError):
            drawdown_proxy_v2(100, 0, "long")
        p = self.params()
        base = protective_hedge_v2(0.2, 0.16, 0.05, 0.2, 0.0, 0.0, p)
        stressed = protective_hedge_v2(0.2, 0.16, 0.05, 0.2, 1.0, 0.0, p)
        tight_budget = protective_hedge_v2(0.2, 0.16, 0.05, 1.0, 0.0, 0.0, p)
        self.assertGreater(stressed.h, base.h)
        self.assertLess(tight_budget.h, base.h)

    def test_units_percent_is_not_fraction(self):
        a = protective_hedge_v1(0.20, 0.16, 0.0, 0.0, 0.0).h
        b = protective_hedge_v1(20.0, 16.0, 0.0, 0.0, 0.0).h
        self.assertNotAlmostEqual(a, b, places=6)

    def test_quantity_conversion(self):
        q = quantity_from_intensity(0.5, 100_000, 10_000, 0.0)
        self.assertAlmostEqual(q, 5.0)
        wider = quantity_from_intensity(0.5, 100_000, 10_000, 0.01)
        self.assertLess(wider, q)

    def test_bad_reference_vol(self):
        with self.assertRaises(ValueError):
            protective_hedge_v1(0.2, 0.0, 0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
