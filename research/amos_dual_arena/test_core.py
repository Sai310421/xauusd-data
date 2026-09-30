import json
import tempfile
import unittest
from pathlib import Path

from core import Candidate, edge_score, feedback_queue, load_candidates, rank, score_candidate


class DualArenaTests(unittest.TestCase):
    def test_profit_and_edge_leagues_can_have_different_winners(self):
        durable = Candidate.from_dict({
            "candidate_id": "durable",
            "agent": "agent-a",
            "strategy": "edge-a",
            "initial_equity": 1000,
            "final_equity": 1800,
            "max_dd_pct": 4,
            "profit_factor": 2.4,
            "trades": 1200,
            "ruin_probability": 0.0,
            "oos_retention": 0.9,
            "cost_retention": 0.9,
            "regime_pass_rate": 0.85,
            "discovery_efficiency": 0.5,
        })
        aggressive = Candidate.from_dict({
            "candidate_id": "aggressive",
            "agent": "agent-b",
            "strategy": "edge-b",
            "initial_equity": 1000,
            "final_equity": 4000,
            "max_dd_pct": 35,
            "profit_factor": 1.4,
            "trades": 300,
            "ruin_probability": 0.08,
            "oos_retention": 0.35,
            "cost_retention": 0.4,
            "regime_pass_rate": 0.4,
            "discovery_efficiency": 0.2,
        })
        scores = [score_candidate(durable), score_candidate(aggressive)]
        leagues = rank(scores)
        self.assertEqual(leagues["arena_a_edge_discovery"][0]["candidate_id"], "durable")
        self.assertEqual(leagues["arena_b_absolute_profit"][0]["candidate_id"], "aggressive")
        self.assertEqual(leagues["arena_b_prop_dd5"][0]["candidate_id"], "durable")

    def test_smoke_manifest_without_metrics_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.json"
            p.write_text(json.dumps({"experiment_id": "smoke", "status": "SMOKE_COMPLETED"}), encoding="utf-8")
            self.assertEqual(load_candidates([p]), [])

    def test_winners_enter_feedback_queue(self):
        c = Candidate.from_dict({
            "candidate_id": "x",
            "final_equity": 1500,
            "max_dd_pct": 3,
            "profit_factor": 2,
            "trades": 500,
            "oos_retention": 0.8,
            "cost_retention": 0.8,
            "regime_pass_rate": 0.8,
        })
        q = feedback_queue([score_candidate(c)], 1)
        self.assertEqual(q[0]["action"], "MATH_EDGE_DIAGNOSTIC")
        self.assertIn("exit", q[0]["decompose"])


if __name__ == "__main__":
    unittest.main()
