import unittest

try:
    from .simulate import compare_strategies, synthetic_candidates, synthetic_paper
except ImportError:  # unittest discover -s demo/sn111_sim loads this as a top-level module.
    from simulate import compare_strategies, synthetic_candidates, synthetic_paper


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = compare_strategies(synthetic_paper(), synthetic_candidates())
        self.by_id = {item["miner_id"]: item for item in self.result["candidates"]}

    def test_expected_gates_are_exercised(self) -> None:
        self.assertTrue(self.by_id["miner-good"]["gates"]["accepted"])
        self.assertIn("duplicate_gate", self.by_id["miner-duplicate"]["gates"]["reasons"])
        self.assertIn("coverage_gate", self.by_id["miner-partial"]["gates"]["reasons"])
        self.assertIn("grounding_gate", self.by_id["miner-ungrounded"]["gates"]["reasons"])
        self.assertIn("provenance_gate", self.by_id["miner-untrusted"]["gates"]["reasons"])

    def test_strategies_make_different_decisions(self) -> None:
        strategies = self.result["strategies"]
        self.assertEqual(strategies["strict_gate_first"]["selected"], ["miner-good"])
        self.assertEqual(strategies["quality_first"]["selected"][0], "miner-good")
        self.assertLess(strategies["quality_first"]["accepted_selected"], 2)


if __name__ == "__main__":
    unittest.main()
