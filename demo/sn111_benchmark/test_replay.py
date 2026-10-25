from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from .replay import benchmark_artifacts, load_artifact_metrics, write_csv
except ImportError:  # unittest discover -s demo/sn111_benchmark imports as top-level
    from replay import benchmark_artifacts, load_artifact_metrics, write_csv


FIXTURES = Path(__file__).parent / "fixtures"


class ReplayBenchmarkTests(unittest.TestCase):
    def test_fixture_metrics_are_deterministic_and_compare_to_baseline(self) -> None:
        result = benchmark_artifacts(
            {
                "baseline": FIXTURES / "baseline.json",
                "candidate": FIXTURES / "candidate.json",
                "repaired": FIXTURES / "repaired.json",
            }
        )
        rows = {row["label"]: row["metrics"] for row in result["artifacts"]}
        self.assertEqual(rows["baseline"]["claim_coverage"], 1.0)
        self.assertEqual(rows["candidate"]["claim_coverage"], 1 / 3)
        self.assertEqual(rows["candidate"]["validation_issue_count"], 6)
        self.assertEqual(
            rows["candidate"]["validation_issue_codes"],
            [
                "duplicate_claim",
                "duplicate_id",
                "evidence_claim_link_not_bidirectional",
                "insufficient_coverage",
                "missing",
                "unresolved_ref",
            ],
        )
        self.assertAlmostEqual(rows["candidate"]["duplicate_rate"], 0.5)
        self.assertEqual(rows["repaired"]["claim_coverage"], 1.0)
        self.assertEqual(rows["repaired"]["validation_issue_count"], 0)
        self.assertEqual(rows["repaired"]["evidence_link_completeness"], 1.0)
        self.assertAlmostEqual(result["comparisons"][0]["deltas"]["claim_coverage"], -2 / 3)

    def test_missing_runtime_metrics_are_explicitly_unavailable(self) -> None:
        metrics = load_artifact_metrics(FIXTURES / "candidate.json")
        self.assertEqual(metrics["latency_status"], "unavailable")
        self.assertEqual(metrics["cost_status"], "unavailable")
        self.assertIsNone(metrics["cost_usd"])

    def test_invalid_json_reports_schema_issue_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{broken", encoding="utf-8")
            metrics = load_artifact_metrics(path)
        self.assertFalse(metrics["schema_valid"])
        self.assertEqual(metrics["schema_issue_count"], 1)

    def test_schema_invalid_artifact_is_gated_from_validation_issue_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"metadata": {}}', encoding="utf-8")
            result = benchmark_artifacts({"baseline": FIXTURES / "baseline.json", "candidate": path})

        comparison = result["comparisons"][0]
        self.assertEqual(comparison["schema_gate"], "blocked")
        self.assertIsNone(comparison["validation_issue_delta"])

    def test_unknown_top_level_field_is_schema_issue_and_blocks_comparison(self) -> None:
        artifact = json.loads((FIXTURES / "candidate.json").read_text(encoding="utf-8"))
        artifact["unexpected"] = "ignored by pydantic"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            metrics = load_artifact_metrics(path)
            result = benchmark_artifacts({"baseline": FIXTURES / "baseline.json", "candidate": path})

        self.assertFalse(metrics["schema_valid"])
        self.assertEqual(metrics["schema_issue_count"], 1)
        self.assertEqual(metrics["schema_issues"][0]["path"], "$.unexpected")
        self.assertEqual(metrics["schema_issues"][0]["type"], "unknown_field")
        self.assertEqual(result["comparisons"][0]["schema_gate"], "blocked")
        self.assertIsNone(result["comparisons"][0]["validation_issue_delta"])

    def test_unknown_nested_field_is_reported(self) -> None:
        artifact = json.loads((FIXTURES / "candidate.json").read_text(encoding="utf-8"))
        artifact["paper"]["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown-nested.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            metrics = load_artifact_metrics(path)

        self.assertFalse(metrics["schema_valid"])
        self.assertEqual(metrics["schema_issue_count"], 1)
        self.assertEqual(metrics["schema_issues"][0]["path"], "$.paper.unexpected")

    def test_csv_export_has_one_row_per_artifact(self) -> None:
        result = benchmark_artifacts({"baseline": FIXTURES / "baseline.json", "candidate": FIXTURES / "candidate.json"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            write_csv(result, path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("claim_coverage", lines[0])


if __name__ == "__main__":
    unittest.main()
