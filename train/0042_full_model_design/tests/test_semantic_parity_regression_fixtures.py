from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = (
    ROOT / "train/0042_full_model_design/tests/fixtures/semantic_parity_v1"
)


class SemanticParityRegressionFixturesTest(unittest.TestCase):
    def test_fixture_hashes_and_schema_are_immutable(self) -> None:
        module = importlib.import_module(
            "train.0042_full_model_design.semantic_parity.freeze_regression_fixtures"
        )
        manifest = module.validate_fixtures(FIXTURES)
        self.assertTrue(manifest["immutable"])
        self.assertEqual(len(manifest["entries"]), 7)

    def test_fixed_trace_is_exactly_283_ordered_primitive_decisions(self) -> None:
        payload = gzip.decompress(
            (FIXTURES / "fixed_283_primitive_trace.jsonl.gz").read_bytes()
        ).decode("utf-8")
        rows = [json.loads(line) for line in payload.splitlines()]
        self.assertEqual(len(rows), 283)
        self.assertEqual([row["decision"] for row in rows], list(range(283)))
        self.assertTrue(all(isinstance(row["ordered_action"], list) for row in rows))

    def test_prefx_failures_are_preserved(self) -> None:
        gate_a = json.loads(
            (FIXTURES / "gate_a_zoroark_decision129.json").read_text()
        )
        self.assertEqual(gate_a["expected"]["cuda_pod_error"], 20)
        self.assertEqual(gate_a["expected"]["cuda_pod_detail"], 403)
        self.assertEqual(gate_a["pre_action_trace_decision"], 128)
        field = json.loads(
            (FIXTURES / "gate_c_decision0_field_diff.json").read_text()
        )
        self.assertIn(
            "option_skill_mask", {row["field"] for row in field["diffs"]}
        )
        greedy = json.loads(
            (FIXTURES / "gate_c_decision60_greedy.json").read_text()
        )
        self.assertEqual(greedy["snapshot"]["decision"], 60)
        self.assertEqual(greedy["policy"]["cpu_greedy"], [4])
        self.assertEqual(greedy["policy"]["cuda_greedy"], [2])
        value = json.loads(
            (FIXTURES / "gate_c_value_divergences.json").read_text()
        )
        self.assertAlmostEqual(value["maximum"]["absolute_error"], 0.4034011, places=6)
        self.assertEqual(value["total_sign_divergences"], 2)

    def test_expanded_bench_protocol_contract(self) -> None:
        fixture = json.loads(
            (FIXTURES / "phantom_n6_n8_protocol.json").read_text()
        )
        self.assertEqual(
            [row["canonical_allocation_count"] for row in fixture["cases"]],
            [462, 924, 1716],
        )
        for row in fixture["cases"]:
            self.assertEqual(sum(row["counters"]), 6)
            self.assertEqual(len(row["primitive_target_serials"]), 6)
            self.assertEqual(row["expected_policy_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
