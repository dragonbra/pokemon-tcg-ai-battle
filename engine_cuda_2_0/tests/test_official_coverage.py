from __future__ import annotations

import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from engine_cuda_2_0.tests.test_official_ir import minimal_payload  # noqa: E402
from ptcg_cuda_engine.official_coverage import (  # noqa: E402
    DOMAIN_SIZES,
    build_official_semantic_coverage,
    validate_official_coverage_inventory,
)


class OfficialCoverageTest(unittest.TestCase):
    def test_matrix_is_complete_and_deterministic(self) -> None:
        left, left_summary = build_official_semantic_coverage(
            minimal_payload(), "test-oracle"
        )
        right, right_summary = build_official_semantic_coverage(
            minimal_payload(), "test-oracle"
        )
        self.assertEqual(left, right)
        self.assertEqual(left_summary, right_summary)
        for name, size in DOMAIN_SIZES.items():
            self.assertEqual(len(left["domains"][name]), size)
            self.assertEqual(
                [entry["id"] for entry in left["domains"][name]],
                list(range(size)),
            )
        self.assertEqual(len(left["domains"]["continuations"]), 1)
        self.assertEqual(len(left["domains"]["cards"]), 1)
        self.assertEqual(len(left["domains"]["skills"]), 1)
        self.assertEqual(len(left["domains"]["attacks"]), 1)
        self.assertEqual(len(left["domains"]["branches"]), 3)
        self.assertEqual(left["summary"]["branch_records"], 3)
        inventory = validate_official_coverage_inventory(minimal_payload(), left)
        self.assertEqual(inventory["branch_records"], 3)

    def test_core_primitive_is_not_reported_as_paired(self) -> None:
        matrix, _ = build_official_semantic_coverage(minimal_payload(), "test-oracle")
        self.assertEqual(
            matrix["domains"]["effects"][0]["status"],
            "core_primitive_ready",
        )
        self.assertNotEqual(matrix["domains"]["effects"][0]["status"], "paired")
        self.assertNotEqual(matrix["domains"]["effects"][0]["status"], "cuda_paired")

    def test_micro_fixture_semantics_are_reported_as_paired(self) -> None:
        matrix, _ = build_official_semantic_coverage(minimal_payload(), "test-oracle")
        for effect_id in (172, 174, 186, 207, 222, 234, 238, 244):
            self.assertEqual(matrix["domains"]["effects"][effect_id]["status"], "paired")
        for target_id in (79, 85, 92):
            self.assertEqual(matrix["domains"]["targets"][target_id]["status"], "paired")
        self.assertEqual(matrix["domains"]["conditions"][9]["status"], "paired")

    def test_branch_inventory_tampering_fails_closed(self) -> None:
        matrix, _ = build_official_semantic_coverage(minimal_payload(), "test-oracle")
        matrix["domains"]["branches"].pop()
        with self.assertRaisesRegex(ValueError, "branch inventory"):
            validate_official_coverage_inventory(minimal_payload(), matrix)


if __name__ == "__main__":
    unittest.main()
