from __future__ import annotations

import unittest
from pathlib import Path

from engine_cuda_2_0.tools import evaluate_policy_0806_cuda as subject


class Policy0806CudaOutputIsolationTest(unittest.TestCase):
    def tearDown(self) -> None:
        subject.configure_output_roots()

    def test_custom_evidence_namespace_does_not_replace_canonical_index(self) -> None:
        output = (
            subject.ROOT
            / "docs/evaluation/combat_mat/policy_0806/postfix_semantic_parity"
        )
        temporary = subject.ROOT / ".tmp/evaluation/postfix_semantic_parity"

        subject.configure_output_roots(output_root=output, temp_root=temporary)

        self.assertEqual(subject.OUTPUT_ROOT, output.resolve())
        self.assertEqual(subject.TEMP_ROOT, temporary.resolve())
        self.assertNotEqual(subject.OUTPUT_ROOT, subject.CANONICAL_OUTPUT_ROOT)

    def test_custom_roots_fail_closed_outside_audit_namespaces(self) -> None:
        with self.assertRaises(ValueError):
            subject.configure_output_roots(output_root=Path("/tmp/not-a-report"))
        with self.assertRaises(ValueError):
            subject.configure_output_roots(temp_root=Path("/tmp/not-a-trace"))


if __name__ == "__main__":
    unittest.main()
