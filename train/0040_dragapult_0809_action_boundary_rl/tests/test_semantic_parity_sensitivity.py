from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import unittest


snapshot_test = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.tests.test_fixed_snapshot_feature_parity")
transition_test = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.tests.test_deterministic_transition_parity")
sensitivity = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.semantic_parity.sensitivity")


class SemanticParitySensitivityTest(unittest.TestCase):
    def test_historical_runtime_failure_and_current_pass_are_immutable(self):
        root = Path(__file__).parent / "fixtures/semantic_parity_v2"
        fixture = root / "gate_h_execution_identity.json"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["immutable"])
        for entry in manifest["entries"]:
            evidence = root / entry["path"]
            self.assertTrue(evidence.is_file())
            self.assertEqual(
                hashlib.sha256(evidence.read_bytes()).hexdigest(),
                entry["sha256"],
            )
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(payload["known_pre_fix_runtime_artifact"]["status"], "FAIL")
        self.assertEqual(
            payload["known_pre_fix_runtime_artifact"]["first_divergence"]["decision"],
            129,
        )
        self.assertEqual(payload["pre_fix_commit_rebuild"]["status"], "PASS")
        self.assertEqual(payload["post_fix_runtime"]["status"], "PASS")
        self.assertFalse(payload["known_pre_fix_runtime_artifact"]["source_commit_attested"])

    def test_known_in_memory_defects_fail_and_restoration_passes(self):
        result = sensitivity.run_negative_controls(snapshot_test.record(), transition_test.step("official_cpu"))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["option_effect_detected"])
        self.assertTrue(result["continuation_detected"])
        self.assertTrue(result["restored"])


if __name__ == "__main__":
    unittest.main()
