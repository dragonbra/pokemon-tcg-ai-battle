from __future__ import annotations

import importlib
from pathlib import Path
import unittest


PROJECT = "train.0040_dragapult_0809_action_boundary_rl"
ROOT = Path(__file__).resolve().parents[3]


class SemanticParityInventoryTest(unittest.TestCase):
    def test_existing_u230_artifact_exposes_release_blockers(self) -> None:
        inventory = importlib.import_module(
            f"{PROJECT}.semantic_parity.inventory"
        )
        payload = inventory.load_archived_runtime_inventory_fixture(
            ROOT
            / "train/0040_dragapult_0809_action_boundary_rl/tests/fixtures/"
            "semantic_parity_v1/u230_package_load_report.json",
        )

        self.assertEqual(payload["checkpoint"]["update"], 230)
        self.assertEqual(payload["checkpoint"]["critical_missing_keys"], [])
        self.assertEqual(payload["checkpoint"]["critical_unexpected_keys"], [])
        # This pre-fix report predates the authoritative feature contract.  Its
        # absence is itself release-blocking; do not recompute or relabel the
        # immutable historical evidence with the current schema.
        self.assertNotIn("feature_preprocessing", payload["contracts"])
        self.assertNotIn(
            "training_feature_preprocessing_version", payload["checkpoint"]
        )
        self.assertTrue(payload["package"]["strict_load"])
        # The already-built archive is immutable audit evidence.  It predates
        # the fail-closed source fix and must not be relabeled submission-ready.
        self.assertTrue(payload["package"]["silent_legacy_fallback"])
        self.assertFalse(payload["package"]["critic_deployed"])
        self.assertTrue(payload["package"]["model_eval"])
        self.assertEqual(payload["package"]["deck_card_count"], 60)
        self.assertEqual(
            payload["contracts"]["action_boundary"],
            payload["package"]["action_boundary"],
        )
        self.assertNotEqual(
            payload["package"]["action_boundary"],
            inventory.ACTION_BOUNDARY_SCHEMA_VERSION,
        )
        for component in (
            "base_encoder",
            "option_encoder_lora",
            "action_decoder",
            "value",
            "allocation_head",
        ):
            self.assertRegex(
                payload["checkpoint"]["component_sha256"][component],
                r"^[0-9a-f]{64}$",
            )
        for key in (
            "cuda_rules_sha256",
            "cuda_extension_sha256",
            "official_cpu_library_sha256",
            "card_database_sha256",
        ):
            self.assertRegex(payload["runtime"][key], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
