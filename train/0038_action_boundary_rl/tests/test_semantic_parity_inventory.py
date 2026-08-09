from __future__ import annotations

import importlib
from pathlib import Path
import unittest


PROJECT = "train.0038_action_boundary_rl"
ROOT = Path(__file__).resolve().parents[3]


class SemanticParityInventoryTest(unittest.TestCase):
    def test_existing_u230_artifact_exposes_release_blockers(self) -> None:
        inventory = importlib.import_module(
            f"{PROJECT}.semantic_parity.inventory"
        )
        payload = inventory.build_runtime_inventory(
            checkpoint=ROOT
            / "rl_runs/0038_action_boundary_rl/versions/"
            "V10_complete_512_rollout_fresh_rl/checkpoint/update-000230.pt",
            package_root=ROOT
            / "archive/submission/0038_dragapult_ex_rl_update230",
        )

        self.assertEqual(payload["checkpoint"]["update"], 230)
        self.assertEqual(payload["checkpoint"]["critical_missing_keys"], [])
        self.assertEqual(payload["checkpoint"]["critical_unexpected_keys"], [])
        self.assertFalse(payload["checkpoint"]["training_distribution_compatible"])
        self.assertTrue(payload["package"]["strict_load"])
        # The already-built archive is immutable audit evidence.  It predates
        # the fail-closed source fix and must not be relabeled submission-ready.
        self.assertTrue(payload["package"]["silent_legacy_fallback"])
        self.assertFalse(payload["package"]["critic_deployed"])
        self.assertTrue(payload["package"]["model_eval"])
        self.assertEqual(payload["package"]["deck_card_count"], 60)
        self.assertNotEqual(
            payload["contracts"]["action_boundary"],
            payload["package"]["action_boundary"],
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
