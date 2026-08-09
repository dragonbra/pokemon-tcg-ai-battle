from __future__ import annotations

import importlib
import math
from pathlib import Path
import unittest

import torch


PROJECT = "train.0038_action_boundary_rl"
ROOT = Path(__file__).resolve().parents[3]
dragapult = importlib.import_module(f"{PROJECT}.action_boundary.dragapult")
collector = importlib.import_module(f"{PROJECT}.rollout.collector")
cuda_boundary = importlib.import_module(f"{PROJECT}.rollout.cuda_action_boundary")


class SemanticParityGateBTest(unittest.TestCase):
    @staticmethod
    def targets(count: int):
        return tuple(
            dragapult.StableTargetIdentity(1, serial, 600 + serial, serial)
            for serial in range(count)
        )

    def test_allocation_contract_covers_area_zero_bench(self) -> None:
        for count in range(1, 9):
            with self.subTest(count=count):
                allocations = dragapult.enumerate_allocations(self.targets(count))
                # Stars-and-bars: six indistinguishable counters across n targets.
                self.assertEqual(len(allocations), math.comb(count + 5, 6))
                self.assertTrue(collector.phantom_macro_eligible(
                    phantom_root=0,
                    target_count=count,
                    action_boundary_mode="enabled",
                    chance_before_allocation=False,
                ))

    def test_cuda_pending_target_relocation_uses_stable_serial(self) -> None:
        # Both options deliberately expose the same card ID and Bench slot.
        # Only option_target -> card serial can identify the intended entity.
        semantic = {
            "option_mask": torch.tensor([[True, True]]),
            "option_cat": torch.zeros((1, 2, 14), dtype=torch.long),
            "option_target": torch.tensor([[1, 2]], dtype=torch.long),
            "card_cat": torch.zeros((1, 2, 14), dtype=torch.long),
        }
        semantic["option_cat"][0, :, 5] = 700
        semantic["option_cat"][0, :, 13] = 4
        semantic["card_cat"][0, 0, 1] = 11  # wire serial 10
        semantic["card_cat"][0, 1, 1] = 12  # wire serial 11
        expected = dragapult.StableTargetIdentity(1, 11, 700, 3)
        matches = cuda_boundary.CudaActionBoundaryAdapter._matching_target_options(
            semantic, row=0, expected=expected
        )
        self.assertEqual(matches.tolist(), [1])

    def test_shared_public_prize_features_match_card_rules(self) -> None:
        features = importlib.import_module(
            f"{PROJECT}.action_boundary.public_card_features"
        )
        prizes = features.card_prize_counts(
            ROOT
            / "train/0038_action_boundary_rl/semantic_policy/assets/"
            "official_full_engine_prototypes_v2.json"
        )
        self.assertEqual(prizes[112], 1)   # Munkidori
        self.assertEqual(prizes[293], 2)   # N's Zoroark ex
        self.assertEqual(prizes[340], 2)   # Yanmega ex is not a Mega Evolution
        self.assertEqual(prizes[652], 3)   # Mega Venusaur ex

    def test_packaged_macro_protocol_error_cannot_fall_back_to_policy(self) -> None:
        source = (
            ROOT
            / "train/0038_action_boundary_rl/kaggle_runtime/compound_inference.py"
        ).read_text(encoding="utf-8")
        handler = source.split("except MacroProtocolError:", 1)[1].split("else:", 1)[0]
        self.assertIn("raise", handler)
        self.assertNotIn("gate = self.gate.classify", handler)

    def test_release_gate_rejects_immutable_u230_archive(self) -> None:
        module = importlib.import_module(
            f"{PROJECT}.semantic_parity.gate_b_macro"
        )
        result = module.build_summary(
            official_parity=ROOT
            / "experiments/0038_action_boundary_rl/OFFICIAL_PARITY_FINAL_V3.json",
            archived_package=ROOT
            / "archive/submission/0038_dragapult_ex_rl_update230",
            fixed_package=ROOT
            / ".tmp/evaluation/0038_semantic_parity_audit/fixed_u230_package",
        )
        self.assertEqual(result["official_exhaustive_n1_n5"]["status"], "PASS")
        self.assertEqual(
            result["expanded_bench_n6_n8"]["enumeration_and_source_contract"],
            "PASS",
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["archived_u230_package"]["maximum_macro_targets"], 5
        )
        self.assertFalse(
            result["archived_u230_package"]["macro_protocol_fail_closed"]
        )


if __name__ == "__main__":
    unittest.main()
