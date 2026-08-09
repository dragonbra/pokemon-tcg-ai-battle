from __future__ import annotations

import importlib
import hashlib
import json
import math
from pathlib import Path
import threading
import unittest

import torch


PROJECT = "train.0040_dragapult_0809_action_boundary_rl"
ROOT = Path(__file__).resolve().parents[3]
dragapult = importlib.import_module(f"{PROJECT}.action_boundary.dragapult")
collector = importlib.import_module(f"{PROJECT}.rollout.collector")
cuda_boundary = importlib.import_module(f"{PROJECT}.rollout.cuda_action_boundary")
official_parity = importlib.import_module(f"{PROJECT}.action_boundary.official_parity")
pool_worker = importlib.import_module(f"{PROJECT}.rollout.pool_worker")


class SemanticParityGateBTest(unittest.TestCase):
    def test_cuda_fail_closed_uses_shared_protocol_error(self) -> None:
        protocol = importlib.import_module(
            f"{PROJECT}.action_boundary.macro_protocol"
        )
        self.assertIs(cuda_boundary.MacroProtocolError, protocol.MacroProtocolError)

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
        # Both options deliberately expose the same card ID.  The intended
        # entity has moved away from its initial Bench slot, so only
        # option_target -> stable serial can identify it.
        semantic = {
            "option_mask": torch.tensor([[True, True]]),
            "option_cat": torch.zeros((1, 2, 14), dtype=torch.long),
            # Official type-3 callbacks encode the in-play card as the option
            # source.  option_target remains empty by CPU schema contract.
            "option_source": torch.tensor([[1, 2]], dtype=torch.long),
            "option_target": torch.tensor([[0, 0]], dtype=torch.long),
            "card_cat": torch.zeros((1, 2, 14), dtype=torch.long),
        }
        semantic["option_cat"][0, :, 5] = 700
        semantic["option_cat"][0, 0, 13] = 4
        semantic["option_cat"][0, 1, 13] = 8
        semantic["card_cat"][0, 0, 1] = 11  # wire serial 10
        semantic["card_cat"][0, 1, 1] = 12  # wire serial 11
        expected = dragapult.StableTargetIdentity(1, 11, 700, 3)
        matches = cuda_boundary.CudaActionBoundaryAdapter._matching_target_options(
            semantic, row=0, expected=expected
        )
        self.assertEqual(matches.tolist(), [1])

    def test_cuda_phantom_relocation_does_not_invent_target_relation(self) -> None:
        semantic = {
            "option_mask": torch.tensor([[True]]),
            "option_cat": torch.zeros((1, 1, 14), dtype=torch.long),
            "option_source": torch.tensor([[1]], dtype=torch.long),
            "option_target": torch.tensor([[0]], dtype=torch.long),
            "card_cat": torch.zeros((1, 1, 14), dtype=torch.long),
        }
        semantic["option_cat"][0, 0, 5] = 131
        semantic["card_cat"][0, 0, 0] = 131
        semantic["card_cat"][0, 0, 1] = 84  # wire serial 83
        expected = dragapult.StableTargetIdentity(1, 83, 131, 2)
        self.assertEqual(
            cuda_boundary.CudaActionBoundaryAdapter._matching_target_options(
                semantic, row=0, expected=expected
            ).tolist(),
            [0],
        )

    def test_shared_public_prize_features_match_card_rules(self) -> None:
        features = importlib.import_module(
            f"{PROJECT}.action_boundary.public_card_features"
        )
        prizes = features.card_prize_counts(
            ROOT
            / "train/0040_dragapult_0809_action_boundary_rl/semantic_policy/assets/"
            "official_full_engine_prototypes_v2.json"
        )
        self.assertEqual(prizes[112], 1)   # Munkidori
        self.assertEqual(prizes[293], 2)   # N's Zoroark ex
        self.assertEqual(prizes[340], 2)   # Yanmega ex is not a Mega Evolution
        self.assertEqual(prizes[652], 3)   # Mega Venusaur ex

    def test_packaged_macro_protocol_error_cannot_fall_back_to_policy(self) -> None:
        source = (
            ROOT
            / "train/0040_dragapult_0809_action_boundary_rl/kaggle_runtime/compound_inference.py"
        ).read_text(encoding="utf-8")
        handler = source.split("except MacroProtocolError:", 1)[1].split("else:", 1)[0]
        self.assertIn("raise", handler)
        self.assertNotIn("gate = self.gate.classify", handler)

    def test_official_rollout_macro_error_cannot_recall_policy(self) -> None:
        source = (
            ROOT / "train/0040_dragapult_0809_action_boundary_rl/rollout/pool_worker.py"
        ).read_text(encoding="utf-8")
        handler = source.split("except MacroProtocolError as error:", 1)[1]
        handler = handler.split("else:", 1)[0]
        self.assertIn("raise", handler)
        self.assertNotIn("phantom_dive_legacy_fallback", handler)

    def test_cuda_first_player_selection_is_harness_bypass(self) -> None:
        semantic = {
            "option_mask": torch.tensor([[True, True], [True, False]]),
            "min_count": torch.tensor([1, 1]),
            "max_count": torch.tensor([1, 1]),
            # Encoded categorical IDs are one-based; 42 means context 41.
            "global_cat": torch.tensor([[1, 42], [1, 1]]),
        }
        mask = cuda_boundary._first_player_harness_mask(
            semantic, torch.tensor([True, True])
        )
        self.assertEqual(mask.tolist(), [True, False])
        semantic["option_mask"][0, 1] = False
        with self.assertRaisesRegex(RuntimeError, "first-player harness contract drift"):
            cuda_boundary._first_player_harness_mask(
                semantic, torch.tensor([True, True])
            )

    def test_release_gate_rejects_immutable_u230_archive(self) -> None:
        module = importlib.import_module(
            f"{PROJECT}.semantic_parity.gate_b_macro"
        )
        result = module.build_summary(
            official_parity=ROOT
            / "experiments/0040_dragapult_0809_action_boundary_rl/OFFICIAL_PARITY_FINAL_V3.json",
            archived_package=ROOT
            / "train/0040_dragapult_0809_action_boundary_rl/tests/fixtures/semantic_parity_v2/"
              "u230_archived_macro_contract.json",
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

    def test_real_official_area_zero_protocol_fixtures_replay(self) -> None:
        from evaluation.runtime.seeded import build_seeded_runtime, load_seeded_library

        fixture_root = (
            ROOT
            / "train/0040_dragapult_0809_action_boundary_rl/tests/fixtures/"
            "phantom_area_zero_official_v1"
        )
        fixture = json.loads((fixture_root / "scenarios.json").read_text())
        focal_path = ROOT / fixture["focal_deck_path"]
        opponent_path = ROOT / fixture["opponent_deck_path"]
        self.assertEqual(
            hashlib.sha256(focal_path.read_bytes()).hexdigest(),
            fixture["focal_deck_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(opponent_path.read_bytes()).hexdigest(),
            fixture["opponent_deck_sha256"],
        )
        focal = tuple(int(value) for value in focal_path.read_text().split())
        opponent = tuple(int(value) for value in opponent_path.read_text().split())
        runtime = build_seeded_runtime()
        library = load_seeded_library(runtime.library_path)
        lock = threading.Lock()

        def public_hash(observation):
            payload = {
                "current": observation["current"],
                "select": observation["select"],
            }
            return hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()

        for raw in fixture["scenarios"]:
            with self.subTest(n=raw["n"]):
                battle = pool_worker._PointerBattle(library, lock, [])
                try:
                    observation = battle.start(
                        focal, opponent, raw["seed"], raw["search_seed"]
                    )
                    for action in raw["primitive_action_prefix"][:-1]:
                        observation = battle.select(action)
                    self.assertEqual(
                        public_hash(observation), raw["pre_root_public_state_sha256"]
                    )
                    self.assertEqual(
                        [item["id"] for item in observation["current"]["stadium"]],
                        [fixture["stadium_card_id"]],
                    )
                    observation = battle.select(raw["primitive_action_prefix"][-1])
                    self.assertEqual(
                        public_hash(observation),
                        raw["first_callback_public_state_sha256"],
                    )
                    for key, value in fixture["callback_contract"].items():
                        self.assertEqual(observation["select"].get(key), value)
                finally:
                    battle.finish()

                targets = tuple(sorted(
                    dragapult.StableTargetIdentity(**target)
                    for target in raw["targets"]
                ))
                scenario = official_parity.ParityScenario(
                    battle_id=f"area-zero-n{raw['n']}",
                    seed=raw["seed"],
                    search_seed=raw["search_seed"],
                    focal_deck=focal,
                    opponent_deck=opponent,
                    primitive_action_prefix=tuple(
                        tuple(action) for action in raw["primitive_action_prefix"]
                    ),
                    targets=targets,
                )
                counters = (6,) + (0,) * (raw["n"] - 1)
                allocation = dragapult.DragapultDamageAllocation(targets, counters)
                macro_state, macro_actions = official_parity._macro_sequence(
                    scenario, allocation, library, lock
                )
                sequential_state, sequential_actions = official_parity._run_sequence(
                    scenario,
                    allocation.primitive_target_serials(),
                    library,
                    lock,
                )
                self.assertEqual(macro_actions, sequential_actions)
                self.assertEqual(
                    official_parity._authoritative_snapshot(macro_state),
                    official_parity._authoritative_snapshot(sequential_state),
                )


if __name__ == "__main__":
    unittest.main()
