from __future__ import annotations

import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.reference import (  # noqa: E402
    BatchedReferenceEngine,
    Error,
    ReferenceResetSpec,
    Status,
)
from ptcg_cuda_engine.schema import RulePack  # noqa: E402


class ReferenceEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = RulePack.load(CUDA_ENGINE_ROOT / "rules" / "smoke_rules.json")

    def test_twelve_policy_routing_has_fixed_capacity(self) -> None:
        engine = BatchedReferenceEngine(self.pack, batch_size=48, policy_count=12)
        specs = [
            ReferenceResetSpec(seed=100 + env, policy_ids=(env % 12, env % 12))
            for env in range(48)
        ]
        engine.reset(specs)
        routes, counts = engine.route(capacity=4)
        self.assertEqual(counts, [4] * 12)
        self.assertTrue(all(len(row) == 4 and -1 not in row for row in routes))

    def test_draw_and_turn_transition(self) -> None:
        engine = BatchedReferenceEngine(self.pack, batch_size=1, policy_count=2)
        state = engine.reset([ReferenceResetSpec(seed=7)])[0]
        acting = state.actor
        engine.step([1])
        self.assertEqual(state.players[acting].deck_count, 52)
        self.assertEqual(state.players[acting].hand_count, 8)
        self.assertEqual(state.actor, acting ^ 1)
        self.assertEqual(state.turn, 1)
        self.assertEqual(state.status, Status.NEEDS_POLICY)

    def test_generic_damage_reaches_terminal(self) -> None:
        engine = BatchedReferenceEngine(self.pack, batch_size=1, policy_count=2)
        state = engine.reset([ReferenceResetSpec(seed=13, active_hp=(60, 60))])[0]
        for _ in range(5):
            engine.step([2])
            if state.status == Status.TERMINAL:
                break
        self.assertEqual(state.status, Status.TERMINAL)
        self.assertIn(state.winner, (0, 1))
        self.assertEqual(state.error, Error.NONE)

    def test_known_660_1207_path_is_not_silently_fixed(self) -> None:
        engine = BatchedReferenceEngine(self.pack, batch_size=1, policy_count=2)
        state = engine.reset([ReferenceResetSpec(seed=99)])[0]
        engine.step([4])
        self.assertEqual(state.status, Status.ERROR)
        self.assertEqual(state.error, Error.KNOWN_DIVERGENCE_660_1207)

    def test_codec_layout_is_fixed_and_actor_relative(self) -> None:
        engine = BatchedReferenceEngine(self.pack, batch_size=1, policy_count=2)
        state = engine.reset([ReferenceResetSpec(seed=3, active_card_ids=(660, 1207))])[0]
        encoded = engine.encode_policy_v1()[0]
        self.assertEqual(len(encoded["global_cat"]), 8)
        self.assertEqual(len(encoded["global_num"]), 16)
        self.assertEqual(len(encoded["entity_cat"]), 128 * 6)
        self.assertEqual(len(encoded["entity_num"]), 128 * 10)
        self.assertEqual(len(encoded["option_cat"]), 80 * 12)
        owners = [encoded["entity_cat"][1], encoded["entity_cat"][7]]
        self.assertEqual(sorted(owners), [1, 2])
        self.assertEqual(sum(encoded["entity_mask"]), len(state.cards))
        self.assertEqual(sum(encoded["option_mask"]), len(self.pack.actions))

    def test_reset_and_action_digest_is_deterministic(self) -> None:
        specs = [ReferenceResetSpec(seed=17 + index, policy_ids=(index % 3, (index + 1) % 3)) for index in range(8)]
        left = BatchedReferenceEngine(self.pack, batch_size=8, policy_count=3)
        right = BatchedReferenceEngine(self.pack, batch_size=8, policy_count=3)
        left.reset(specs)
        right.reset(specs)
        for actions in ([0] * 8, [1] * 8, [3] * 8):
            left.step(actions)
            right.step(actions)
        self.assertEqual(left.digests(), right.digests())


if __name__ == "__main__":
    unittest.main()
