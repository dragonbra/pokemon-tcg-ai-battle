from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


collector = importlib.import_module(
    "train.0042_full_model_design.rollout.cuda_collector"
)
protocol = importlib.import_module(
    "train.0042_full_model_design.rollout.protocol"
)


def _job(game_id: str, *, full_round_draw_limit: int = 50):
    deck = tuple(range(1, 61))
    return protocol.RolloutJob(
        game_id=game_id,
        opponent_id="opponent",
        focal_first=True,
        seed=123,
        source_policy_update=4,
        focal_deck=deck,
        opponent_deck=deck,
        runtime_root=Path("runtime"),
        opponent_policy_id="Policy-0809",
        policy_seed=17,
        action_boundary_mode="enabled",
        full_round_draw_limit=full_round_draw_limit,
    )


class CudaCollectorTerminationContractTest(unittest.TestCase):
    def test_mixed_deck_rows_use_each_lane_exact_resource_ledger(self) -> None:
        deck_a = (1,) * 30 + (2,) * 30
        deck_b = (3,) * 20 + (4,) * 40
        jobs = (
            SimpleNamespace(decks=(deck_a, deck_b), focal_player=0),
            SimpleNamespace(decks=(deck_a, deck_b), focal_player=0),
        )
        semantic = {
            "resource_cat": torch.tensor([
                [[1, 0, 0, 0], [2, 0, 0, 0]],
                [[3, 0, 0, 0], [4, 0, 0, 0]],
            ]),
            "resource_num": torch.tensor([
                [[30.0] + [0.0] * 14, [30.0] + [0.0] * 14],
                [[20.0] + [0.0] * 14, [40.0] + [0.0] * 14],
            ]),
            "resource_mask": torch.ones((2, 2), dtype=torch.bool),
        }
        records = collector.audit_exact_deck_rows(
            semantic=semantic,
            ready=torch.ones(2, dtype=torch.bool),
            focal_route=torch.tensor([True, False]),
            lane_job=torch.tensor([0, 1]),
            jobs=jobs,
            already_audited=set(),
        )
        self.assertEqual(records, {(0, "focal"), (1, "opponent")})

    def test_swapped_lane_static_fields_hard_fail(self) -> None:
        deck_a = (1,) * 60
        deck_b = (2,) * 60
        jobs = (
            SimpleNamespace(decks=(deck_a, deck_b), focal_player=0),
            SimpleNamespace(decks=(deck_a, deck_b), focal_player=0),
        )
        semantic = {
            "resource_cat": torch.tensor([[[1, 0, 0, 0]], [[1, 0, 0, 0]]]),
            "resource_num": torch.tensor([[[60.0] + [0.0] * 14]] * 2),
            "resource_mask": torch.ones((2, 1), dtype=torch.bool),
        }
        with self.assertRaisesRegex(RuntimeError, "exact-deck routing mismatch"):
            collector.audit_exact_deck_rows(
                semantic=semantic,
                ready=torch.ones(2, dtype=torch.bool),
                focal_route=torch.tensor([True, False]),
                lane_job=torch.tensor([0, 1]),
                jobs=jobs,
                already_audited=set(),
            )

    def test_job_requires_concrete_opponent_policy_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "opponent_policy_id"):
            protocol.RolloutJob(
                "a", "opponent", True, 1, 0, (1,) * 60, (2,) * 60,
                Path("runtime"), "",
            )

    def test_lane_policy_binding_mismatch_hard_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "FATAL.*binding mismatch"):
            protocol.require_opponent_policy_binding(
                [_job("a")], materialized_policy_id="Policy-not-0809"
            )

    def test_fifty_full_rounds_map_to_engine_turn_99(self) -> None:
        self.assertEqual(collector._engine_turn_draw_limit([_job("a")]), 99)

    def test_zero_disables_turn_limit(self) -> None:
        self.assertEqual(
            collector._engine_turn_draw_limit(
                [_job("a", full_round_draw_limit=0)]
            ),
            0,
        )

    def test_mixed_limits_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "full-round draw limit"):
            collector._engine_turn_draw_limit(
                [_job("a"), _job("b", full_round_draw_limit=40)]
            )

    def test_negative_limit_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "full_round_draw_limit"):
            _job("a", full_round_draw_limit=-1)

    def test_turn_limit_draw_has_distinct_terminal_reason(self) -> None:
        self.assertEqual(
            collector._termination_status(2, forfeits=set(), turn_limit_draws={2}),
            "turn_limit_draw",
        )

    def test_repeat_forfeit_takes_precedence(self) -> None:
        self.assertEqual(
            collector._termination_status(2, forfeits={2}, turn_limit_draws={2}),
            "repeat_forfeit",
        )

    def test_chunked_collector_adds_turn_limit_draw_counts(self) -> None:
        original = collector.CudaFullSemanticRolloutCollector

        class FakeCollector:
            counts = iter((2.0, 3.0))

            def __init__(self, *_args, **_kwargs) -> None:
                self.count = next(self.counts)

            def collect(self, _jobs):
                return []

            def metrics(self):
                return {"rollout/cuda_turn_limit_draws": self.count}

        collector.CudaFullSemanticRolloutCollector = FakeCollector
        try:
            chunked = collector.ChunkedCudaRolloutCollector(
                None, None, rollout_batch_size=1
            )
            chunked.collect([_job("a"), _job("b")])
            self.assertEqual(
                chunked.metrics()["rollout/cuda_turn_limit_draws"], 5.0
            )
        finally:
            collector.CudaFullSemanticRolloutCollector = original

    def test_fixed_budget_is_sampled_across_the_rollout(self) -> None:
        original = collector.CudaFullSemanticRolloutCollector

        class FakeCollector:
            captured = []

            def __init__(self, *_args, **kwargs) -> None:
                self.captured.append(kwargs.get("record_job_indices"))

            def collect(self, _jobs):
                return []

            def metrics(self):
                return {}

        collector.CudaFullSemanticRolloutCollector = FakeCollector
        try:
            chunked = collector.ChunkedCudaRolloutCollector(
                None, None, rollout_batch_size=2,
                trajectory_games_per_update=2,
                record_trajectory=True,
            )
            chunked.collect([_job(str(index)) for index in range(4)])
            self.assertEqual(sum(len(item) for item in FakeCollector.captured), 2)
            self.assertEqual(len(FakeCollector.captured), 2)
        finally:
            collector.CudaFullSemanticRolloutCollector = original

    def test_trajectory_sampling_is_balanced_across_frequency_units(self) -> None:
        jobs = [_job(str(index)) for index in range(2048)]
        selected = collector._trajectory_job_indices(jobs, 512)
        self.assertEqual(len(selected), 512)
        self.assertEqual(
            [sum(start <= index < start + 256 for index in selected)
             for start in range(0, 2048, 256)],
            [64] * 8,
        )
        self.assertEqual(selected, collector._trajectory_job_indices(jobs, 512))

    def test_diagnostic_batch_smaller_than_budget_keeps_every_episode(self) -> None:
        original = collector.CudaFullSemanticRolloutCollector

        class FakeCollector:
            captured = None

            def __init__(self, *_args, **kwargs) -> None:
                self.captured = kwargs.get("record_job_indices")
                FakeCollector.captured = self.captured

            def collect(self, _jobs):
                return []

            def metrics(self):
                return {}

        collector.CudaFullSemanticRolloutCollector = FakeCollector
        try:
            chunked = collector.ChunkedCudaRolloutCollector(
                None, None, rollout_batch_size=512,
                trajectory_games_per_update=512,
                record_trajectory=True,
            )
            chunked.collect([_job("a"), _job("b")])
            self.assertEqual(FakeCollector.captured, {0, 1})
        finally:
            collector.CudaFullSemanticRolloutCollector = original


if __name__ == "__main__":
    unittest.main()
