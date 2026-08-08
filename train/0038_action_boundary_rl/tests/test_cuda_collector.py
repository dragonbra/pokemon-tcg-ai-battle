from __future__ import annotations

import importlib
from pathlib import Path
import unittest


collector = importlib.import_module(
    "train.0038_action_boundary_rl.rollout.cuda_collector"
)
protocol = importlib.import_module(
    "train.0038_action_boundary_rl.rollout.protocol"
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
        policy_seed=17,
        action_boundary_mode="enabled",
        full_round_draw_limit=full_round_draw_limit,
    )


class CudaCollectorTerminationContractTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
