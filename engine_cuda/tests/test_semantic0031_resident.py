from __future__ import annotations

import inspect
import unittest

import torch

from ptcg_cuda_engine.semantic0031_resident import (
    ResidentJob,
    ResidentLaneQueue,
    apply_turn_limit_draws,
    run_resident_greedy_jobs,
)


def _jobs(count: int) -> tuple[ResidentJob, ...]:
    deck = tuple(range(1, 61))
    return tuple(
        ResidentJob(
            schedule_index=index,
            decks=(deck, deck),
            engine_seed=100 + index // 2,
            focal_player=index % 2,
            opponent_id=f"opponent-{index // 2}",
        )
        for index in range(count)
    )


class ResidentLaneQueueTest(unittest.TestCase):
    def test_short_lane_refills_without_waiting_for_long_lane(self) -> None:
        queue = ResidentLaneQueue(_jobs(5), lane_count=2, device="cpu")
        initial = queue.initial()
        self.assertEqual(initial.job_indices.tolist(), [0, 1])

        refill = queue.complete(
            torch.tensor([True, False]), torch.tensor([1, 0], dtype=torch.uint8)
        )

        self.assertEqual(refill.lane_indices.tolist(), [0])
        self.assertEqual(refill.job_indices.tolist(), [2])
        self.assertEqual(queue.lane_job.tolist(), [2, 1])

    def test_results_remain_in_schedule_order_after_out_of_order_completion(self) -> None:
        queue = ResidentLaneQueue(_jobs(4), lane_count=2, device="cpu")
        queue.initial()
        queue.complete(
            torch.tensor([False, True]), torch.tensor([0, 2], dtype=torch.uint8)
        )
        queue.complete(
            torch.tensor([True, True]), torch.tensor([1, 1], dtype=torch.uint8)
        )
        queue.complete(
            torch.tensor([True, False]), torch.tensor([2, 0], dtype=torch.uint8)
        )

        self.assertEqual(queue.completed, 4)
        self.assertEqual(queue.results.tolist(), [1, 2, 1, 2])

    def test_paired_seed_and_seat_metadata_survive_refill(self) -> None:
        jobs = _jobs(4)
        queue = ResidentLaneQueue(jobs, lane_count=1, device="cpu")
        queue.initial()
        seen = []
        for result in (1, 2, 1, 2):
            job = jobs[int(queue.lane_job[0])]
            seen.append((job.engine_seed, job.focal_player))
            queue.complete(
                torch.tensor([True]), torch.tensor([result], dtype=torch.uint8)
            )

        self.assertEqual(seen, [(100, 0), (100, 1), (101, 0), (101, 1)])


class ResidentTurnLimitTest(unittest.TestCase):
    def test_decision_trace_hook_is_default_off(self) -> None:
        parameter = inspect.signature(run_resident_greedy_jobs).parameters[
            "decision_trace_sink"
        ]
        self.assertIsNone(parameter.default)

    def test_default_contract_is_fifty_full_rounds(self) -> None:
        parameter = inspect.signature(run_resident_greedy_jobs).parameters[
            "engine_turn_draw_limit"
        ]
        self.assertEqual(parameter.default, 100)

    def test_turn_limit_marks_only_selected_lanes_as_terminal_draws(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self._statuses = torch.tensor([1, 1, 2], dtype=torch.uint8)
                self._results = torch.tensor([2, 1, 2], dtype=torch.uint8)

            def statuses(self):
                return self._statuses

            def game_results(self):
                return self._results

        engine = FakeEngine()
        apply_turn_limit_draws(engine, torch.tensor([True, False, False]))
        self.assertEqual(engine.statuses().tolist(), [2, 1, 2])
        self.assertEqual(engine.game_results().tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
