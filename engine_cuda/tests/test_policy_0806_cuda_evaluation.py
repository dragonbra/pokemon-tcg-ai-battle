from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evaluation.frozen_0806 import exact_deck_sha256
from engine_cuda.tools.evaluate_policy_0806_cuda import (
    build_cuda_schedule,
    build_cuda_game_records,
    merge_cuda_chunk_results,
    resolve_candidate_deck_path,
)


@dataclass(frozen=True)
class Entry:
    deck_id: str
    games: int


class Policy0806CudaEvaluationTest(unittest.TestCase):
    def test_candidate_deck_resolves_numbered_pool_not_shared_policy_root(self) -> None:
        cards = tuple(range(1, 61))
        candidate = SimpleNamespace(
            root=Path("shared/policy/root"),
            deck=cards,
            package_manifest={
                "frozen_deck_number": "007",
                "exact_deck_sha256": exact_deck_sha256(cards),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            deck_root = Path(directory)
            package = deck_root / "007_dragapult_ex"
            package.mkdir()
            (package / "deck.csv").write_text(
                "".join(f"{card}\n" for card in cards), encoding="ascii"
            )
            (package / "manifest.json").write_text(
                json.dumps({"deck_id": "dragapult_ex_test"}), encoding="utf-8"
            )

            resolved = resolve_candidate_deck_path(candidate, deck_root)

        self.assertEqual(resolved.name, "deck.csv")
        self.assertEqual(resolved.parent.name, "007_dragapult_ex")

    def test_chunk_merge_preserves_schedule_order_and_totals(self) -> None:
        def chunk(offset: int, results: list[int]) -> dict:
            games = len(results)
            return {
                "passed": True,
                "collector": {
                    "games": games,
                    "completed_games": games,
                    "errors": 0,
                    "wall_seconds": 2.0,
                    "gpu_seconds": 1.5,
                    "routed_ready_rows": 10,
                },
                "determinism": {
                    "game_results": results,
                    "game_results_sha256": "unused",
                    "terminal_state_sha256": f"terminal-{offset}",
                },
                "progress_guard": {
                    "forfeit_schedule_indices": [offset] if offset else [],
                    "forfeit_count": int(bool(offset)),
                },
                "schedule": {"schedule_offset": offset, "used_jobs": games},
                "memory": {
                    "torch_peak_allocated_bytes": 100 + offset,
                    "torch_peak_reserved_bytes": 200 + offset,
                },
            }

        merged = merge_cuda_chunk_results(
            [chunk(2, [2, 1]), chunk(0, [1, 2])], expected_games=4
        )

        self.assertEqual(merged["determinism"]["game_results"], [1, 2, 2, 1])
        self.assertEqual(merged["collector"]["completed_games"], 4)
        self.assertEqual(merged["collector"]["chunk_count"], 2)
        self.assertEqual(merged["progress_guard"]["forfeit_schedule_indices"], [2])
        self.assertEqual(merged["memory"]["torch_peak_reserved_bytes"], 202)

    def test_schedule_has_eight_independent_seed_replicas_and_balanced_seats(self) -> None:
        schedule = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=(Entry("opponent_a", 2), Entry("opponent_b", 1)),
            evaluation_seed=341_512_806,
        )

        jobs = schedule["jobs"]
        self.assertEqual(len(jobs), 24)
        self.assertEqual(sum(job["focal_first"] for job in jobs), 12)
        grouped: dict[tuple[str, int], list[dict]] = {}
        for job in jobs:
            grouped.setdefault((job["opponent_id"], job["slot"]), []).append(job)
        self.assertEqual(len(grouped), 3)
        for replicas in grouped.values():
            self.assertEqual(len(replicas), 8)
            self.assertEqual(len({job["engine_seed"] for job in replicas}), 8)
            self.assertEqual(len({job["search_seed"] for job in replicas}), 8)
            self.assertEqual(sum(job["focal_first"] for job in replicas), 4)

    def test_schedule_changes_with_focal_identity_but_is_reproducible(self) -> None:
        entries = (Entry("opponent_a", 1),)
        left = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=entries,
            evaluation_seed=341_512_806,
        )
        replay = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=entries,
            evaluation_seed=341_512_806,
        )
        other = build_cuda_schedule(
            focal_deck_id="candidate_b",
            entries=entries,
            evaluation_seed=341_512_806,
        )

        self.assertEqual(left, replay)
        self.assertNotEqual(
            left["jobs"][0]["engine_seed"], other["jobs"][0]["engine_seed"]
        )

    def test_game_records_preserve_pairing_keys_and_focal_outcome(self) -> None:
        schedule = {
            "jobs": [
                {
                    "game_id": "first",
                    "opponent_id": "opponent_a",
                    "replica": 0,
                    "slot": 0,
                    "engine_seed": 11,
                    "search_seed": 12,
                    "focal_first": True,
                },
                {
                    "game_id": "second",
                    "opponent_id": "opponent_a",
                    "replica": 1,
                    "slot": 0,
                    "engine_seed": 21,
                    "search_seed": 22,
                    "focal_first": False,
                },
            ]
        }
        result = {
            "determinism": {"game_results": [1, 1]},
            "progress_guard": {
                "forfeit_schedule_indices": [1],
                "turn_limit_draw_schedule_indices": [],
            },
        }

        records = build_cuda_game_records(schedule, result)

        self.assertEqual(records[0]["focal_outcome"], "win")
        self.assertEqual(records[1]["focal_outcome"], "loss")
        self.assertFalse(records[0]["repeat_forfeit"])
        self.assertTrue(records[1]["repeat_forfeit"])
        self.assertEqual(records[1]["engine_seed"], 21)


if __name__ == "__main__":
    unittest.main()
