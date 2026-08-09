from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evaluation.frozen_0806 import exact_deck_sha256
from engine_cuda.tools.evaluate_policy_0806_cuda import (
    EXPECTED_GAMES,
    POLICY_SHA256,
    _cached_result_is_reusable,
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
    def test_cached_result_requires_current_schedule_and_diagnostic_schema(self) -> None:
        cached = {
            "passed": True,
            "schema_version": "cuda_semantic0031_resident_refill_strict_fp32_v2",
            "collector": {"completed_games": EXPECTED_GAMES, "errors": 0},
            "models": {
                "actor_checkpoint_sha256": POLICY_SHA256,
                "opponent_checkpoint_sha256": POLICY_SHA256,
            },
            "schedule": {"sha256": "current-schedule"},
            "device": {
                "float32_matmul_precision": "highest",
                "matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            },
            "per_game_diagnostics": {
                "schema": "cuda_resident_terminal_diagnostics_v1",
                "terminal_turns": [1] * EXPECTED_GAMES,
                "engine_selections": [1] * EXPECTED_GAMES,
                "terminal_prize_counts": [[0, 0]] * EXPECTED_GAMES,
            },
        }
        self.assertTrue(
            _cached_result_is_reusable(
                cached, schedule_file_sha256="current-schedule"
            )
        )
        self.assertFalse(
            _cached_result_is_reusable(cached, schedule_file_sha256="new-seed-schedule")
        )
        cached["per_game_diagnostics"].pop("terminal_turns")
        self.assertFalse(
            _cached_result_is_reusable(
                cached, schedule_file_sha256="current-schedule"
            )
        )

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
                "per_game_diagnostics": {
                    "schema": "cuda_resident_terminal_diagnostics_v1",
                    "terminal_turns": [5 + offset + index for index in range(games)],
                    "engine_selections": [10 + offset + index for index in range(games)],
                    "terminal_prize_counts": [[2, 4] for _ in range(games)],
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
        self.assertEqual(merged["per_game_diagnostics"]["terminal_turns"], [5, 6, 7, 8])

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

    def test_schedule_changes_with_independent_evaluation_seed(self) -> None:
        entries = (Entry("opponent_a", 1),)
        canonical = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=entries,
            evaluation_seed=341_512_806,
        )
        independent = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=entries,
            evaluation_seed=934_151_280,
        )

        self.assertNotEqual(canonical["schedule_sha256"], independent["schedule_sha256"])
        self.assertTrue(
            {job["engine_seed"] for job in canonical["jobs"]}.isdisjoint(
                {job["engine_seed"] for job in independent["jobs"]}
            )
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
            "passed": True,
            "collector": {"completed_games": 2, "errors": 0},
            "determinism": {"game_results": [1, 1]},
            "per_game_diagnostics": {
                "schema": "cuda_resident_terminal_diagnostics_v1",
                "terminal_turns": [12, 5],
                "engine_selections": [170, 121],
                "terminal_prize_counts": [[0, 3], [4, 1]],
            },
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
        self.assertEqual(records[0]["game_length"], 6)
        self.assertEqual(records[1]["game_length"], 3)
        self.assertEqual(records[0]["prize_differential"], 3)
        self.assertEqual(records[1]["prize_differential"], 3)
        self.assertEqual(records[1]["engine_selections"], 121)
        self.assertFalse(records[0]["error"])
        self.assertFalse(records[0]["continuation_error"])


if __name__ == "__main__":
    unittest.main()
