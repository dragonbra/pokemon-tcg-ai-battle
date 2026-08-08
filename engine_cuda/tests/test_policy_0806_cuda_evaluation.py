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

    def test_schedule_is_paired_seeded_and_seat_balanced(self) -> None:
        schedule = build_cuda_schedule(
            focal_deck_id="candidate_a",
            entries=(Entry("opponent_a", 2), Entry("opponent_b", 1)),
            evaluation_seed=341_512_806,
        )

        jobs = schedule["jobs"]
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum(job["focal_first"] for job in jobs), 3)
        for offset in range(0, len(jobs), 2):
            first, second = jobs[offset : offset + 2]
            self.assertTrue(first["focal_first"])
            self.assertFalse(second["focal_first"])
            self.assertEqual(first["opponent_id"], second["opponent_id"])
            self.assertEqual(first["engine_seed"], second["engine_seed"])
            self.assertEqual(first["search_seed"], second["search_seed"])

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


if __name__ == "__main__":
    unittest.main()
