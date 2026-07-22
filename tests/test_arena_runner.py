from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from arena.models import RunRecord
from arena.runner import ArenaRunConfig, ArenaRunner
from arena.rating import RatingState
from arena.scheduler import Pairing
from arena.storage import ArenaStore
from evaluation.packages.loader import SubmissionPackage
from evaluation.runner.models import GameRequest, GameResult
from evaluation.runner.single import run_game
from evaluation.runtime.loader import compute_cg_manifest


class SingleGameBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.request = self._request()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_single_game_delegates_to_isolated_worker_and_keeps_trace(self) -> None:
        result = GameResult(
            self.request.game_id,
            "b",
            True,
            0,
            True,
            0,
            "finished",
            None,
            None,
            3,
            self.root / "temp" / "game-1.json",
        )
        with patch("evaluation.runner.single._run_worker", return_value=result) as worker:
            with patch(
                "evaluation.runner.single._read_or_create_trace",
                return_value=(result, {"trace": [], "result": {"winner": 0}}),
            ):
                actual, trace = run_game(self.request, temp_root=self.root / "temp")
        self.assertEqual(actual.game_id, self.request.game_id)
        self.assertEqual(trace["result"]["winner"], 0)
        worker.assert_called_once()

    def _request(self) -> GameRequest:
        package = SubmissionPackage("a", self.root, [7] * 60, self.root / "main.py", "a", "a", {})
        other = SubmissionPackage("b", self.root, [7] * 60, self.root / "main.py", "b", "b", {})
        return GameRequest("run-1", "game-1", package, other, True, 10, False)


class ArenaRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = ArenaStore(self.root / "arena")
        self.store.initialize()
        self.store.create_run(RunRecord("run-1", "smoke", {"seed": 7}, "now", None, "running"))
        self.packages = {"a": self._package("a"), "b": self._package("b")}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_runner_alternates_seats_and_records_two_rating_events(self) -> None:
        def game_runner(request: GameRequest, temp_root: Path, timeout_seconds: float) -> tuple[GameResult, dict[str, object]]:
            return (
                GameResult(request.game_id, request.opponent, request.candidate_first, 0 if request.candidate_first else 1, True, 0, "finished", None, None, 3, temp_root / "trace.json"),
                {"trace": []},
            )

        runner = ArenaRunner(
            self.store,
            self.packages,
            ArenaRunConfig("run-1", "smoke", games_per_pair=10, seed=7),
            game_runner=game_runner,
        )
        record = runner.run_one(Pairing("a", "b"), 1)

        self.assertTrue(record.player_a_first)
        self.assertEqual(record.winner, "a")
        events = self.store.load_latest_ratings()
        self.assertEqual(len(events), 2)
        self.assertEqual({event["engine"] for event in events}, {"official_gaussian_approx", "elo_compat"})
        self.assertGreater(runner.gaussian_states["a"].mu, 600)

    def test_worker_failure_is_retained_without_rating_change(self) -> None:
        def game_runner(request: GameRequest, temp_root: Path, timeout_seconds: float) -> tuple[GameResult, dict[str, object]]:
            return (
                GameResult(request.game_id, request.opponent, True, 0, False, None, "worker_crash", "worker_crash", "boom", 0, temp_root / "trace.json"),
                {"trace": []},
            )

        runner = ArenaRunner(
            self.store,
            self.packages,
            ArenaRunConfig("run-1", "smoke", seed=7),
            game_runner=game_runner,
        )
        record = runner.run_one(Pairing("a", "b"), 1)

        self.assertEqual(record.result, "error")
        self.assertEqual(runner.gaussian_states["a"].mu, 600)
        self.assertEqual(self.store.load_latest_ratings(), ())

    def test_runner_persists_bounded_trace_metadata_instead_of_full_trace(self) -> None:
        def game_runner(request: GameRequest, temp_root: Path, timeout_seconds: float) -> tuple[GameResult, dict[str, object]]:
            trace_path = temp_root / "trace.json"
            return (
                GameResult(request.game_id, request.opponent, True, 0, True, 0, "finished", None, None, 3, trace_path),
                {
                    "trace": [{"observation": {"large": "x" * 10000}}] * 3,
                    "result": {"winner": 0, "status": "finished"},
                },
            )

        runner = ArenaRunner(
            self.store,
            self.packages,
            ArenaRunConfig("run-1", "smoke", seed=7),
            game_runner=game_runner,
        )
        record = runner.run_one(Pairing("a", "b"), 1)

        self.assertNotIn("trace", record.payload)
        self.assertEqual(record.payload["trace_steps"], 3)
        self.assertLess(len(json.dumps(record.payload)), 2000)

    def test_replaying_same_game_id_does_not_call_worker_or_double_rating(self) -> None:
        calls = 0

        def game_runner(request: GameRequest, temp_root: Path, timeout_seconds: float) -> tuple[GameResult, dict[str, object]]:
            nonlocal calls
            calls += 1
            return (
                GameResult(request.game_id, request.opponent, True, 0, True, 0, "finished", None, None, 3, temp_root / "trace.json"),
                {"trace": []},
            )

        config = ArenaRunConfig("run-1", "smoke", seed=7)
        first = ArenaRunner(self.store, self.packages, config, game_runner=game_runner)
        first.run_one(Pairing("a", "b"), 1)
        second = ArenaRunner(self.store, self.packages, config, game_runner=game_runner)
        replayed = second.run_one(Pairing("a", "b"), 1)

        self.assertEqual(calls, 1)
        self.assertEqual(replayed.game_id, "run-1-smoke-a-b-0001")
        self.assertGreater(second.gaussian_states["a"].mu, 600)
        self.assertEqual(len(self.store.load_games("run-1")), 1)

    def test_continuous_round_fills_requested_number_of_games(self) -> None:
        calls = 0

        def game_runner(request: GameRequest, temp_root: Path, timeout_seconds: float) -> tuple[GameResult, dict[str, object]]:
            nonlocal calls
            calls += 1
            return (
                GameResult(request.game_id, request.opponent, request.candidate_first, 0, True, 0, "finished", None, None, 1, temp_root / "trace.json"),
                {"trace": []},
            )

        runner = ArenaRunner(
            self.store,
            self.packages,
            ArenaRunConfig("run-1", "continuous", seed=7, checkpoint_every=100),
            game_runner=game_runner,
        )
        runner.run_continuous_round(5)

        self.assertEqual(calls, 5)
        self.assertEqual(len(self.store.load_games("run-1")), 5)

    def test_checkpoint_callback_is_called_after_persisting_checkpoint(self) -> None:
        checkpoints: list[int] = []

        def game_runner(request: GameRequest, temp_root: Path, timeout_seconds: float) -> tuple[GameResult, dict[str, object]]:
            return (
                GameResult(request.game_id, request.opponent, True, 0, True, 0, "finished", None, None, 1, temp_root / "trace.json"),
                {"trace": []},
            )

        runner = ArenaRunner(
            self.store,
            self.packages,
            ArenaRunConfig("run-1", "smoke", checkpoint_every=1),
            game_runner=game_runner,
            checkpoint_callback=lambda: checkpoints.append(len(self.store.load_games("run-1"))),
        )
        runner.run_one(Pairing("a", "b"), 1)

        self.assertEqual(checkpoints, [1])

    def test_duplicate_checkpoint_sequence_does_not_count_twice_for_demotion(self) -> None:
        runner = ArenaRunner(
            self.store,
            self.packages,
            ArenaRunConfig("run-1", "smoke"),
            game_runner=lambda request, temp_root, timeout_seconds: (_ for _ in ()).throw(AssertionError()),
        )
        runner.gaussian_states["a"] = RatingState("a", mu=299, games=20, below_threshold_checkpoints=0)

        runner.checkpoint()
        first = runner.gaussian_states["a"]
        runner.checkpoint()
        second = runner.gaussian_states["a"]

        self.assertEqual(first.below_threshold_checkpoints, 1)
        self.assertEqual(first.status, "active")
        self.assertEqual(second.below_threshold_checkpoints, 1)
        self.assertEqual(second.status, "active")

    def _package(self, name: str) -> SubmissionPackage:
        package_root = self.root / name
        cg_root = package_root / "cg"
        cg_root.mkdir(parents=True)
        for filename in ("__init__.py", "api.py", "game.py", "libcg.so"):
            (cg_root / filename).write_bytes(b"fixture")
        return SubmissionPackage(
            name,
            package_root,
            [7] * 60,
            package_root / "main.py",
            f"{name}-package",
            f"{name}-deck",
            compute_cg_manifest(cg_root),
        )


if __name__ == "__main__":
    unittest.main()
