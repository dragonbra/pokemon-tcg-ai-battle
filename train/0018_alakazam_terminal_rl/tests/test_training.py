from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ..checkpoint import checkpoint_metadata, save_model_only
from ..compare_frozen_evaluations import (
    compare_binomial,
    compare_reports,
    validate_comparison_contract,
    wilson_interval,
)
from ..export_candidate import write_candidate_model
from ..observability.metrics import OutcomeTracker
from ..rollout.collector import (
    RolloutCollector,
    _coalesce_ready_connections,
    _update_diagnostics,
)
from ..rollout.protocol import Episode, TrajectoryDecision
from ..run import build_parser, main, seed_training_rng
from ..storage import storage_guard
from ..training.batch import prepare_episodes, training_batch_metrics


def _decision(value: float = 0.0) -> TrajectoryDecision:
    return TrajectoryDecision(
        features={"dummy": torch.tensor([[1.0]])},
        action=(0,),
        stopped=False,
        old_log_prob=-0.5,
        old_value=value,
        entropy=0.5,
        turn=1,
    )


def _episode(name: str, reward: float, length: int, first: bool = True) -> Episode:
    return Episode(
        episode_id=name,
        opponent="opponent",
        opponent_hash="hash",
        candidate_first=first,
        seed=1,
        valid=True,
        reward=reward,
        winner=0 if reward > 0 else 1 if reward < 0 else None,
        status="finished",
        error=None,
        engine_selections=10,
        final_turn=4,
        complete_rounds=2,
        decisions=[_decision() for _ in range(length)],
    )


class TrainingContractTest(unittest.TestCase):
    def test_alakazam_diagnostics_use_target_card_ids(self) -> None:
        metrics: dict[str, float] = {}
        _update_diagnostics(
            metrics,
            (
                {"cardId": 743, "attackId": 1},
                {"cardId": 66, "attackId": 0},
                {"cardId": 121, "attackId": 1},
            ),
        )
        self.assertEqual(metrics["selected_card_743"], 1.0)
        self.assertEqual(metrics["selected_card_66"], 1.0)
        self.assertEqual(metrics["alakazam_attack_submissions"], 1.0)
        self.assertNotIn("selected_card_121", metrics)

    def test_coalesce_collects_staggered_worker_requests(self) -> None:
        connections = [object(), object(), object()]
        collector_module = sys.modules[_coalesce_ready_connections.__module__]
        with patch.object(
            collector_module, "wait", return_value=connections[1:]
        ) as mocked_wait:
            ready = _coalesce_ready_connections(
                connections, [connections[0]], window_seconds=1.0  # type: ignore[arg-type]
            )
        self.assertEqual(ready, connections)
        mocked_wait.assert_called_once()

    def test_independent_win_rate_difference_detects_clear_gain(self) -> None:
        result = compare_binomial(100, 1000, 150, 1000)
        self.assertAlmostEqual(result["difference"], 0.05)
        self.assertGreater(result["difference_95"][0], 0.0)
        self.assertLess(result["two_sided_score_p"], 0.01)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(20, 100)
        self.assertLess(low, 0.2)
        self.assertGreater(high, 0.2)

    def test_comparison_contract_rejects_seat_mismatch(self) -> None:
        manifest = {
            "swap_policy": "alternate_candidate_first",
            "engine_runtime": {"cg_tree_hash": "same"},
            "metric_profile": {"id": "core"},
            "worker_cpu_threads": 1,
            "opponents": [{"package_hash": "opponent"}],
        }
        baseline = {
            "manifest": manifest,
            "games": [{"opponent": "opponent", "candidate_first": True}],
        }
        candidate = {
            "manifest": manifest,
            "games": [{"opponent": "opponent", "candidate_first": False}],
        }
        with self.assertRaisesRegex(ValueError, "allocation"):
            validate_comparison_contract(baseline, candidate)

    def test_report_comparison_uses_full_arena_and_separates_seats(self) -> None:
        manifest = {
            "swap_policy": "alternate_candidate_first",
            "engine_runtime": {"cg_tree_hash": "same"},
            "metric_profile": {"id": "core"},
            "worker_cpu_threads": 1,
            "opponents": [
                {"package_hash": "train-hash"},
                {"package_hash": "holdout-hash"},
            ],
        }
        first_opponent = "alakazam_dudunsparce_01"
        second_opponent = "new_opponent"

        def games(winners: tuple[int, int, int, int]) -> list[dict[str, object]]:
            return [
                {
                    "opponent": opponent,
                    "candidate_first": first,
                    "status": "finished",
                    "winner": winner,
                }
                for opponent, first, winner in zip(
                    (first_opponent, first_opponent, second_opponent, second_opponent),
                    (True, False, True, False),
                    winners,
                    strict=True,
                )
            ]

        result = compare_reports(
            {"manifest": manifest, "games": games((1, 1, 1, 1))},
            {"manifest": manifest, "games": games((0, 1, 0, 1))},
        )
        self.assertEqual(result["groups"]["overall"]["candidate"]["wins"], 2)
        self.assertNotIn("holdout_pool", result["groups"])
        self.assertEqual(result["groups"]["candidate_first"]["candidate"]["wins"], 2)
        self.assertFalse(result["paired"])

    def test_rollout_collector_rejects_train_mode_policy(self) -> None:
        model = torch.nn.Linear(2, 1)
        with self.assertRaisesRegex(ValueError, "eval mode"):
            RolloutCollector(  # type: ignore[arg-type]
                model, device=torch.device("cpu"), workers=1
            )

    def test_ppo_parser_exposes_gae_lambda(self) -> None:
        args = build_parser().parse_args(
            [
                "ppo",
                "--version",
                "V_test",
                "--warm-start",
                "checkpoint.pt",
                "--gae-lambda",
                "1.0",
            ]
        )
        self.assertEqual(args.gae_lambda, 1.0)

    def test_training_rng_seeds_torch(self) -> None:
        with patch("torch.manual_seed") as manual_seed, patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.manual_seed_all") as manual_seed_all:
            seed_training_rng(20260729)
        manual_seed.assert_called_once_with(20260729)
        manual_seed_all.assert_called_once_with(20260729)

    def test_ppo_parser_exposes_loss_guards_and_dense_checkpoints(self) -> None:
        args = build_parser().parse_args(
            [
                "ppo",
                "--version",
                "V_test",
                "--warm-start",
                "checkpoint.pt",
                "--entropy-coefficient",
                "0",
                "--reference-kl-coefficient",
                "0.02",
                "--target-behavior-kl",
                "0.01",
                "--checkpoint-every",
                "1",
                "--checkpoint-retention",
                "100",
            ]
        )
        self.assertEqual(args.entropy_coefficient, 0.0)
        self.assertEqual(args.reference_kl_coefficient, 0.02)
        self.assertEqual(args.target_behavior_kl, 0.01)
        self.assertEqual(args.checkpoint_every, 1)
        self.assertEqual(args.checkpoint_retention, 100)

    def test_ppo_rejects_nonpositive_checkpoint_cadence(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint-every must be positive"):
            main(
                [
                    "ppo",
                    "--version",
                    "V_test",
                    "--warm-start",
                    "checkpoint.pt",
                    "--checkpoint-every",
                    "0",
                    "--wandb-mode",
                    "disabled",
                ]
            )

    def test_terminal_returns_and_episode_weights(self) -> None:
        batch = prepare_episodes(
            [_episode("win", 1.0, 2), _episode("loss", -1.0, 4)]
        )
        self.assertEqual(batch.decisions, 6)
        self.assertTrue(torch.equal(batch.terminal_return[:2], torch.ones(2)))
        self.assertTrue(torch.equal(batch.terminal_return[2:], -torch.ones(4)))
        self.assertAlmostEqual(float(batch.episode_weight[:2].sum()), 1.0)
        self.assertAlmostEqual(float(batch.episode_weight[2:].sum()), 1.0)
        self.assertAlmostEqual(float((batch.advantage * batch.episode_weight).sum()), 0.0, places=5)

    def test_training_batch_metrics_are_episode_balanced_and_finite(self) -> None:
        batch = prepare_episodes(
            [_episode("win", 1.0, 2), _episode("loss", -1.0, 8)]
        )
        metrics = training_batch_metrics(batch)
        self.assertAlmostEqual(metrics["ppo/advantage_mean"], 0.0, places=5)
        self.assertAlmostEqual(metrics["ppo/advantage_std"], 1.0, places=5)
        self.assertGreaterEqual(metrics["ppo/return_mean"], -1.0)
        self.assertLessEqual(metrics["ppo/return_mean"], 1.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_outcome_tracker_has_rolling_wilson_and_seats(self) -> None:
        tracker = OutcomeTracker()
        tracker.add(_episode("win", 1.0, 1, True))
        tracker.add(_episode("loss", -1.0, 1, False))
        metrics = tracker.metrics()
        self.assertEqual(metrics["rollout/rolling_100/games"], 2.0)
        self.assertEqual(metrics["rollout/rolling_100/win_rate"], 0.5)
        self.assertLess(metrics["rollout/rolling_100/wilson_low"], 0.5)
        self.assertGreater(metrics["rollout/rolling_100/wilson_high"], 0.5)
        self.assertEqual(metrics["rollout/seat/first/win_rate"], 1.0)

    def test_model_only_checkpoint_excludes_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update-000001.pt"
            model = torch.nn.Linear(2, 1)
            save_model_only(  # type: ignore[arg-type]
                path, model, update=1, metadata={"test": True}
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                payload["schema_version"],
                "0018_alakazam_actor_critic_model_only_v1",
            )
            self.assertFalse(
                {
                    "optimizer",
                    "scheduler",
                    "scaler",
                    "rng_state",
                    "rollout",
                }.intersection(payload)
            )
            self.assertEqual(checkpoint_metadata(path)["update"], 1)

    def test_candidate_export_strips_source_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            output = root / "model.bin"
            torch.save(
                {
                    "epoch": 10,
                    "global_step": 42,
                    "model": {"weight": torch.ones(2)},
                    "metadata": {"model_family": "r15"},
                    "optimizer": {"state": {1: {"momentum": torch.ones(2)}}},
                },
                source,
            )

            audit = write_candidate_model(source, output)
            payload = torch.load(output, map_location="cpu", weights_only=False)

            self.assertEqual(
                set(payload), {"schema_version", "model", "metadata", "source_progress"}
            )
            self.assertEqual(payload["source_progress"], {"epoch": 10, "global_step": 42})
            self.assertNotIn("optimizer", payload)
            self.assertTrue(audit["source_had_optimizer_state"])
            self.assertFalse(audit["packaged_optimizer_state_saved"])

    def test_storage_guard_reports_both_filesystems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = storage_guard(
                Path(directory),
                warning_free_gib=10_000.0,
                hard_stop_free_gib=0.0,
                version_cap_gib=1.0,
            )
            self.assertGreater(snapshot.root_free_gib, 0.0)
            self.assertGreater(snapshot.windows_free_gib, 0.0)
            self.assertTrue(snapshot.warning)


if __name__ == "__main__":
    unittest.main()
