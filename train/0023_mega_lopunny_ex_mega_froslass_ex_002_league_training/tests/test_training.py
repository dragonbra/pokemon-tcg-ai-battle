from __future__ import annotations

import subprocess
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ..rollout import (
    EpisodeTrajectory,
    LeaguePolicyView,
    RolloutJob,
    TrajectoryDecision,
)
from ..training.batch import _episode_gae, prepare_episodes
from ..cli import build_parser
from ..training.run import LeagueTrainingConfig, StopRequest, compose_training_config, schedule_jobs
from ..training import league_run
from ..training.league_run import (
    resolve_initial_checkpoint_map,
    resolve_project_version_checkpoint_map,
)
from ..decks import load_deck_plugins
from ..league import DEFAULT_DECK_ROOT


def _episode(update: int, reward: float) -> EpisodeTrajectory:
    deck = tuple(range(1, 61))
    job = RolloutJob("g", "dragapult_ex_001", "opponent", LeaguePolicyView.FROZEN, True, 1, update, deck, deck, Path("runtime"))
    episode = EpisodeTrajectory(job)
    episode.decisions = [
        TrajectoryDecision({}, (0,), True, -0.2, 0.1, 0.0, "dragapult_ex_001", update, 1),
        TrajectoryDecision({}, (1,), True, -0.3, 0.2, 0.0, "dragapult_ex_001", update, 1),
    ]
    episode.finish(reward, 4)
    return episode


class LeagueTrainingTest(unittest.TestCase):
    def test_curated_catalog_uses_limitless_mega_lopunny_focal(self) -> None:
        plugins = load_deck_plugins(DEFAULT_DECK_ROOT)
        self.assertEqual(
            [plugin.deck_id for plugin in plugins if plugin.focal],
            ["mega_lopunny_ex_001"],
        )

    def test_initialization_inherits_prior_member_and_foundation_initializes_new_deck(self) -> None:
        plugins = load_deck_plugins(DEFAULT_DECK_ROOT)
        inherited = next(item for item in plugins if item.deck_id == "dragapult_ex_001")
        new = next(
            item for item in plugins
            if item.deck_id == "mega_lopunny_ex_mega_froslass_ex_002"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifact").mkdir()
            (root / "artifact/status.json").write_text(
                json.dumps({"last_complete_update": 81}), encoding="utf-8"
            )
            (root / "artifact/league_catalog.json").write_text(
                json.dumps({"decks": [{
                    "deck_id": inherited.deck_id,
                    "deck_sha256": inherited.deck_sha256,
                }]}),
                encoding="utf-8",
            )
            checkpoint = (
                root / "checkpoint/live" / inherited.deck_id / "update-000081.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            decoder_audit = SimpleNamespace(
                update=81,
                checkpoint_sha256="a" * 64,
                deck_sha256=inherited.deck_sha256,
                foundation_sha256="b" * 64,
            )
            with patch.object(
                league_run,
                "load_decoder_checkpoint",
                return_value=decoder_audit,
            ):
                paths, audit = resolve_initial_checkpoint_map(
                    (inherited, new), previous_root=root
                )
            self.assertEqual(paths[inherited.deck_id], checkpoint)
            self.assertIsNone(paths[new.deck_id])
            self.assertEqual(audit[inherited.deck_id]["source_update"], 81)
            self.assertEqual(audit[new.deck_id]["initialization"], "foundation")

    def test_prior_catalog_member_without_final_checkpoint_is_rejected(self) -> None:
        plugin = next(
            item for item in load_deck_plugins(DEFAULT_DECK_ROOT)
            if item.deck_id == "dragapult_ex_001"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifact").mkdir()
            (root / "artifact/status.json").write_text(
                json.dumps({"last_complete_update": 81}), encoding="utf-8"
            )
            (root / "artifact/league_catalog.json").write_text(
                json.dumps({"decks": [{
                    "deck_id": plugin.deck_id,
                    "deck_sha256": plugin.deck_sha256,
                }]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "lacks update-81"):
                resolve_initial_checkpoint_map((plugin,), previous_root=root)

    def test_existing_0023_version_checkpoint_can_initialize_restart(self) -> None:
        plugin = next(
            item for item in load_deck_plugins(DEFAULT_DECK_ROOT)
            if item.deck_id == "mega_lopunny_ex_mega_froslass_ex_002"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifact").mkdir()
            (root / "artifact/status.json").write_text(
                json.dumps({"checkpoint_update": 3, "state": "failed"}),
                encoding="utf-8",
            )
            (root / "artifact/league_catalog.json").write_text(
                json.dumps({"decks": [{
                    "deck_id": plugin.deck_id,
                    "deck_sha256": plugin.deck_sha256,
                }]}),
                encoding="utf-8",
            )
            checkpoint = (
                root / "checkpoint/live" / plugin.deck_id / "update-000003.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            decoder_audit = SimpleNamespace(
                update=3,
                checkpoint_sha256="c" * 64,
                deck_sha256=plugin.deck_sha256,
                foundation_sha256="d" * 64,
            )
            with patch.object(
                league_run,
                "load_decoder_checkpoint",
                return_value=decoder_audit,
            ):
                paths, audit, metadata = resolve_project_version_checkpoint_map(
                    (plugin,), source_version="V2_source", source_root=root,
                )
            self.assertEqual(paths[plugin.deck_id], checkpoint)
            self.assertEqual(audit[plugin.deck_id]["source_update"], 3)
            self.assertEqual(audit[plugin.deck_id]["source_status_state"], "failed")
            self.assertEqual(metadata["source_version"], "V2_source")

    def test_formal_config_preserves_initialized_decoder_assets(self) -> None:
        initial = {"checkpoints": {"deck": {"decoder_ref": "x"}}, "catalog": {"deck_count": 48}}
        result = compose_training_config(
            initial, LeagueTrainingConfig("V1_test"), identity={"weights_sha256": "a"},
            catalog_count=48, runtime_root=Path("runtime"),
        )
        self.assertEqual(result["checkpoints"], initial["checkpoints"])
        self.assertFalse(result["live_opponent_updates_enabled"])
        self.assertEqual(result["run"]["games_per_update"], 512)
        self.assertEqual(result["run"]["coalesce_ms"], 0.5)
        self.assertEqual(result["schema_version"], "0023_focal_league_ppo_v1")

    def test_cli_propagates_versioned_coalescing_value(self) -> None:
        args = build_parser().parse_args([
            "train", "--version", "V2_test", "--workers", "64",
            "--coalesce-ms", "5",
        ])
        self.assertEqual(args.workers, 64)
        self.assertEqual(args.coalesce_ms, 5.0)

    def test_league_cli_accepts_initial_version(self) -> None:
        args = build_parser().parse_args([
            "train-league", "--version", "V4_restart",
            "--initial-version", "V2_source",
        ])
        self.assertEqual(args.initial_version, "V2_source")

    def test_worker_benchmark_cli_requires_explicit_output(self) -> None:
        args = build_parser().parse_args([
            "benchmark-workers", "--workers", "128", "--games", "256",
            "--output", ".tmp/evaluation/0023_worker_scaling/workers-128.json",
        ])
        self.assertEqual(args.workers, 128)
        self.assertEqual(args.games, 256)
        self.assertEqual(args.coalesce_ms, 5.0)

    def test_engine_worker_import_path_is_torch_light(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", (
                "import importlib, sys; "
                "importlib.import_module('train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training.rollout.worker'); "
                "assert 'torch' not in sys.modules"
            )],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_stop_request_records_signal_without_raising(self) -> None:
        stop = StopRequest()
        with patch("signal.getsignal"), patch("signal.signal"):
            stop.install()
            stop._handle(2, None)
            self.assertTrue(stop.requested)
            self.assertEqual(stop.signal_number, 2)
            stop.restore()

    def test_schedule_balances_seats_and_preserves_policy_update(self) -> None:
        plugins = load_deck_plugins(DEFAULT_DECK_ROOT)
        jobs = schedule_jobs(plugins, count=512, update=9, seed=3, runtime_root=Path("runtime"))
        self.assertEqual(len(jobs), 512)
        self.assertEqual(sum(job.focal_first for job in jobs), 256)
        self.assertEqual({job.source_policy_update for job in jobs}, {9})
        self.assertEqual({job.opponent_view for job in jobs}, {LeaguePolicyView.FROZEN, LeaguePolicyView.LIVE})

    def test_frozen_evaluation_covers_every_deck_and_both_seats(self) -> None:
        plugins = load_deck_plugins(DEFAULT_DECK_ROOT)
        jobs = schedule_jobs(plugins, count=0, update=0, seed=3, runtime_root=Path("runtime"), evaluation=True)
        self.assertEqual(len(jobs), 100)
        self.assertEqual({job.opponent_deck_id for job in jobs}, {item.deck_id for item in plugins})
        self.assertTrue(all(job.opponent_view is LeaguePolicyView.FROZEN for job in jobs))
    def test_terminal_gae_gamma_one(self) -> None:
        advantages, returns = _episode_gae([0.1, 0.2], 1.0, gamma=1.0, gae_lambda=1.0)
        self.assertTrue(torch.allclose(torch.tensor(advantages), torch.tensor([0.9, 0.8])))
        self.assertTrue(torch.allclose(torch.tensor(returns), torch.ones(2)))

    def test_prepare_rejects_mixed_behavior_policy_updates(self) -> None:
        with self.assertRaisesRegex(ValueError, "mixes source_policy_update"):
            prepare_episodes([_episode(0, 1.0), _episode(1, -1.0)], policy_deck_id="dragapult_ex_001")

    def test_prepare_records_single_behavior_policy_update(self) -> None:
        batch = prepare_episodes([_episode(7, 1.0)], policy_deck_id="dragapult_ex_001")
        self.assertEqual(batch.source_policy_update, 7)
        self.assertEqual(batch.decisions, 2)
        self.assertEqual(batch.policy_deck_id, "dragapult_ex_001")

    def test_prepare_filters_exact_policy_and_flips_opponent_reward(self) -> None:
        episode = _episode(7, 1.0)
        episode.decisions.append(
            TrajectoryDecision({}, (2,), True, -0.4, 0.3, 0.0, "alakazam_dudunsparce_001", 4, -1)
        )
        batch = prepare_episodes([episode], policy_deck_id="alakazam_dudunsparce_001")
        self.assertEqual(batch.policy_deck_id, "alakazam_dudunsparce_001")
        self.assertEqual(batch.source_policy_update, 4)
        self.assertEqual(batch.decisions, 1)
        self.assertEqual(float(batch.terminal_return[0]), -1.0)


if __name__ == "__main__":
    unittest.main()
