from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl_environment.runs import VersionPaths

from .. import league
from .test_decks import _write_plugin


def _paths(root: Path, version: str) -> VersionPaths:
    run = root / "rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions" / version
    artifact = run / "artifact"
    checkpoint = run / "checkpoint"
    for path in (artifact, checkpoint, run / "tensorboard", run / "wandb"):
        path.mkdir(parents=True)
    (artifact / "status.json").write_text(
        json.dumps({"state": "allocated", "version": version}), encoding="utf-8"
    )
    return VersionPaths(
        project_id="0023_mega_lopunny_ex_mega_froslass_ex_002_league_training",
        version_name=version,
        project_archive=root / "experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training",
        run_root=run,
        artifact=artifact,
        checkpoints=checkpoint,
        tensorboard=run / "tensorboard",
        wandb=run / "wandb",
        evaluation=root / f"experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/evaluation/{version}.html",
        config=artifact / "training_config.json",
        metrics=artifact / "training_metrics.jsonl",
        summary=artifact / "training_summary.json",
        status=artifact / "status.json",
        checkpoint_selection=artifact / "checkpoint_selection.json",
        model_contract=artifact / "model_contract.json",
        dataset_reference=artifact / "dataset_reference.json",
        metrics_snapshot=artifact / "metrics_snapshot.json",
        wandb_snapshot_manifest=artifact / "wandb_snapshot_manifest.json",
    )


class LeagueVersionTest(unittest.TestCase):
    def test_initialization_requires_a_focal_live_deck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "at least one deck"):
                league.initialize_league_version("V1_test", deck_root=root)
            _write_plugin(root, "frozen_deck", role="frozen", focal=False)
            with self.assertRaisesRegex(ValueError, "focal Live"):
                league.initialize_league_version("V1_test", deck_root=root)

    def test_live_materializes_and_foundation_frozen_remains_a_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deck_root = root / "decks"
            deck_root.mkdir()
            _write_plugin(deck_root, "live_deck", offset=100, role="live", focal=True)
            _write_plugin(deck_root, "frozen_deck", role="frozen", focal=False)
            paths = _paths(root, "V1_test")
            with (
                mock.patch.object(league, "REPOSITORY_ROOT", root),
                mock.patch.object(league, "allocate_version", return_value=paths),
            ):
                result = league.initialize_league_version("V1_test", deck_root=deck_root)
                self.assertEqual(result, paths)
                live = paths.checkpoints / "decks/live_deck.pt"
                frozen = paths.checkpoints / "decks/frozen_deck.pt"
                self.assertTrue(live.is_file())
                self.assertTrue(live.with_suffix(".pt.sha256").is_file())
                self.assertFalse(frozen.exists())
                with mock.patch.object(league, "project_version_paths", return_value=paths):
                    audit = league.audit_league_version("V1_test")
                self.assertTrue(audit["valid"])
                self.assertEqual(audit["decks"]["frozen_deck"]["sentinel"], "foundation")

    def test_reused_version_refusal_is_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            deck_root = Path(temporary)
            _write_plugin(deck_root, "live_deck", role="live", focal=True)
            with mock.patch.object(
                league,
                "allocate_version",
                side_effect=FileExistsError("version assets already exist"),
            ):
                with self.assertRaisesRegex(FileExistsError, "already exist"):
                    league.initialize_league_version("V1_test", deck_root=deck_root)


if __name__ == "__main__":
    unittest.main()
