from __future__ import annotations

import hashlib
import io
import json
import os
from importlib import import_module
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import torch


BUILDER = import_module("train.0020_pluggable_deck_rl.package_builder")
EXPECTED_SHA256 = "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"


class PackageBuilderTests(unittest.TestCase):
    def test_cli_without_deck_keys_builds_every_candidate(self) -> None:
        with patch.object(BUILDER, "build_candidate") as build, redirect_stdout(io.StringIO()):
            self.assertEqual(BUILDER.main([]), 0)
        self.assertEqual(
            [call.args[0] for call in build.call_args_list],
            list(BUILDER.DECK_SOURCES),
        )

    def test_builds_self_contained_exact_deck_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            BUILDER, "CANDIDATE_ROOT", Path(directory)
        ):
            package = BUILDER.build_candidate("dragapult")
            deck = [
                line
                for line in (package / "deck.csv").read_text().splitlines()
                if line
            ]
            self.assertEqual(len(deck), 60)
            self.assertTrue((package / "cg/libcg.so").is_file())
            self.assertTrue((package / "strategy/inference.py").is_file())
            checkpoint = package / "strategy/model.bin"
            self.assertEqual(
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                EXPECTED_SHA256,
            )
            manifest = json.loads((package / "manifest.json").read_text())
            self.assertEqual(manifest["source_id"], 0)
            self.assertEqual(manifest["training_updates"], 0)

    def test_entrypoint_loads_without_file_like_kaggle(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            BUILDER, "CANDIDATE_ROOT", Path(directory)
        ):
            package = BUILDER.build_candidate("marnie")
            source = (package / "main.py").read_text(encoding="utf-8")
            previous_cwd = Path.cwd()
            try:
                os.chdir(package)
                namespace = {"__name__": "__kaggle_agent__"}
                exec(compile(source, "<kaggle-agent>", "exec"), namespace)
                self.assertEqual(len(namespace["read_deck_csv"]()), 60)
            finally:
                os.chdir(previous_cwd)

    def test_all_package_templates_use_kaggle_safe_root_resolution(self) -> None:
        for template in (BUILDER.PROJECT_ROOT / "package_main.py", BUILDER.RL_PACKAGE_MAIN):
            source = template.read_text(encoding="utf-8")
            self.assertIn("def _agent_directory()", source)
            self.assertIn('Path("/kaggle_simulations/agent")', source)
            self.assertNotIn("ROOT = Path(__file__)", source)

    def test_builds_region_bot_from_audited_kaggle_deck(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            BUILDER, "CANDIDATE_ROOT", Path(directory)
        ):
            package = BUILDER.build_candidate("region_bot")
            deck = [
                int(line)
                for line in (package / "deck.csv").read_text().splitlines()
                if line
            ]
            self.assertEqual(len(deck), 60)
            self.assertEqual(deck.count(63), 2)
            self.assertEqual(deck.count(96), 3)
            self.assertEqual(deck.count(756), 3)
            manifest = json.loads((package / "manifest.json").read_text())
            provenance = manifest["kaggle_provenance"]
            self.assertEqual(provenance["rank"], 1)
            self.assertEqual(provenance["submission_id"], 54954310)
            self.assertEqual(provenance["episode_id"], 88830387)
            self.assertEqual(
                provenance["deck_sha256"],
                "f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a",
            )

    def test_builds_fail_closed_source_98_persona_poc(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            BUILDER, "CANDIDATE_ROOT", Path(directory)
        ):
            package = BUILDER.build_candidate("region_bot", persona_poc=True)
            self.assertEqual(package.name, "0020_persona98_raging_bolt_poc")
            manifest = json.loads((package / "manifest.json").read_text())
            self.assertEqual(manifest["source_id"], 98)
            self.assertTrue(manifest["audit_only"])
            self.assertEqual(
                manifest["persona_team_name"], "James Cox & Henry Chao"
            )
            main_source = (package / "main.py").read_text()
            self.assertIn('MANIFEST["source_id"]', main_source)

    def test_builds_fail_closed_source_226_dragapult_persona_poc(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            BUILDER, "CANDIDATE_ROOT", Path(directory)
        ):
            package = BUILDER.build_candidate("dragapult", persona_poc=True)
            self.assertEqual(package.name, "0020_persona226_dragapult_poc")
            manifest = json.loads((package / "manifest.json").read_text())
            self.assertEqual(manifest["source_id"], 226)
            self.assertTrue(manifest["audit_only"])
            self.assertEqual(manifest["persona_team_name"], "THIRD PTCG Club")
            evidence = manifest["persona_evidence"]
            self.assertEqual(evidence["0019_exact_deck_winning_episodes"], 0)
            self.assertEqual(evidence["0019_closest_deck_shared_card_slots"], 49)

    def test_persona_poc_rejects_unregistered_deck(self) -> None:
        with self.assertRaisesRegex(ValueError, "not registered for alakazam"):
            BUILDER.build_candidate("alakazam", persona_poc=True)

    def test_builds_self_contained_ppo_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            BUILDER, "CANDIDATE_ROOT", Path(directory) / "candidates"
        ):
            checkpoint = Path(directory) / "update-000010.pt"
            torch.save(
                {
                    "schema_version": "0020_dragapult_actor_critic_model_only_v1",
                    "update": 10,
                    "model": {},
                    "metadata": {
                        "version": "V9_dragapult_ppo_initial",
                        "stage": "ppo",
                    },
                },
                checkpoint,
            )
            package = BUILDER.build_rl_candidate(
                "dragapult",
                checkpoint=checkpoint,
                target_name="0020_dragapult_ppo_u0010",
            )
            manifest = json.loads((package / "manifest.json").read_text())
            self.assertEqual(manifest["training_updates"], 10)
            self.assertEqual(manifest["source_id"], 0)
            self.assertFalse(manifest["optimizer_state_saved"])
            self.assertTrue((package / "strategy/portable_inference.py").is_file())
            self.assertEqual(
                manifest["checkpoint_sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )

    def test_rl_candidate_rejects_non_model_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            torch.save(
                {
                    "schema_version": "0020_dragapult_actor_critic_model_only_v1",
                    "update": 10,
                    "model": {},
                    "metadata": {"version": "V9", "stage": "ppo"},
                    "optimizer": {},
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "forbidden resumable state"):
                BUILDER.build_rl_candidate(
                    "dragapult",
                    checkpoint=checkpoint,
                    target_name="invalid",
                )


if __name__ == "__main__":
    unittest.main()
