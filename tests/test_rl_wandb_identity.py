from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import rl_environment.runs as runs


class WandbRunIdentityTests(unittest.TestCase):
    def test_wandb_run_id_is_stable_safe_and_version_specific(self) -> None:
        first = runs.wandb_run_id(
            "0012-alakazam_sota_feature_engineering",
            "V10_online_ppo",
        )
        repeated = runs.wandb_run_id(
            "0012-alakazam_sota_feature_engineering",
            "V10_online_ppo",
        )
        next_version = runs.wandb_run_id(
            "0012-alakazam_sota_feature_engineering",
            "V11_online_ppo",
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_version)
        self.assertLessEqual(len(first), 64)
        self.assertRegex(first, re.compile(r"^[a-z0-9_-]+$"))

    def test_new_experiment_manifest_declares_safe_tracking_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "rl_runs"
            artifact_root = root / "artifact"
            with (
                patch.object(runs, "RUNS_ROOT", root),
                patch.object(runs, "CHECKPOINT_ROOT", root / "checkpoint"),
            ):
                experiment = runs.initialize_experiment(
                    "wandb-test",
                    objective="Verify tracking defaults",
                    runs_root=artifact_root,
                )
            manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["tracking"]["wandb_mode"], "disabled")
        self.assertEqual(
            manifest["tracking"]["wandb_project"],
            "pokemon-tcg-policy-learning",
        )
        self.assertEqual(manifest["tracking"]["metric_schema"], "ptcg_tracking_v1")
        self.assertNotIn("api_key", json.dumps(manifest).lower())


if __name__ == "__main__":
    unittest.main()
