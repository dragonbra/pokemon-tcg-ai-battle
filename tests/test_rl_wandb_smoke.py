from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rl_environment.wandb_smoke import run_fake_training


class WandbSmokeTests(unittest.TestCase):
    def test_disabled_fake_training_writes_deterministic_canonical_curve(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=True):
                summary = run_fake_training(output, steps=3)
            rows = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["steps"], 3)
        self.assertEqual([row["step"] for row in rows], [1, 2, 3])
        self.assertGreater(rows[0]["train/bc_loss"], rows[-1]["train/bc_loss"])
        self.assertEqual(rows[-1]["validation/legal_action_rate"], 1.0)

    def test_fake_training_refuses_to_overwrite_existing_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=True):
                run_fake_training(output, steps=1)
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    run_fake_training(output, steps=1)


if __name__ == "__main__":
    unittest.main()
