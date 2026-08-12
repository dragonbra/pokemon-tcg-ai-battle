from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import unittest


curriculum = importlib.import_module(
    "train.0042_full_model_design.training.run_deck_curriculum"
)


class DeckCurriculumTest(unittest.TestCase):
    def test_sequence_and_derived_archetypes_are_exact(self) -> None:
        self.assertEqual(
            [(row["version"].split("_", 1)[0], row["deck"].rsplit("_", 1)[-1], row["class_id"])
             for row in curriculum.SEGMENTS],
            [("V16", "010", 6), ("V17", "011", 5)],
        )
        for row in curriculum.SEGMENTS:
            deck, manifest = curriculum.validate_deck(row)
            source = curriculum.ROOT / manifest["provenance"]["source_deck"]
            self.assertEqual(deck.read_bytes(), source.read_bytes())
            self.assertEqual(
                hashlib.sha256(deck.read_bytes()).hexdigest(),
                manifest["provenance"]["source_deck_file_sha256"],
            )

    def test_each_command_is_finite_u50_and_preserves_v11_parameters(self) -> None:
        checkpoint = curriculum.ROOT / (
            "rl_runs/0042_full_model_design/versions/previous/checkpoint/update-000050.pt"
        )
        command = curriculum.training_command(curriculum.SEGMENTS[0], checkpoint)
        joined = " ".join(command)
        for token in (
            "--updates 50", "--games-per-update 256", "--cuda-lane-count 256",
            "--ppo-minibatch-size 2048", "--ppo-forward-microbatch-size 1024",
            "--ppo-epochs 3", "--eval-every 10", "--decoder-lr 2e-5",
            "--policy-adapter-lr 4e-5", "--allocation-lr 2e-5",
            "--allow-fp16-deployment-numeric-drift", "--wandb-mode online",
        ):
            self.assertIn(token, joined)
        self.assertNotIn("--resume-update0", command)


if __name__ == "__main__":
    unittest.main()
