from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import unittest


curriculum = importlib.import_module(
    "train.0042_full_model_design.training.run_remaining_archetype_curriculum"
)


class RemainingArchetypeCurriculumTest(unittest.TestCase):
    def test_sequence_and_derived_archetypes_are_exact(self) -> None:
        self.assertEqual(
            [
                (row["version"].split("_", 1)[0], row["deck"].rsplit("_", 1)[-1], row["class_id"])
                for row in curriculum.SEGMENTS
            ],
            [
                ("V18", "019", 7),
                ("V19", "013", 11),
                ("V20", "021", 9),
                ("V21", "044", 12),
                ("V22", "048", 13),
            ],
        )
        for row in curriculum.SEGMENTS:
            deck, manifest = curriculum.validate_deck(row)
            source = curriculum.ROOT / manifest["provenance"]["source_deck"]
            self.assertEqual(deck.read_bytes(), source.read_bytes())
            self.assertEqual(
                hashlib.sha256(deck.read_bytes()).hexdigest(),
                manifest["provenance"]["source_deck_file_sha256"],
            )

    def test_each_command_is_finite_u10_and_preserves_parameters(self) -> None:
        checkpoint = curriculum.ROOT / (
            "rl_runs/0042_full_model_design/versions/previous/checkpoint/update-000010.pt"
        )
        command = curriculum.training_command(curriculum.SEGMENTS[0], checkpoint)
        joined = " ".join(command)
        for token in (
            "--updates 10", "--games-per-update 256", "--cuda-lane-count 256",
            "--ppo-minibatch-size 2048", "--ppo-forward-microbatch-size 1024",
            "--ppo-epochs 3", "--eval-every 10", "--decoder-lr 2e-5",
            "--policy-adapter-lr 4e-5", "--allocation-lr 2e-5",
            "--allow-fp16-deployment-numeric-drift", "--wandb-mode online",
        ):
            self.assertIn(token, joined)
        self.assertNotIn("--resume-update0", command)

    def test_initial_predecessor_is_v17_u50(self) -> None:
        self.assertEqual(curriculum.INITIAL_PREDECESSOR, "V17_mega_kangaskhan_ex_crustle_011")
        self.assertEqual(curriculum.INITIAL_PREDECESSOR_UPDATE, 50)


if __name__ == "__main__":
    unittest.main()
