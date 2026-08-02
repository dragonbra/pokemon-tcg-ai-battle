from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BASE = "train.0026_raging_bolt_canonical_decoder_rl"


class TrainingContractTests(unittest.TestCase):
    def test_used_version_is_rejected(self) -> None:
        run = importlib.import_module(f"{BASE}.training.run")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            used = root / "V1_used" / "artifact"
            used.mkdir(parents=True)
            with patch.object(run, "RUN_ROOT", root):
                with self.assertRaises(FileExistsError):
                    run.assert_fresh_version("V1_used")

    def test_formal_config_rejects_noncanonical_eval_size(self) -> None:
        run = importlib.import_module(f"{BASE}.training.run")
        with self.assertRaises(ValueError):
            run.RunConfig(version="V2_bad", eval_games=100).validate()


if __name__ == "__main__":
    unittest.main()
