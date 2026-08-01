from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .. import package_builder
from ..package_builder import _write_portable, build


class PackageBuilderTest(unittest.TestCase):
    def test_export_exposes_shared_gpu_inference_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "strategy").mkdir()
            _write_portable(target)
            portable = (target / "strategy/portable_inference.py").read_text(
                encoding="utf-8"
            )
            self.assertTrue((target / "strategy/inference.py").is_file())
            self.assertIn("self.model = actor.eval()", portable)
            self.assertIn("self.config = actor.config", portable)

    def test_build_routes_to_checkpoint_deck_instead_of_focal_deck(self) -> None:
        checkpoint_payload = {
            "deck_id": "raging_bolt_ex_james_cox_henry_chao_001",
            "update": 75,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "decoder.pt"
            torch.save(checkpoint_payload, checkpoint)
            with patch.object(
                package_builder, "load_deck_plugins", return_value=()
            ), self.assertRaisesRegex(ValueError, "raging_bolt_ex_james_cox_henry_chao_001"):
                build(checkpoint, "unused")


if __name__ == "__main__":
    unittest.main()
