from __future__ import annotations

import hashlib
import io
import json
from importlib import import_module
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout


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


if __name__ == "__main__":
    unittest.main()
