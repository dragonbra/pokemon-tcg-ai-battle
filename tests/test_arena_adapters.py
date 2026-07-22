from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arena.adapters import reconstruct_notebook_package


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source"
        self.package = self.root / "package"
        self.baseline_cg = self.root / "baseline-cg"
        self.source.mkdir()
        self.baseline_cg.mkdir()
        for filename in ("__init__.py", "api.py", "game.py", "libcg.so"):
            (self.baseline_cg / filename).write_bytes(b"fixture")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reconstructs_literal_deck_and_copies_cg(self) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "my_deck = [7] * 60\n",
                        "def agent(obs):\n",
                        "    return my_deck if obs.get('select') is None else [0]\n",
                    ],
                }
            ]
        }
        (self.source / "agent.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

        result = reconstruct_notebook_package(self.source, self.package, self.baseline_cg)

        self.assertEqual(result.status, "notebook_reconstructed")
        self.assertEqual(len((self.package / "deck.csv").read_text().splitlines()), 60)
        self.assertTrue((self.package / "cg" / "game.py").is_file())
        self.assertIn("def agent", (self.package / "main.py").read_text())

    def test_marks_notebook_without_literal_deck_invalid(self) -> None:
        (self.source / "agent.ipynb").write_text(
            json.dumps({"cells": [{"cell_type": "code", "source": ["def agent(obs): pass\n"]}]}),
            encoding="utf-8",
        )

        result = reconstruct_notebook_package(self.source, self.package, self.baseline_cg)

        self.assertEqual(result.status, "invalid")
        self.assertIn("60", result.error or "")

    def test_reconstructs_static_main_and_deck_files_without_executing_source(self) -> None:
        (self.source / "output").mkdir()
        (self.source / "output" / "main.py").write_text(
            "SIDE_EFFECT = open('must-not-exist', 'w')\n"
            "DECK = [7] * 60\n"
            "def agent(observation):\n"
            "    return DECK\n",
            encoding="utf-8",
        )
        (self.source / "output" / "deck.csv").write_text("\n".join(["7"] * 60) + "\n", encoding="utf-8")

        result = reconstruct_notebook_package(self.source, self.package, self.baseline_cg)

        self.assertEqual(result.status, "source_reconstructed")
        self.assertEqual(result.deck, (7,) * 60)
        self.assertFalse((self.package / "must-not-exist").exists())
        self.assertTrue((self.package / "cg" / "game.py").is_file())


if __name__ == "__main__":
    unittest.main()
