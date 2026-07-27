from __future__ import annotations

from collections import Counter
from importlib import import_module
import json
from pathlib import Path
import unittest


PROJECT = import_module("train.0016_alakazam_multideck_bc")


class ProjectContractTest(unittest.TestCase):
    def test_target_deck_and_manifest_are_exact(self) -> None:
        deck = [
            int(line)
            for line in PROJECT.TARGET_DECK_PATH.read_text().splitlines()
            if line
        ]
        manifest_path = Path(__file__).parents[1] / "deck_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = Counter(
            {int(row["card_id"]): int(row["count"]) for row in manifest["cards"]}
        )
        self.assertEqual(PROJECT.PROJECT_ID, "0016_alakazam_multideck_bc")
        self.assertEqual(len(deck), 60)
        self.assertEqual(Counter(deck), declared)
        self.assertEqual(declared[1264], 4)
        self.assertEqual(declared[1146], 1)

    def test_no_numbered_project_runtime_imports(self) -> None:
        root = Path(__file__).parents[1]
        forbidden = ("train.0014_", "train.0015_", "from ..0014", "from ..0015")
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, str(path))


if __name__ == "__main__":
    unittest.main()
