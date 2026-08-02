from __future__ import annotations

import ast
import importlib
from pathlib import Path
import unittest


BASE = "train.0026_raging_bolt_canonical_decoder_rl"
ROOT = Path("train/0026_raging_bolt_canonical_decoder_rl")


class ProjectContractTests(unittest.TestCase):
    def test_no_other_numbered_project_imports(self) -> None:
        violations = []
        for path in ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.startswith("train.00") and not name.startswith(BASE):
                        violations.append((str(path), name))
        self.assertEqual(violations, [])

    def test_frozen_assets_strict_load(self) -> None:
        focal = importlib.import_module(f"{BASE}.focal")
        opponents = importlib.import_module(f"{BASE}.opponents")
        league = importlib.import_module(f"{BASE}.league")
        self.assertEqual(focal.verify_focal_checkpoint().parameter_count, 21_837_082)
        self.assertEqual(opponents.verify_foundation().parameter_count, 17_756_162)
        self.assertEqual(len(league.load_frozen_catalog()), 51)

    def test_portable_candidate_runtime_is_project_local(self) -> None:
        deployment = ROOT / "focal" / "deployment"
        self.assertTrue((deployment / "canonical_inference.py").is_file())
        self.assertTrue((deployment / "inference.py").is_file())


if __name__ == "__main__":
    unittest.main()
