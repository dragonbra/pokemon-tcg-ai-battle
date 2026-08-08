from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_no_numbered_training_project_imports(self) -> None:
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
                    if name.startswith("train.00"):
                        violations.append((path, name))
        self.assertEqual(violations, [])

    def test_frozen_prototype_hashes_match_release(self) -> None:
        expected = {
            "official_public_prototypes_v1.json": "5b45041cd14beed8a847f2e99ce955a105fbeb63d09256769d41f8cc88bdbc9d",
            "official_full_engine_prototypes_v2.json": "7228b2612da118e1ec353c7a2e08d4a306de0d4128f0c9e68b4075f2c6b54f73",
        }
        for name, digest in expected.items():
            actual = hashlib.sha256((ROOT / "assets" / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
