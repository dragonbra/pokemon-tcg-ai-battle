from __future__ import annotations

import ast
from importlib import import_module
import unittest
from pathlib import Path


PROJECT = import_module("train.0021_persona_free_universal_bc")


class ProjectContractTests(unittest.TestCase):
    def test_project_identity(self) -> None:
        self.assertEqual(PROJECT.PROJECT_ID, "0021_persona_free_universal_bc")

    def test_no_numbered_project_runtime_imports(self) -> None:
        root = Path(__file__).parents[1]
        forbidden: list[tuple[Path, str]] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if name.startswith("train.00") and not name.startswith(
                        "train.0021_persona_free_universal_bc"
                    ):
                        forbidden.append((path, name))
        self.assertEqual(forbidden, [])

    def test_source_id_is_metadata_only(self) -> None:
        materialized = import_module(
            "train.0021_persona_free_universal_bc.training.materialized"
        )
        ac_data = import_module(
            "train.0021_persona_free_universal_bc.training.ac_data"
        )
        self.assertIn("source_id", materialized.CACHE_FIELDS)
        self.assertIn("source_id", materialized._INT32_FIELDS)
        self.assertNotIn("source_id", materialized._INT16_FIELDS)
        self.assertNotIn("source_id", ac_data.ACTOR_INPUT_FIELDS)

    def test_persona_modules_do_not_exist(self) -> None:
        root = Path(__file__).parents[1]
        self.assertFalse((root / "source_model.py").exists())
        self.assertFalse((root / "source_r15_model.py").exists())

    def test_every_production_module_imports(self) -> None:
        root = Path(__file__).parents[1]
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.parts:
                continue
            relative = path.relative_to(root.parent.parent)
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            with self.subTest(module=".".join(parts)):
                import_module(".".join(parts))


if __name__ == "__main__":
    unittest.main()
