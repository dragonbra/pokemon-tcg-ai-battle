from __future__ import annotations

import ast
from pathlib import Path

import pokemon_tcg_ai


PACKAGE_ROOT = Path(pokemon_tcg_ai.__file__).resolve().parent


def test_public_package_has_no_numbered_training_imports() -> None:
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "train" or name.startswith("train."):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}:{name}")
    assert violations == []


def test_public_and_archive_project_ids_are_explicitly_distinct() -> None:
    assert pokemon_tcg_ai.PROJECT_ID == "pokemon_tcg_ai"
    assert pokemon_tcg_ai.ARCHIVE_PROJECT_ID == "0045_single_deck_expert_minimal_lora"


def test_public_package_has_no_versioned_training_scripts() -> None:
    training = PACKAGE_ROOT / "training"
    assert list(training.glob("run_v*.py")) == []
