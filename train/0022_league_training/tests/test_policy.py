from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from ..policy import load_league_actor_critic


class LeaguePolicyTest(unittest.TestCase):
    def test_only_decoder_and_value_are_trainable(self) -> None:
        model, _ = load_league_actor_critic("cpu")
        model.assert_trainable_contract()
        names = model.trainable_parameter_names()
        self.assertTrue(any(name.startswith("value_head.") for name in names))
        self.assertTrue(any(name.startswith("actor.decoder.") for name in names))
        loss = sum(parameter.float().sum() for parameter in model.parameters() if parameter.requires_grad)
        loss.backward()
        self.assertTrue(all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if not name.startswith("value_head.")
            and not any(name.startswith(f"actor.{component}.") for component in ("pointer_key", "pointer_query", "option_bias", "decoder_init", "decoder", "stop"))
        ))

    def test_numbered_project_import_boundary(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    violations.extend(name for name in names if name.startswith("train.00") and not name.startswith("train.0022"))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
