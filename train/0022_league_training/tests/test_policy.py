from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from ..policy import load_league_actor_critic
from ..policy.league_pool import LeaguePolicyPool
from ..decks import load_deck_plugins
from ..league import DEFAULT_DECK_ROOT


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

    def test_pool_shares_one_encoder_and_isolates_deck_heads(self) -> None:
        plugins = load_deck_plugins(DEFAULT_DECK_ROOT)[:2]
        pool = LeaguePolicyPool.from_foundation(plugins, device=torch.device("cpu"))
        first = pool.policy(plugins[0].deck_id)
        second = pool.policy(plugins[1].deck_id)
        self.assertIs(first.actor.shared_actor, second.actor.shared_actor)
        self.assertIsNot(first.actor.decoder, second.actor.decoder)
        before = second.actor.decoder.weight_ih.detach().clone()
        with torch.no_grad():
            first.actor.decoder.weight_ih.add_(1.0)
        self.assertTrue(torch.equal(second.actor.decoder.weight_ih, before))
        self.assertTrue(all(not parameter.requires_grad for parameter in first.actor.shared_actor.parameters()))
        self.assertGreater(pool.trainable_parameter_count, 0)
        self.assertLess(pool.total_parameter_storage_count, 18_000_000 + 2 * 1_200_000)


if __name__ == "__main__":
    unittest.main()
