from __future__ import annotations

import importlib
import unittest


canonical = importlib.import_module(
    "train.0038_action_boundary_rl.semantic_parity.canonical"
)


class SemanticParityCanonicalTest(unittest.TestCase):
    def observation(self) -> dict:
        return {
            "current": {
                "yourIndex": 0,
                "turn": 7,
                "firstPlayer": 1,
                "result": -1,
                "supporterPlayed": True,
                "energyAttached": False,
                "retreated": False,
                "players": [
                    {
                        "active": [{"id": 10, "serial": 3, "hp": 90, "maxHp": 120}],
                        "bench": [{"id": 11, "serial": 9, "hp": 80, "maxHp": 80}],
                        "discard": [{"id": 4, "serial": 12}],
                        "hand": [{"id": 99, "serial": 99}],
                        "deck": [{"id": 98, "serial": 98}],
                        "prize": [{"id": 97, "serial": 97}],
                    },
                    {
                        "active": [{"id": 20, "serial": 55, "hp": 100, "maxHp": 200}],
                        "bench": [{"id": 21, "serial": 61, "hp": 70, "maxHp": 70}],
                        "discard": [],
                        "hand": [{"id": 777, "serial": 777}],
                        "deck": [{"id": 778, "serial": 778}],
                        "prize": [{"id": 779, "serial": 779}],
                    },
                ],
            },
            "select": {
                "type": 0,
                "context": 14,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": 6, "playerIndex": 1, "index": 0, "id": 21},
                    {"type": 6, "playerIndex": 1, "index": 1, "id": 22},
                ],
            },
            "logs": [{"type": 15, "player": 0, "card": 10}],
        }

    def test_public_hash_excludes_hidden_card_identity(self) -> None:
        left = self.observation()
        right = self.observation()
        right["current"]["players"][1]["hand"][0]["id"] = 123456
        right["current"]["players"][1]["deck"][0]["id"] = 123457
        right["current"]["players"][1]["prize"][0]["id"] = 123458
        self.assertEqual(
            canonical.stable_hash(canonical.canonical_public_observation(left)),
            canonical.stable_hash(canonical.canonical_public_observation(right)),
        )

    def test_legal_signature_is_order_independent_but_order_hash_is_not(self) -> None:
        left = self.observation()["select"]
        right = {**left, "option": list(reversed(left["option"]))}
        self.assertEqual(
            canonical.canonical_legal_set(left)["set_hash"],
            canonical.canonical_legal_set(right)["set_hash"],
        )
        self.assertNotEqual(
            canonical.canonical_legal_set(left)["ordered_hash"],
            canonical.canonical_legal_set(right)["ordered_hash"],
        )

    def test_first_divergence_reports_first_stage_only(self) -> None:
        rows = [
            ("observation", "a", "a"),
            ("legal_options", "b", "c"),
            ("logits", "d", "e"),
        ]
        divergence = canonical.first_divergence(rows)
        self.assertEqual(divergence.stage, "legal_options")
        self.assertEqual(divergence.left_hash, "b")
        self.assertEqual(divergence.right_hash, "c")


if __name__ == "__main__":
    unittest.main()
