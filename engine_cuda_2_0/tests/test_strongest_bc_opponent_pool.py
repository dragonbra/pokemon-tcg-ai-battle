from __future__ import annotations

import json
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]


class StrongestBCOpponentPoolTest(unittest.TestCase):
    def test_pool_contains_only_the_six_requested_bc_archetypes(self) -> None:
        payload = json.loads(
            (CUDA_ENGINE_ROOT / "configs" / "strongest_bc_opponent_pool_v1.json").read_text(
                encoding="utf-8"
            )
        )
        opponents = payload["opponents"]
        self.assertEqual(len(opponents), 6)
        self.assertEqual(
            {row["archetype"] for row in opponents},
            {
                "lucario",
                "cynthia",
                "kangaskhan_crustle",
                "marnie",
                "alakazam",
                "dragapult",
            },
        )
        self.assertTrue(all("rule" not in row["kind"] for row in opponents))
        self.assertTrue(all("search" not in row["kind"] for row in opponents))

    def test_ppo_smoke_uses_the_same_archetypes(self) -> None:
        pool = json.loads(
            (CUDA_ENGINE_ROOT / "configs" / "strongest_bc_opponent_pool_v1.json").read_text(
                encoding="utf-8"
            )
        )
        smoke = json.loads(
            (CUDA_ENGINE_ROOT / "configs" / "pure_lucario_bc_pool_ppo_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [row["archetype"] for row in smoke["opponents"]],
            [row["archetype"] for row in pool["opponents"]],
        )
        self.assertIn("agent_pure_lucario_v1_bc512_e4_probe", smoke["teacher"])


if __name__ == "__main__":
    unittest.main()
