from __future__ import annotations

import unittest

from ..data.catalog import _model_deck_hash, _outcomes
from ..data.source_index import _deterministic_t4_cap, _source_key


class CatalogOutcomeTest(unittest.TestCase):
    def test_unsettled_rewards_are_unknown(self) -> None:
        self.assertEqual(_outcomes([0, None]), ["unknown", "unknown"])

    def test_unique_finite_winner(self) -> None:
        self.assertEqual(_outcomes([1, 0]), ["win", "loss"])

    def test_equal_finite_rewards_are_draw(self) -> None:
        self.assertEqual(_outcomes([0, 0]), ["draw", "draw"])

    def test_source_key_uses_team_identity_not_storage_directory(self) -> None:
        self.assertEqual(_source_key("LumenLiquidity"), "lumenliquidity")
        self.assertEqual(_source_key("THIRD PTCG Club"), "third_ptcg_club")

    def test_source_key_hashes_non_ascii_identity_stably(self) -> None:
        self.assertEqual(_source_key("ペンギン"), "unicode_675a84845363")
        self.assertEqual(_source_key("ペンギン"), _source_key("ペンギン"))

    def test_observed_unicode_team_names_remain_distinct(self) -> None:
        names = (
            "やる気元気ミワハルキ",
            "カントー地方マスター",
            "ペンギン",
            "今井大登",
            "懒惰的金枪鱼",
            "西松大祐",
            "서주영",
        )
        keys = {_source_key(name) for name in names}
        self.assertEqual(len(keys), len(names))
        self.assertTrue(all(key.startswith("unicode_") for key in keys))

    def test_t4_cap_is_deterministic_and_outcome_proportional(self) -> None:
        rows = [
            {"episode_id": index, "player_index": 0, "outcome": outcome}
            for index, outcome in enumerate(["win"] * 6 + ["loss"] * 3 + ["draw"])
        ]
        first = _deterministic_t4_cap(rows, 5)
        second = _deterministic_t4_cap(list(reversed(rows)), 5)
        self.assertEqual(
            [(row["episode_id"], row["player_index"]) for row in first],
            [(row["episode_id"], row["player_index"]) for row in second],
        )
        self.assertEqual(
            {
                outcome: sum(row["outcome"] == outcome for row in first)
                for outcome in {"win", "loss", "draw"}
            },
            {"win": 3, "loss": 1, "draw": 1},
        )

    def test_model_deck_hash_matches_legacy_canonical_newline(self) -> None:
        self.assertEqual(
            _model_deck_hash([1] * 30 + [2] * 30),
            "2a839860b8e5d6b92bd3fd1e80388f94987814d04cc153728c4e7268adbac7e4",
        )


if __name__ == "__main__":
    unittest.main()
