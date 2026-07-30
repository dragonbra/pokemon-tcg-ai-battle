from __future__ import annotations

from importlib import import_module
import unittest


MATERIALIZED = import_module(
    "train.0021_persona_free_universal_bc.training.materialized"
)


def _trajectory(deck: str, episode_id: int, split: str = "train") -> dict:
    return {
        "deck_manifest": {"sha256": deck},
        "episode_id": episode_id,
        "player_index": episode_id % 2,
        "split": split,
    }


class DatasetSplitTests(unittest.TestCase):
    def test_ood_selection_is_deterministic_and_order_independent(self) -> None:
        trajectories = [
            *(_trajectory("deck-a", index) for index in range(3)),
            *(_trajectory("deck-b", 100 + index) for index in range(4)),
            *(_trajectory("deck-c", 200 + index) for index in range(1)),
        ]
        expected = MATERIALIZED.choose_deck_ood_hashes(
            trajectories,
            minimum_episodes=2,
            maximum_episodes=4,
            deck_count=2,
            seed="test-split",
        )
        actual = MATERIALIZED.choose_deck_ood_hashes(
            list(reversed(trajectories)),
            minimum_episodes=2,
            maximum_episodes=4,
            deck_count=2,
            seed="test-split",
        )
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 2)
        self.assertNotIn("deck-c", actual)

    def test_ood_deck_overrides_legacy_episode_split(self) -> None:
        ood = frozenset({"deck-a"})
        train_row = {"deck_manifest": {"sha256": "deck-a"}, "split": "train"}
        validation_row = {
            "deck_manifest": {"sha256": "deck-a"},
            "split": "validation",
        }
        self.assertEqual(
            MATERIALIZED.assign_corrected_split(train_row, ood),
            "validation_deck_ood",
        )
        self.assertEqual(
            MATERIALIZED.assign_corrected_split(validation_row, ood),
            "validation_deck_ood",
        )

    def test_non_ood_legacy_validation_becomes_iid(self) -> None:
        row = {"deck_manifest": {"sha256": "deck-b"}, "split": "validation"}
        self.assertEqual(
            MATERIALIZED.assign_corrected_split(row, frozenset({"deck-a"})),
            "validation_iid",
        )


if __name__ == "__main__":
    unittest.main()
