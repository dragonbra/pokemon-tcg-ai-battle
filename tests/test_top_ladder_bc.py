from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from train.kaggle_bc_top20.training.prepare_top_ladder_bc import (
    collect,
    deck_profile,
    episode_outcome,
    load_card_catalog,
    source_slug,
    summarize_known_deck_matchups,
    summarize_outcomes,
)


def _episode(episode_id: int, own_reward: int, opponent_reward: int, opponent_id: int) -> dict:
    return {
        "episode_id": episode_id,
        "agents": [
            {"submission_id": 10, "reward": own_reward},
            {"submission_id": opponent_id, "reward": opponent_reward},
        ],
    }


class TopLadderBCTests(unittest.TestCase):
    def test_collect_does_not_overwrite_frozen_campaign(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                collect(raw, root / "dataset")

    def test_source_slug_has_unicode_fallback(self) -> None:
        self.assertEqual(source_slug("Dries @ Tufa Labs", 5), "dries-tufa-labs")
        self.assertEqual(source_slug("懒惰的金枪鱼", 9), "team-9")

    def test_metadata_rewards_produce_windowed_win_rate(self) -> None:
        episodes = [
            _episode(1, 1, -1, 20),
            _episode(2, -1, 1, 21),
            _episode(3, 0, 0, 22),
        ]
        self.assertEqual(episode_outcome(episodes[0], 10), "win")
        summary = summarize_outcomes(episodes, 10, api_limit=3)
        self.assertEqual((summary["wins"], summary["losses"], summary["draws"]), (1, 1, 1))
        self.assertAlmostEqual(summary["win_rate"], 1 / 3)
        self.assertTrue(summary["window_censored"])

    def test_known_deck_matchups_report_partial_coverage(self) -> None:
        episodes = [_episode(1, 1, -1, 20), _episode(2, -1, 1, 21)]
        result = summarize_known_deck_matchups(
            episodes,
            10,
            {
                20: {
                    "deck_sha256": "abc",
                    "pokemon_summary": "Alakazam x2",
                }
            },
        )
        self.assertEqual(result["known_episode_count"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["by_opponent_deck"][0]["wins"], 1)

    def test_deck_profile_uses_official_card_names(self) -> None:
        with TemporaryDirectory() as directory:
            card_data = Path(directory) / "cards.csv"
            card_data.write_text(
                "Card ID,Card Name,Stage (Pokémon)/Type (Energy and Trainer),Previous stage,HP\n"
                "1,Basic Energy,Basic Energy,n/a,n/a\n"
                "2,Abra,Basic,n/a,60\n",
                encoding="utf-8",
            )
            catalog = load_card_catalog(card_data)
        profile = deck_profile([1] * 56 + [2] * 4, catalog)
        self.assertEqual(profile["card_count"], 60)
        self.assertEqual(profile["pokemon_summary"], "Abra x4")


if __name__ == "__main__":
    unittest.main()
