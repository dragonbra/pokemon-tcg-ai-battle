from __future__ import annotations

import unittest

from train.kaggle_bc_top20.training.render_top100_research import (
    MANDATORY_TEAM,
    classify_archetype,
    package_name,
    sample_sufficiency,
    select_rosters,
    selection_score,
    wilson_lcb95,
)


def _episode_window(episodes: int, win_rate: float) -> dict[str, int | float]:
    wins = round(episodes * win_rate)
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": episodes - wins,
        "draws": 0,
        "win_rate": wins / episodes,
    }


def _player(rank: int, name: str, episodes: int, win_rate: float) -> dict:
    return {
        "rank": rank,
        "team_name": name,
        "submission_id": 1000 + rank,
        "episode_window": _episode_window(episodes, win_rate),
    }


class Top100ResearchReportTests(unittest.TestCase):
    def test_wilson_lower_bound_penalizes_small_samples(self) -> None:
        small = _episode_window(100, 0.6)
        large = _episode_window(500, 0.6)
        self.assertLess(wilson_lcb95(small), wilson_lcb95(large))

    def test_selection_score_caps_volume_reward_at_500(self) -> None:
        five_hundred = _player(1, "a", 500, 0.6)
        one_thousand = _player(2, "b", 1000, 0.6)
        self.assertEqual(sample_sufficiency(five_hundred["episode_window"]), 1.0)
        self.assertEqual(sample_sufficiency(one_thousand["episode_window"]), 1.0)
        self.assertGreater(selection_score(one_thousand), selection_score(five_hundred))

    def test_roster_forces_lumen_and_enforces_ordinary_200_floor(self) -> None:
        players = [_player(1, MANDATORY_TEAM, 140, 0.68)]
        players.extend(
            _player(rank, f"eligible-{rank}", 200 + rank, 0.55 + rank / 1000)
            for rank in range(2, 23)
        )
        players.append(_player(23, "too-small", 199, 0.99))
        strict, selected = select_rosters(players)
        self.assertEqual(strict, [])
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected[0]["team_name"], MANDATORY_TEAM)
        self.assertNotIn("too-small", {player["team_name"] for player in selected})

    def test_classifies_each_top100_archetype(self) -> None:
        cases = {
            "Alakazam / Dudunsparce": ("Alakazam",),
            "Marnie's Grimmsnarl ex / Froslass": ("Marnie's Grimmsnarl ex",),
            "Team Rocket's Mewtwo ex / Spidops": ("Team Rocket's Mewtwo ex",),
            "Cynthia's Garchomp ex / Roserade": ("Cynthia's Garchomp ex",),
            "Mega Kangaskhan ex / Crustle": ("Mega Kangaskhan ex",),
            "Dragapult ex / Dusknoir": ("Dragapult ex", "Dusknoir"),
            "Dragapult ex / Dudunsparce": ("Dragapult ex", "Dudunsparce"),
            "Dragapult ex": ("Dragapult ex",),
            "Mega Lopunny ex / Mega Froslass ex": ("Mega Lopunny ex",),
            "Archaludon ex / Cinderace": ("Archaludon ex",),
            "Mega Lucario ex / Solrock": ("Mega Lucario ex",),
            "N's Zoroark ex": ("N’s Zoroark ex",),
            "Festival Lead / Dipplin": ("Dipplin", "Thwackey"),
            "Mega Starmie ex / Dusknoir": ("Mega Starmie ex",),
            "Mega Gardevoir ex / toolbox": ("Mega Gardevoir ex",),
            "Hydrapple ex / Ogerpon": ("Hydrapple ex",),
            "Brambleghast / Comfey": ("Brambleghast",),
        }
        for expected, names in cases.items():
            with self.subTest(expected=expected):
                profile = {"pokemon": [{"name": name} for name in names]}
                self.assertEqual(classify_archetype(profile), expected)

    def test_package_name_uses_static_rank_and_core_pokemon(self) -> None:
        player = {"rank": 2}
        self.assertEqual(
            package_name(player, "Dragapult ex / Dusknoir"),
            "rank_002_dragapult_ex_dusknoir_bc",
        )


if __name__ == "__main__":
    unittest.main()
