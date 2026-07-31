from __future__ import annotations

import unittest

from evaluation.metrics.base import GameContext
from evaluation.metrics.league_quality import LeagueQualityPlugin
from evaluation.metrics.profiles import LEAGUE_QUALITY_PROFILE_ID, get_metric_profile
from evaluation.metrics.registry import create_metric_registry
from tests.test_evaluation_metrics import player, pokemon, strategy_step, worker_trace


def _context() -> GameContext:
    return GameContext(
        game_id="dragapult-game-1",
        candidate_name="dragapult_ex_001",
        opponent_name="raging_bolt_ex_mega_kangaskhan_ex_002",
        candidate_physical_index=0,
        candidate_first=True,
    )


def _player(
    *,
    active_card=None,
    bench_cards=None,
    deck_count: int,
    prizes: int,
):
    value = player(
        active_card=active_card,
        bench_cards=bench_cards,
        deck_count=deck_count,
    )
    value["prize"] = [{"id": index + 1} for index in range(prizes)]
    return value


class LeagueQualityMetricTest(unittest.TestCase):
    def test_profile_registers_generic_and_deck_specific_metric(self) -> None:
        profile = get_metric_profile(LEAGUE_QUALITY_PROFILE_ID)
        registry = create_metric_registry([], profile_id=LEAGUE_QUALITY_PROFILE_ID)

        self.assertEqual(profile.revision, 1)
        self.assertEqual(profile.metric_ids[-1], "league_quality")
        self.assertEqual(registry.plugins[-1].metric_id, "league_quality")
        self.assertEqual(
            profile.manifest()["semantic_groups"][-1]["id"],
            "deck_category_focus",
        )

    def test_digest_tracks_setup_attack_prize_resource_and_denial(self) -> None:
        dreepy = pokemon(119, 10)
        dragapult = pokemon(121, 10, energies=[2, 5])
        dragapult["preEvolution"] = [pokemon(120, 10)]
        plugin = LeagueQualityPlugin()
        opening = strategy_step(
            step_index=0,
            turn=1,
            your_index=0,
            first_player=0,
            players=[
                _player(active_card=dreepy, deck_count=40, prizes=6),
                _player(deck_count=40, prizes=6),
            ],
        )
        ability = strategy_step(
            step_index=1,
            turn=1,
            your_index=0,
            first_player=0,
            action=[0],
            options=[{"type": 10, "cardId": 112}],
            players=[
                _player(active_card=dreepy, deck_count=37, prizes=6),
                _player(deck_count=40, prizes=6),
            ],
        )
        opponent_turn = strategy_step(
            step_index=2,
            turn=2,
            your_index=1,
            first_player=0,
            players=[
                _player(active_card=dreepy, deck_count=37, prizes=6),
                _player(deck_count=38, prizes=6),
            ],
        )
        attack = strategy_step(
            step_index=3,
            turn=3,
            your_index=0,
            first_player=0,
            action=[0],
            options=[{"type": 13, "attackId": 999}],
            players=[
                _player(active_card=dragapult, deck_count=35, prizes=6),
                _player(deck_count=38, prizes=6),
            ],
            logs=[{"type": 16, "playerIndex": 1, "serial": 20, "value": -200}],
        )
        attack["observation"]["current"].update(
            {"supporterPlayed": True, "energyAttached": True}
        )
        resolved = strategy_step(
            step_index=4,
            turn=3,
            your_index=0,
            first_player=0,
            players=[
                _player(active_card=dragapult, deck_count=35, prizes=4),
                _player(deck_count=38, prizes=6),
            ],
        )
        denied_opponent_turn = strategy_step(
            step_index=5,
            turn=4,
            your_index=1,
            first_player=0,
            players=[
                _player(active_card=dragapult, deck_count=35, prizes=4),
                _player(deck_count=37, prizes=6),
            ],
        )

        result = plugin.analyze_game(
            worker_trace(
                steps=[opening, ability, opponent_turn, attack, resolved, denied_opponent_turn]
            ),
            _context(),
        )

        self.assertEqual(result.payload["profile"]["id"], "dragapult_spread")
        self.assertEqual(result.payload["first_attack_round"], 2)
        self.assertEqual(result.payload["key_setup_round"], 2)
        self.assertEqual(result.payload["prizes_taken"], 2)
        self.assertEqual(result.payload["prizes_per_attack"], 2.0)
        self.assertEqual(result.payload["multi_prize_turns"], 1)
        self.assertEqual(result.payload["ability_actions"], 1)
        self.assertEqual(result.payload["damage_events"], 1)
        self.assertEqual(result.payload["max_evolved_pokemon"], 1)
        self.assertEqual(result.payload["max_attached_energy"], 2)
        self.assertEqual(result.payload["minimum_deck_count"], 35)
        self.assertEqual(result.payload["opponent_attack_denial_rate"], 1.0)
        self.assertEqual(result.payload["supporter_turn_rate"], 0.5)

    def test_aggregate_preserves_profile_and_per_opponent_payload(self) -> None:
        plugin = LeagueQualityPlugin()
        step = strategy_step(
            step_index=0,
            turn=1,
            your_index=0,
            first_player=0,
            action=[0],
            options=[{"type": 13, "attackId": 999}],
            players=[
                _player(active_card=pokemon(121, 1), deck_count=30, prizes=6),
                _player(deck_count=30, prizes=6),
            ],
        )
        result = plugin.analyze_game(worker_trace(steps=[step]), _context())
        aggregate = plugin.aggregate([result, result])

        self.assertEqual(aggregate.payload["games"], 2)
        self.assertEqual(aggregate.payload["profile"]["id"], "dragapult_spread")
        self.assertEqual(aggregate.payload["attack_turns"], 2)
        self.assertIn("first_attack_round", aggregate.payload["metric_vocabulary"])
        self.assertEqual(
            aggregate.by_opponent["raging_bolt_ex_mega_kangaskhan_ex_002"]["games"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
