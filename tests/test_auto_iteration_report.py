from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.auto_iteration_report import (
    analyze_game_record,
    aggregate_iteration,
    render_iteration_html,
)


ALAKAZAM = 743
ABRA = 741
DUNSPARCE = 305
DUDUNSPARCE = 66
RARE_CANDY = 1079
POKE_PAD = 1152
HILDA = 1225
PSYCHIC_ENERGY = 5
ALAKAZAM_ATTACK = 1072


def pokemon(card_id: int, serial: int, hp: int = 50) -> dict[str, object]:
    return {"id": card_id, "serial": serial, "hp": hp, "maxHp": hp, "energies": [], "energyCards": []}


def state(
    *,
    active: list[dict[str, object]],
    bench: list[dict[str, object]] | None = None,
    hand_ids: list[int] | None = None,
    discard_ids: list[int] | None = None,
    deck_count: int = 30,
    prize_count: int = 6,
) -> dict[str, object]:
    return {
        "active": active,
        "bench": bench or [],
        "hand": [{"id": card_id, "serial": index + 100} for index, card_id in enumerate(hand_ids or [])],
        "discard": [{"id": card_id, "serial": index + 200} for index, card_id in enumerate(discard_ids or [])],
        "deckCount": deck_count,
        "prize": [None] * prize_count,
    }


def step(
    *,
    turn: int,
    role: str,
    player: dict[str, object],
    action: list[int] | None = None,
    options: list[dict[str, object]] | None = None,
    logs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "turn": turn,
        "yourIndex": 0,
        "role": role,
        "action": action or [],
        "select": {"type": 0, "options": options or []},
        "players": [player, {}],
        "observation": {"current": {"players": [player, {}]}, "logs": logs or []},
    }


def attack_option(index: int = 0) -> dict[str, object]:
    return {"index": index, "type": 13, "attackId": ALAKAZAM_ATTACK}


def game(*, swap: bool = False, steps: list[dict[str, object]]) -> dict[str, object]:
    return {
        "opponent": "fixture",
        "game": 1,
        "swap": swap,
        "alakazamPhysicalIndex": 1 if swap else 0,
        "winner": 0,
        "error": "",
        "finished": True,
        "trace": steps,
    }


class AutoIterationReportTests(unittest.TestCase):
    def test_game_analysis_counts_first_turn_standard_attack(self) -> None:
        opening = state(
            active=[pokemon(ABRA, 1)],
            hand_ids=[RARE_CANDY, ALAKAZAM, PSYCHIC_ENERGY],
        )
        second_turn = state(active=[pokemon(ALAKAZAM, 2)])
        result = analyze_game_record(
            game(
                steps=[
                    step(turn=1, role="alakazam_v8_current", player=opening),
                    step(
                        turn=3,
                        role="alakazam_v8_current",
                        player=second_turn,
                        action=[0],
                        options=[attack_option()],
                    ),
                ]
            ),
            agent_label="alakazam_v8_current",
        )

        self.assertTrue(result["turn_order"]["first"])
        self.assertTrue(result["t2"]["attack_success"])
        self.assertTrue(result["t2"]["opening_four_components"]["all_four"])
        self.assertNotIn("opening_four_components_available", result["t2"])
        self.assertNotIn("opening_four_components_attack_completed", result["t2"])

    def test_component_state_accepts_search_cards_and_counts_extra_draws(self) -> None:
        opening = state(
            active=[pokemon(ABRA, 1)],
            hand_ids=[RARE_CANDY, POKE_PAD, HILDA],
        )
        result = analyze_game_record(
            game(
                steps=[
                    step(turn=1, role="alakazam_v8_current", player=opening),
                    step(
                        turn=3,
                        role="alakazam_v8_current",
                        player=state(active=[pokemon(ALAKAZAM, 2)]),
                        logs=[
                            {"type": 2, "playerIndex": 0},
                            {"type": 4, "playerIndex": 0, "cardId": PSYCHIC_ENERGY},
                        ],
                    ),
                    step(
                        turn=3,
                        role="alakazam_v8_current",
                        player=state(active=[pokemon(ALAKAZAM, 2)]),
                        action=[0],
                        options=[attack_option()],
                        logs=[
                            {"type": 4, "playerIndex": 0, "cardId": 741},
                            {"type": 4, "playerIndex": 0, "cardId": 742},
                        ],
                    ),
                ]
            ),
            agent_label="alakazam_v8_current",
        )

        components = result["t2"]["opening_four_components"]
        self.assertTrue(components["alakazam_or_search"])
        self.assertTrue(components["psychic_energy_or_hilda"])
        self.assertTrue(components["all_four"])
        self.assertEqual(result["t2"]["draws_to_second_turn"]["ability_draw_cards"], 2)
        self.assertEqual(result["t2"]["draws_to_second_turn"]["normal_draw_cards"], 1)

        aggregate = aggregate_iteration([result], metadata={"iteration_id": "iteration-001"})
        self.assertEqual(
            aggregate["metrics"]["second_turn_draws"]["all_games"]["average"],
            2.0,
        )
        self.assertEqual(aggregate["metrics"]["second_turn_draws"]["first"]["average"], 2.0)
        self.assertEqual(
            aggregate["metrics"]["second_turn_draws"]["normal_draw_cards"]["all_games"]["total"],
            1,
        )

    def test_aggregate_splits_win_rate_and_second_turn_success_by_turn_order(self) -> None:
        first = game(
            swap=False,
            steps=[step(turn=1, role="alakazam_v8_current", player=state(active=[pokemon(ABRA, 1)]))],
        )
        second = game(
            swap=True,
            steps=[step(turn=2, role="alakazam_v8_current", player=state(active=[pokemon(ABRA, 1)]))],
        )
        second["winner"] = 1

        aggregate = aggregate_iteration(
            [
                analyze_game_record(first, agent_label="alakazam_v8_current"),
                analyze_game_record(second, agent_label="alakazam_v8_current"),
            ],
            metadata={"iteration_id": "iteration-001", "profile": "v8-setup-relay"},
        )

        self.assertEqual(aggregate["sample"]["games"], 2)
        self.assertEqual(aggregate["metrics"]["win_rate"]["first"]["rate"], 1.0)
        self.assertEqual(aggregate["metrics"]["win_rate"]["second"]["rate"], 0.0)
        self.assertEqual(aggregate["metrics"]["t2_alakazam"]["all_games"]["denominator"], 2)

    def test_trace_first_player_overrides_swap_flag_for_turn_order(self) -> None:
        second_turn = step(
            turn=3,
            role="alakazam_v8_current",
            player=state(active=[pokemon(ALAKAZAM, 2)]),
            action=[0],
            options=[attack_option()],
        )
        second_turn["observation"]["current"]["firstPlayer"] = 1
        result = analyze_game_record(
            game(swap=True, steps=[second_turn]),
            agent_label="alakazam_v8_current",
        )

        self.assertTrue(result["turn_order"]["first"])
        self.assertEqual(result["t2"]["engine_turn"], 3)
        self.assertTrue(result["t2"]["attack_success"])

    def test_attack_quality_counts_resolved_attacks_without_prize(self) -> None:
        attack = step(
            turn=3,
            role="alakazam_v8_current",
            player=state(active=[pokemon(ALAKAZAM, 2)], prize_count=6),
            action=[0],
            options=[attack_option()],
        )
        resolved = step(
            turn=3,
            role="alakazam_v8_current",
            player=state(active=[pokemon(ALAKAZAM, 2)], prize_count=6),
            logs=[
                {"type": 15, "playerIndex": 0, "cardId": ALAKAZAM, "attackId": ALAKAZAM_ATTACK},
                {"type": 16, "playerIndex": 1, "cardId": 344, "value": -120},
            ],
        )
        after = step(
            turn=4,
            role="opponent",
            player=state(active=[pokemon(ALAKAZAM, 2)], prize_count=6),
        )
        result = analyze_game_record(
            game(steps=[attack, resolved, after]),
            agent_label="alakazam_v8_current",
        )

        quality = result["attack_quality"]
        self.assertEqual(quality["resolved_attacks"], 1)
        self.assertEqual(quality["non_prize_attacks"], 1)
        self.assertEqual(quality["powerful_hand"]["resolved_attacks"], 1)
        self.assertEqual(quality["powerful_hand"]["non_prize_attacks"], 1)
        self.assertTrue(quality["cases"][0]["damage_observed"])

        prize_after = step(
            turn=4,
            role="opponent",
            player=state(active=[pokemon(ALAKAZAM, 2)], prize_count=5),
        )
        prize_result = analyze_game_record(
            game(steps=[attack, resolved, prize_after]),
            agent_label="alakazam_v8_current",
        )
        self.assertEqual(prize_result["attack_quality"]["non_prize_attacks"], 0)

        aggregate = aggregate_iteration(
            [result, prize_result], metadata={"iteration_id": "iteration-001"}
        )
        self.assertEqual(
            aggregate["metrics"]["non_prize_attacks"]["overall"]["non_prize_attacks"]["rate"],
            0.5,
        )

    def test_relay_classification_marks_visible_discard_route_and_terminal_resource(self) -> None:
        previous = state(active=[pokemon(ALAKAZAM, 7)], prize_count=6)
        recoverable = state(active=[pokemon(ABRA, 9)], discard_ids=[ALAKAZAM], prize_count=5)
        second_previous = state(active=[pokemon(ALAKAZAM, 10)], prize_count=5)
        terminal = state(active=[], discard_ids=[], deck_count=0, prize_count=4)
        result = analyze_game_record(
            game(
                steps=[
                    step(turn=3, role="alakazam_v8_current", player=previous),
                    step(turn=5, role="alakazam_v8_current", player=recoverable),
                    step(turn=7, role="alakazam_v8_current", player=second_previous),
                    step(turn=9, role="alakazam_v8_current", player=terminal),
                ]
            ),
            agent_label="alakazam_v8_current",
        )

        self.assertEqual(result["relay"]["opportunities"], 2)
        self.assertEqual(result["relay"]["failure_counts"]["recoverable_discard_miss"], 1)
        self.assertEqual(result["relay"]["failure_counts"]["terminal_no_resource"], 1)

    def test_rendered_iteration_html_embeds_summary_data(self) -> None:
        document = aggregate_iteration([], metadata={"iteration_id": "iteration-001", "profile": "v8-setup-relay"})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "index.html"
            render_iteration_html(document, output)
            html = output.read_text(encoding="utf-8")

        self.assertIn("iteration-001", html)
        self.assertIn("胜率", html)
        self.assertIn("第二回合平均过牌张数", html)
        self.assertIn("四组件状态（仅状态观测）", html)
        self.assertIn("攻击但未拿奖赏", html)
        self.assertNotIn("起手四组件具备率", html)
        self.assertIn("recoverable_discard_miss", html)
        json.loads(html.split("<script id=\"iteration-data\" type=\"application/json\">")[1].split("</script>", 1)[0])


if __name__ == "__main__":
    unittest.main()
