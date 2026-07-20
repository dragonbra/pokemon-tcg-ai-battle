import json
import tempfile
import unittest
from pathlib import Path

from scripts.alakazam_auto_iter import (
    EvaluationMetrics,
    analyze_records,
    build_replay_command,
    compare_reports,
    decide_promotion,
    _has_direct_handoff_option,
    write_analysis,
)


LABEL = "alakazam_v7"
PSYCHIC_ENERGY = 5
ABRA = 741
DUDUNSPARCE = 66
POWERFUL_HAND = 1072
KADABRA = 742
ALAKAZAM = 743
ENRICHING_ENERGY = 13
BASIC_PSYCHIC = 5
TELEPATH_ENERGY = 19
LANAS_AID = 1184
POFFIN = 1086
WONDROUS_PATCH = 1146
NIGHT_STRETCHER = 1097
POKE_PAD = 1152


def _option(option_type, *, card_id=None, attack_id=None):
    return {
        "index": 0,
        "type": option_type,
        "cardId": card_id,
        "attackId": attack_id,
    }


def _step(
    *,
    turn,
    role=LABEL,
    action=None,
    options=None,
    players=None,
    observation=None,
):
    step = {
        "turn": turn,
        "yourIndex": 0,
        "result": -1,
        "role": role,
        "action": action or [],
        "select": {"type": 0, "options": options or []},
        "players": players or [],
    }
    if observation is not None:
        step["observation"] = observation
    return step


def _game(*, game=1, steps, winner=0):
    return {
        "finished": True,
        "winner": winner,
        "error": "",
        "steps": len(steps),
        "swap": False,
        "alakazamPhysicalIndex": 0,
        "traceMode": "full",
        "opponent": "fixture_opponent",
        "game": game,
        "trace": steps,
    }


def _agent_step(turn, *, attack_id=None, options=None):
    if attack_id is not None:
        options = [_option(13, attack_id=attack_id)]
        action = [0]
    else:
        action = [0] if options else []
    return _step(turn=turn, action=action, options=options)


def _state(*, active=None, bench=None, hand_count=7, deck_count=18, prizes=3):
    return {
        "active": [] if active is None else [active],
        "bench": bench or [],
        "handCount": hand_count,
        "deckCount": deck_count,
        "prize": [{} for _ in range(prizes)],
        "discard": [],
    }


def _raw_post_ko_step(*, game, ready_count):
    bench = [
        {"id": KADABRA, "energies": [PSYCHIC_ENERGY], "serial": 30 + game}
        for _ in range(ready_count)
    ]
    current = {
        "turn": 5,
        "yourIndex": 0,
        "firstPlayer": 0,
        "players": [
            _state(active=None, bench=bench),
            _state(active={"id": 900, "energies": []}, bench=[]),
        ],
    }
    observation = {
        "current": current,
        "select": {"type": 0, "option": []},
        "logs": [
            {
                "type": 16,
                "playerIndex": 0,
                "cardId": ALAKAZAM,
                "serial": 10 + game,
                "value": -10,
                "putDamageCounter": True,
            },
            {
                "type": 6,
                "playerIndex": 0,
                "cardId": ALAKAZAM,
                "serial": 10 + game,
                "fromArea": 4,
                "toArea": 3,
            }
        ],
    }
    return _step(
        turn=5,
        observation=observation,
        players=[current["players"][0], current["players"][1]],
    )


class TestTraceMetrics(unittest.TestCase):
    def test_opponent_side_evaluator_error_is_not_counted_as_agent_error(self):
        record = _game(steps=[_step(turn=4, role="opponent")])
        record["error"] = "IndexError"

        result = analyze_records([record])

        self.assertEqual(result.metrics.errors, 0)

    def test_counts_powerful_hand_only_on_agent_turn_three_or_four(self):
        result = analyze_records(
            [
                _game(
                    steps=[
                        _agent_step(3, attack_id=1072),
                        _step(turn=3, role="opponent"),
                    ]
                ),
                _game(
                    game=2,
                    steps=[_agent_step(4, attack_id=1072)],
                ),
            ]
        )

        self.assertEqual(result.metrics.games, 2)
        self.assertEqual(result.metrics.second_turn_powerful_hand_games, 2)
        self.assertEqual(result.metrics.second_turn_powerful_hand_rate, 1.0)

    def test_mismatched_explicit_label_falls_back_to_trace_agent_role(self):
        result = analyze_records(
            [_game(steps=[_agent_step(3, attack_id=POWERFUL_HAND)])],
            agent_label="stale_submission_label",
        )

        self.assertEqual(result.metrics.second_turn_powerful_hand_games, 1)

    def test_does_not_count_powerful_hand_on_later_turn(self):
        result = analyze_records(
            [_game(steps=[_agent_step(5, attack_id=1072)])]
        )

        self.assertEqual(result.metrics.second_turn_powerful_hand_games, 0)
        self.assertEqual(
            [case.failure_class for case in result.cases],
            ["second_turn_powerful_hand_unavailable"],
        )

    def test_legal_powerful_hand_not_selected_creates_missing_case(self):
        options = [
            {"index": 0, "type": 13, "attackId": POWERFUL_HAND},
            {"index": 1, "type": 14},
        ]
        result = analyze_records(
            [_game(steps=[_step(turn=3, action=[1], options=options)])]
        )

        self.assertEqual(result.metrics.second_turn_powerful_hand_games, 0)
        self.assertEqual(
            [case.failure_class for case in result.cases],
            ["second_turn_powerful_hand_missing"],
        )

    def test_second_turn_without_powerful_hand_is_unavailable(self):
        result = analyze_records(
            [_game(steps=[_agent_step(3, options=[_option(14)])])]
        )

        self.assertEqual(result.metrics.second_turn_powerful_hand_games, 0)
        self.assertEqual(
            [case.failure_class for case in result.cases],
            ["second_turn_powerful_hand_unavailable"],
        )

    def test_counts_only_selected_attack_option(self):
        options = [_option(13, attack_id=1071), _option(13, attack_id=1072)]
        result = analyze_records(
            [
                _game(
                    steps=[
                        _step(turn=3, action=[0], options=options),
                    ]
                )
            ]
        )

        self.assertEqual(result.metrics.second_turn_powerful_hand_games, 0)

class TestCaseReports(unittest.TestCase):
    def test_direct_bench_energy_is_not_a_bench_insurance_miss(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[{"id": ALAKAZAM, "energies": [], "serial": 2}],
            hand_count=1,
            prizes=3,
        )
        player["hand"] = [{"id": BASIC_PSYCHIC, "serial": 10}]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200, "maxHp": 200})],
        }
        options = [
            {
                "index": 0,
                "type": 8,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        self.assertTrue(_has_direct_handoff_option(_step(
            turn=7, action=[0], options=options, observation=observation
        ), 0))
        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[0], options=options, observation=observation)])]
        )

        self.assertNotIn("bench_insurance_missed", {case.failure_class for case in result.cases})

    def test_selected_lanas_aid_counts_as_bench_handoff_progress(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[{"id": ABRA, "energies": [], "serial": 2}],
            hand_count=2,
            prizes=3,
        )
        player["hand"] = [
            {"id": LANAS_AID, "serial": 10},
            {"id": POFFIN, "serial": 11},
        ]
        player["discard"] = [
            {"id": ABRA, "serial": 12},
            {"id": BASIC_PSYCHIC, "serial": 13},
        ]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200, "maxHp": 200})],
        }
        options = [
            {"index": 0, "type": 7, "indexInArea": 0},
            {"index": 1, "type": 7, "indexInArea": 1},
            {"index": 2, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }
        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[0], options=options, observation=observation)])]
        )

        insurance_cases = [
            case for case in result.cases if case.failure_class == "bench_insurance_missed"
        ]
        self.assertEqual(len(insurance_cases), 1)
        self.assertEqual(insurance_cases[0].case_status, "pass")

    def test_selected_lanas_aid_recovery_counts_with_only_dudunsparce_on_bench(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[{"id": DUDUNSPARCE, "energies": [], "serial": 2}],
            hand_count=1,
            prizes=3,
        )
        player["hand"] = [{"id": LANAS_AID, "serial": 10}]
        player["discard"] = [
            {"id": ABRA, "serial": 11},
            {"id": BASIC_PSYCHIC, "serial": 12},
        ]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200, "maxHp": 200})],
        }
        options = [
            {"index": 0, "type": 7, "cardId": LANAS_AID},
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }
        result = analyze_records(
            [
                _game(
                    steps=[
                        _step(
                            turn=7,
                            action=[0],
                            options=options,
                            observation=observation,
                        )
                    ]
                )
            ]
        )

        insurance_cases = [
            case for case in result.cases if case.failure_class == "bench_insurance_missed"
        ]
        self.assertEqual(len(insurance_cases), 1)
        self.assertEqual(insurance_cases[0].case_status, "pass")

    def test_selected_wondrous_patch_counts_as_bench_anchor_progress(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[{"id": KADABRA, "energies": [], "serial": 2}],
            hand_count=1,
            prizes=3,
        )
        player["hand"] = [{"id": WONDROUS_PATCH, "serial": 10}]
        player["discard"] = [{"id": BASIC_PSYCHIC, "serial": 11}]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200, "maxHp": 200})],
        }
        options = [
            {"index": 0, "type": 7, "indexInArea": 0},
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[0], options=options, observation=observation)])]
        )

        insurance_cases = [
            case for case in result.cases if case.failure_class == "bench_insurance_missed"
        ]
        self.assertEqual(len(insurance_cases), 1)
        self.assertEqual(insurance_cases[0].case_status, "pass")

    def test_wondrous_patch_on_unrouted_abra_is_not_bench_anchor_progress(self):
        """Patch energy alone cannot create a usable Abra-line handoff."""
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[{"id": ABRA, "energies": [], "appearThisTurn": True, "serial": 2}],
            hand_count=1,
            prizes=3,
        )
        player["hand"] = [{"id": WONDROUS_PATCH, "serial": 10}]
        player["discard"] = [{"id": BASIC_PSYCHIC, "serial": 11}]
        current = {
            "turn": 8,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200})],
        }
        options = [
            {"index": 0, "type": 7, "indexInArea": 0},
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=8, action=[1], options=options, observation=observation)])]
        )

        insurance_cases = [
            case for case in result.cases if case.failure_class == "bench_insurance_missed"
        ]
        self.assertEqual(insurance_cases, [])

    def test_selected_enriching_energy_to_bench_dunsparce_is_not_insurance_miss(self):
        """The agreed Dudunsparce draw route is valid Bench preparation progress."""
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[
                {"id": ABRA, "energies": [], "serial": 2},
                {"id": DUDUNSPARCE, "energies": [], "serial": 3},
            ],
            hand_count=2,
            prizes=3,
        )
        player["hand"] = [
            {"id": ENRICHING_ENERGY, "serial": 10},
            {"id": POFFIN, "serial": 11},
        ]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200})],
        }
        options = [
            {
                "index": 0,
                "type": 8,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {"index": 1, "type": 7, "indexInArea": 1},
            {"index": 2, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[0], options=options, observation=observation)])]
        )

        insurance_cases = [
            case for case in result.cases if case.failure_class == "bench_insurance_missed"
        ]
        self.assertEqual(len(insurance_cases), 1)
        self.assertEqual(insurance_cases[0].case_status, "pass")

    def test_selected_night_stretcher_abra_recovery_is_not_insurance_miss(self):
        """Night Stretcher followed by Psychic recovery is a valid handoff chain."""
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[],
            hand_count=3,
            prizes=3,
        )
        player["hand"] = [
            {"id": NIGHT_STRETCHER, "serial": 10},
            {"id": TELEPATH_ENERGY, "serial": 11},
            {"id": 305, "serial": 12},
        ]
        player["discard"] = [{"id": ABRA, "serial": 13}]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200})],
        }
        options = [
            {"index": 0, "type": 7, "indexInArea": 0},
            {"index": 1, "type": 7, "indexInArea": 2},
            {"index": 2, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[0], options=options, observation=observation)])]
        )

        insurance_cases = [
            case for case in result.cases if case.failure_class == "bench_insurance_missed"
        ]
        self.assertEqual(len(insurance_cases), 1)
        self.assertEqual(insurance_cases[0].case_status, "pass")

    def test_night_stretcher_is_not_insurance_route_when_attack_line_is_on_bench(self):
        """Night Stretcher should not override an existing Abra-line Bench."""
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[{"id": KADABRA, "energies": [], "serial": 2}],
            hand_count=3,
            prizes=3,
        )
        player["hand"] = [
            {"id": NIGHT_STRETCHER, "serial": 10},
            {"id": TELEPATH_ENERGY, "serial": 11},
            {"id": 305, "serial": 12},
        ]
        player["discard"] = [{"id": ABRA, "serial": 13}]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200})],
        }
        options = [
            {"index": 0, "type": 7, "indexInArea": 0},
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[1], options=options, observation=observation)])]
        )

        self.assertNotIn("bench_insurance_missed", {case.failure_class for case in result.cases})

    def test_empty_bench_alakazam_poffin_case_detects_attack_instead(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[],
            hand_count=3,
            prizes=3,
        )
        player["hand"] = [
            {"id": POFFIN, "serial": 10},
            {"id": ABRA, "serial": 11},
            {"id": 305, "serial": 12},
        ]
        current = {
            "turn": 3,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [
                player,
                _state(active={"id": 900, "hp": 40, "maxHp": 40}),
            ],
        }
        options = [
            {"index": 0, "type": 7},
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [
                _game(
                    steps=[
                        _step(
                            turn=3,
                            action=[1],
                            options=options,
                            observation=observation,
                        )
                    ]
                )
            ]
        )

        self.assertEqual(
            [case.failure_class for case in result.cases],
            ["bench_insurance_missed"],
        )
        self.assertEqual(result.cases[0].case_status, "fail")

    def test_bench_continuity_case_does_not_require_basic_in_hand(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[{"id": 305, "energies": [], "serial": 2}],
            hand_count=3,
            prizes=3,
        )
        player["hand"] = [
            {"id": POFFIN, "serial": 10},
            {"id": 900, "serial": 11},
            {"id": 901, "serial": 12},
        ]
        current = {
            "turn": 3,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [
                player,
                _state(active={"id": 900, "hp": 40, "maxHp": 40}),
            ],
        }
        options = [
            {"index": 0, "type": 7, "cardId": POFFIN},
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [
                _game(
                    steps=[
                        _step(
                            turn=3,
                            action=[1],
                            options=options,
                            observation=observation,
                        )
                    ]
                )
            ]
        )

        self.assertEqual(
            [case.failure_class for case in result.cases],
            ["bench_insurance_missed"],
        )
        self.assertEqual(result.cases[0].case_status, "fail")

    def test_existing_abra_line_is_continuity_even_when_unenergized(self):
        """An established Bench Abra is not the same as an empty Bench."""
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[
                {
                    "id": ABRA,
                    "energies": [],
                    "serial": 2,
                    "appearThisTurn": False,
                }
            ],
            hand_count=2,
            prizes=3,
        )
        player["hand"] = [
            {"id": POFFIN, "serial": 10},
            {"id": 305, "serial": 11},
        ]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200})],
        }
        options = [
            {"index": 0, "type": 7, "indexInArea": 0},
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[1], options=options, observation=observation)])]
        )

        self.assertNotIn("bench_insurance_missed", {case.failure_class for case in result.cases})

    def test_fresh_abra_on_bench_is_continuity(self):
        """A newly placed Abra still prevents the empty-Bench failure state."""
        player = _state(
            active={"id": KADABRA, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[
                {"id": DUDUNSPARCE, "energies": [], "serial": 2},
                {"id": DUDUNSPARCE, "energies": [], "serial": 3},
                {
                    "id": ABRA,
                    "energies": [],
                    "serial": 4,
                    "appearThisTurn": True,
                },
            ],
            hand_count=10,
            prizes=3,
        )
        player["hand"] = [{"id": POKE_PAD, "serial": 10}]
        current = {
            "turn": 10,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900, "hp": 200})],
        }
        options = [
            {"index": 0, "type": 7, "cardId": POKE_PAD},
            {"index": 1, "type": 13, "attackId": 1071},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=10, action=[0], options=options, observation=observation)])]
        )

        self.assertNotIn("bench_insurance_missed", {case.failure_class for case in result.cases})

    def test_later_same_turn_bench_anchor_prevents_insurance_miss(self):
        """A later Patch in the same turn validates an earlier search action."""
        def make_player(hand):
            player = _state(
                active={"id": KADABRA, "energies": [PSYCHIC_ENERGY], "serial": 1},
                bench=[
                    {"id": DUDUNSPARCE, "energies": [], "serial": 2},
                    {"id": DUDUNSPARCE, "energies": [], "serial": 3},
                    {
                        "id": ABRA,
                        "energies": [],
                        "serial": 4,
                        "appearThisTurn": True,
                    },
                ],
                hand_count=10,
                prizes=3,
            )
            player["hand"] = hand
            player["discard"] = [{"id": BASIC_PSYCHIC, "serial": 20}]
            return player

        first_player = make_player(
            [
                {"id": POKE_PAD, "serial": 10},
                {"id": WONDROUS_PATCH, "serial": 11},
            ]
        )
        first_options = [
            {"index": 0, "type": 7, "cardId": POKE_PAD},
            {"index": 1, "type": 7, "cardId": WONDROUS_PATCH},
            {"index": 2, "type": 13, "attackId": 1071},
        ]
        first_observation = {
            "current": {
                "turn": 10,
                "yourIndex": 0,
                "firstPlayer": 0,
                "players": [first_player, _state(active={"id": 900, "hp": 200})],
            },
            "select": {"type": 0, "option": first_options},
            "logs": [],
        }

        second_player = make_player([{"id": WONDROUS_PATCH, "serial": 11}])
        second_options = [
            {"index": 0, "type": 7, "cardId": WONDROUS_PATCH},
            {"index": 1, "type": 13, "attackId": 1071},
        ]
        second_observation = {
            "current": {
                "turn": 10,
                "yourIndex": 0,
                "firstPlayer": 0,
                "players": [second_player, _state(active={"id": 900, "hp": 200})],
            },
            "select": {"type": 0, "option": second_options},
            "logs": [],
        }

        result = analyze_records(
            [
                _game(
                    steps=[
                        _step(
                            turn=10,
                            action=[0],
                            options=first_options,
                            observation=first_observation,
                        ),
                        _step(
                            turn=10,
                            action=[0],
                            options=second_options,
                            observation=second_observation,
                        ),
                    ]
                )
            ]
        )

        self.assertFalse(
            any(
                case.failure_class == "bench_insurance_missed"
                and case.case_status == "fail"
                for case in result.cases
            )
        )

    def test_missed_handoff_preparation_creates_a_target_case(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[
                {"id": KADABRA, "energies": [], "serial": 2, "appearThisTurn": False},
                {"id": DUDUNSPARCE, "energies": [], "serial": 3},
            ],
            hand_count=3,
        )
        player["hand"] = [
            {"id": BASIC_PSYCHIC, "serial": 10},
            {"id": ENRICHING_ENERGY, "serial": 11},
        ]
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, _state(active={"id": 900})],
        }
        options = [
            {
                "index": 0,
                "type": 8,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "area": 2,
                "indexInArea": 1,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {"index": 2, "type": 13, "attackId": 1072},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [
                _game(
                    steps=[
                        _agent_step(3, attack_id=1072),
                        _step(turn=7, action=[1], options=options, observation=observation),
                    ]
                )
            ]
        )

        self.assertEqual(
            [case.failure_class for case in result.cases],
            ["handoff_preparation_missed"],
        )
        self.assertEqual(result.cases[0].case_status, "fail")

    def test_terminal_powerful_hand_is_not_a_handoff_failure(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[
                {"id": KADABRA, "energies": [], "serial": 2, "appearThisTurn": False},
            ],
            hand_count=10,
            prizes=1,
        )
        player["hand"] = [{"id": BASIC_PSYCHIC, "serial": 10}]
        opponent = _state(active={"id": 900, "hp": 180, "maxHp": 180}, prizes=3)
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, opponent],
        }
        options = [
            {
                "index": 0,
                "type": 8,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 1, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[1], options=options, observation=observation)])]
        )

        self.assertNotIn(
            "handoff_preparation_missed", {case.failure_class for case in result.cases}
        )

    def test_enriching_draw_that_closes_prize_is_not_a_handoff_failure(self):
        player = _state(
            active={"id": ALAKAZAM, "energies": [PSYCHIC_ENERGY], "serial": 1},
            bench=[
                {"id": KADABRA, "energies": [], "serial": 2, "appearThisTurn": False},
                {"id": DUDUNSPARCE, "energies": [], "serial": 3},
            ],
            hand_count=7,
            prizes=1,
        )
        player["hand"] = [
            {"id": BASIC_PSYCHIC, "serial": 10},
            {"id": ENRICHING_ENERGY, "serial": 11},
        ]
        opponent = _state(active={"id": 900, "hp": 200, "maxHp": 200}, prizes=3)
        current = {
            "turn": 7,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [player, opponent],
        }
        options = [
            {
                "index": 0,
                "type": 8,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "area": 2,
                "indexInArea": 1,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {"index": 2, "type": 13, "attackId": POWERFUL_HAND},
        ]
        observation = {
            "current": current,
            "select": {"type": 0, "option": options},
            "logs": [],
        }

        result = analyze_records(
            [_game(steps=[_step(turn=7, action=[1], options=options, observation=observation)])]
        )

        self.assertNotIn(
            "handoff_preparation_missed", {case.failure_class for case in result.cases}
        )

    def test_empty_bench_run_away_draw_creates_hard_case(self):
        player = _state(active={"id": DUDUNSPARCE, "energies": []}, bench=[])
        result = analyze_records(
            [
                _game(
                    steps=[
                        _step(
                            turn=5,
                            action=[0],
                            options=[_option(10, card_id=DUDUNSPARCE)],
                            players=[player, _state(active={"id": 900})],
                        )
                    ]
                )
            ]
        )

        self.assertEqual(result.metrics.empty_bench_run_away_draw_count, 1)
        self.assertEqual(result.cases[0].failure_class, "empty_bench_run_away_draw")
        self.assertEqual(result.cases[0].case_status, "fail")

    def test_resolves_run_away_draw_card_from_active_area_option(self):
        player = _state(
            active={"id": DUDUNSPARCE, "energies": [], "serial": 12}, bench=[]
        )
        result = analyze_records(
            [
                _game(
                    steps=[
                        _step(
                            turn=5,
                            action=[0],
                            options=[
                                {
                                    "index": 0,
                                    "type": 10,
                                    "cardId": None,
                                    "area": 4,
                                    "indexInArea": 0,
                                }
                            ],
                            players=[player, _state(active={"id": 900})],
                        )
                    ]
                )
            ]
        )

        self.assertEqual(result.metrics.empty_bench_run_away_draw_count, 1)

    def test_post_ko_metrics_use_events_and_games_as_separate_denominators(self):
        ko_steps = _raw_post_ko_step(game=1, ready_count=0)
        ko_steps_ready = _raw_post_ko_step(game=2, ready_count=1)
        result = analyze_records(
            [
                _game(game=1, steps=[ko_steps]),
                _game(game=2, steps=[ko_steps_ready]),
            ]
        )

        self.assertEqual(result.metrics.post_ko_count, 2)
        self.assertEqual(result.metrics.post_ko_zero_ready_count, 1)
        self.assertEqual(result.metrics.post_ko_zero_ready_event_rate, 0.5)
        self.assertEqual(result.metrics.games_with_post_ko_break, 1)
        self.assertEqual(result.metrics.games_with_post_ko_break_rate, 0.5)

    def test_ignores_field_to_discard_movement_without_knockout_damage(self):
        current = {
            "turn": 5,
            "yourIndex": 0,
            "players": [
                _state(active=None, bench=[]),
                _state(active={"id": 900}),
            ],
        }
        observation = {
            "current": current,
            "select": {"type": 0, "option": []},
            "logs": [
                {
                    "type": 6,
                    "playerIndex": 0,
                    "cardId": ALAKAZAM,
                    "serial": 88,
                    "fromArea": 5,
                    "toArea": 3,
                }
            ],
        }
        result = analyze_records(
            [_game(steps=[_step(turn=5, observation=observation)])]
        )

        self.assertEqual(result.metrics.post_ko_count, 0)

    def test_write_analysis_creates_compact_machine_and_human_reports(self):
        result = analyze_records(
            [
                _game(
                    steps=[
                        _step(
                            turn=5,
                            action=[0],
                            options=[_option(10, card_id=DUDUNSPARCE)],
                            players=[
                                _state(active={"id": DUDUNSPARCE}, bench=[]),
                                _state(active={"id": 900}),
                            ],
                        )
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_analysis(result, output_dir, {"label": LABEL, "games": 1})
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "cases.jsonl").exists())
            self.assertTrue((output_dir / "analysis.md").exists())
            json.loads((output_dir / "metrics.json").read_text())


def _metrics(*, win_rate=0.60, second_turn=0.27, empty_bench_draws=0):
    return EvaluationMetrics(
        games=100,
        wins=round(win_rate * 100),
        losses=round((1 - win_rate) * 100),
        draws=0,
        errors=0,
        win_rate=win_rate,
        meta_weighted_win_rate=win_rate,
        second_turn_powerful_hand_games=round(second_turn * 100),
        second_turn_powerful_hand_rate=second_turn,
        post_ko_count=10,
        post_ko_zero_ready_count=1,
        post_ko_zero_ready_event_rate=0.1,
        games_with_post_ko_break=1,
        games_with_post_ko_break_rate=0.01,
        empty_bench_run_away_draw_count=empty_bench_draws,
        first_alakazam_turns=(),
    )


class TestComparisonAndRunner(unittest.TestCase):
    def test_candidate_passes_when_case_improves_and_win_rate_is_flat(self):
        decision = decide_promotion(
            _metrics(win_rate=0.60, second_turn=0.27),
            _metrics(win_rate=0.60, second_turn=0.28, empty_bench_draws=0),
            target_case_improved=True,
        )

        self.assertEqual(decision.status, "accept")

    def test_candidate_is_rejected_after_two_independent_win_rate_declines(self):
        decision = decide_promotion(
            _metrics(win_rate=0.60),
            _metrics(win_rate=0.55),
            target_case_improved=True,
            independent_pairs=[(0.60, 0.55), (0.61, 0.56)],
        )

        self.assertEqual(decision.status, "reject")
        self.assertIn("win_rate", decision.reasons)

    def test_replay_command_keeps_fixed_protocol(self):
        command = build_replay_command(
            evaluator_root=Path("/tmp/ptcg-agent-kaggle"),
            agent=Path("submission/alakazam_v7_auto_iter/main.py"),
            label="alakazam_v7_auto_iter",
            opponents=["kiyotah_dragapult"],
            games=10,
            output=Path("/tmp/iter-focus"),
            cg_path=Path("submission/alakazam_v7_auto_iter"),
            save_traces=True,
        )

        self.assertIn("--save-traces", command)
        self.assertIn("--games", command)
        self.assertIn("10", command)
        self.assertIn("kiyotah_dragapult", command)

    def test_replay_command_defaults_to_summary_only(self):
        command = build_replay_command(
            evaluator_root=Path("/tmp/ptcg-agent-kaggle"),
            agent=Path("submission/alakazam_v7_auto_iter/main.py"),
            label="alakazam_v7_auto_iter",
            opponents=["kiyotah_dragapult"],
            games=10,
            output=Path("/tmp/iter-summary-default"),
            cg_path=Path("submission/alakazam_v7_auto_iter"),
        )

        self.assertNotIn("--save-traces", command)

    def test_replay_command_can_skip_large_full_traces(self):
        command = build_replay_command(
            evaluator_root=Path("/tmp/ptcg-agent-kaggle"),
            agent=Path("submission/alakazam_v7_auto_iter/main.py"),
            label="alakazam_v7_auto_iter",
            opponents=["kiyotah_dragapult"],
            games=10,
            output=Path("/tmp/iter-summary"),
            cg_path=Path("submission/alakazam_v7_auto_iter"),
            save_traces=False,
        )

        self.assertNotIn("--save-traces", command)

    def test_compare_reports_writes_comparison_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            control = root / "control"
            candidate = root / "candidate"
            output = root / "comparison"
            for path, metrics in (
                (control, _metrics()),
                (candidate, _metrics(second_turn=0.28)),
            ):
                path.mkdir()
                (path / "metrics.json").write_text(
                    json.dumps({"metrics": metrics.__dict__, "cases": []})
                )

            comparison = compare_reports(control, candidate, output)

            self.assertEqual(comparison["status"], "observe")
            self.assertTrue((output / "comparison.json").exists())
            self.assertTrue((output / "decision.md").exists())


if __name__ == "__main__":
    unittest.main()
