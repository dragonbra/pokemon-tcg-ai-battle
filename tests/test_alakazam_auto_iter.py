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
    write_analysis,
)


LABEL = "alakazam_v7"
PSYCHIC_ENERGY = 5
DUDUNSPARCE = 66
KADABRA = 742
ALAKAZAM = 743


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

    def test_does_not_count_powerful_hand_on_later_turn(self):
        result = analyze_records(
            [_game(steps=[_agent_step(5, attack_id=1072)])]
        )

        self.assertEqual(result.metrics.second_turn_powerful_hand_games, 0)
        self.assertEqual(
            sum(
                case.failure_class == "second_turn_powerful_hand_missing"
                for case in result.cases
            ),
            1,
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

    def test_second_turn_without_powerful_hand_creates_missing_case(self):
        result = analyze_records(
            [_game(steps=[_agent_step(3, options=[_option(14)])])]
        )

        self.assertEqual(result.metrics.second_turn_powerful_hand_games, 0)
        self.assertEqual(
            [case.failure_class for case in result.cases],
            ["second_turn_powerful_hand_missing"],
        )


class TestCaseReports(unittest.TestCase):
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
        )

        self.assertIn("--save-traces", command)
        self.assertIn("--games", command)
        self.assertIn("10", command)
        self.assertIn("kiyotah_dragapult", command)

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
