from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import scripts.auto_iteration_report as auto_iteration_report
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
    def _write_native_run(
        self,
        history_root: Path,
        iteration_id: str,
        *,
        wins: int,
        errors: int,
    ) -> tuple[Path, Path]:
        run_root = history_root / iteration_id / f"run-{iteration_id}"
        run_root.mkdir(parents=True)
        first_wins = min(wins, 2)
        second_wins = max(wins - 2, 0)
        first_losses = 2 - first_wins
        second_losses = max(2 - second_wins - errors, 0)
        summary = {
            "total_games": 4,
            "completed_games": 4 - errors,
            "wins": wins,
            "losses": 4 - wins - errors,
            "draws": 0,
            "errors": errors,
            "unfinished": 0,
            "win_rate": wins / 4,
            "by_opponent": {
                "fixture_a": {
                    "games": 2,
                    "wins": first_wins,
                    "losses": first_losses,
                    "draws": 0,
                    "errors": 0,
                    "unfinished": 0,
                    "win_rate": first_wins / 2,
                },
                "fixture_b": {
                    "games": 2,
                    "wins": second_wins,
                    "losses": second_losses,
                    "draws": 0,
                    "errors": errors,
                    "unfinished": 0,
                    "win_rate": second_wins / 2,
                },
            },
        }
        outcome_groups = {
            "all_games": {
                "games": 4,
                "numerator": wins,
                "denominator": 4,
                "value": wins / 4,
                "wins": wins,
                "losses": 4 - wins - errors,
                "draws": 0,
                "errors": errors,
                "unfinished": 0,
            },
            "by_turn_order": {
                "first": {
                    "games": 2,
                    "numerator": first_wins,
                    "denominator": 2,
                    "value": first_wins / 2,
                    "wins": first_wins,
                    "losses": first_losses,
                    "draws": 0,
                    "errors": 0,
                    "unfinished": 0,
                },
                "second": {
                    "games": 2,
                    "numerator": second_wins,
                    "denominator": 2,
                    "value": second_wins / 2,
                    "wins": second_wins,
                    "losses": second_losses,
                    "draws": 0,
                    "errors": errors,
                    "unfinished": 0,
                },
            },
        }
        metrics = {
            "outcome": {
                "metric_id": "outcome",
                "numerator": wins,
                "denominator": 4,
                "value": wins / 4,
                "payload": outcome_groups,
            },
            "correctness": {
                "metric_id": "correctness",
                "numerator": errors,
                "denominator": 4,
                "value": errors / 4,
                "payload": {},
            },
            "powerful_hand": {
                "metric_id": "powerful_hand",
                "numerator": 2,
                "denominator": 4,
                "value": 0.5,
                "payload": {
                    "all_games": {"numerator": 2, "denominator": 4, "value": 0.5},
                    "by_turn_order": {
                        "first": {"numerator": 1, "denominator": 2, "value": 0.5},
                        "second": {"numerator": 1, "denominator": 2, "value": 0.5},
                    },
                },
            },
            "setup_relay": {
                "metric_id": "setup_relay",
                "numerator": 2,
                "denominator": 4,
                "value": 0.5,
                "payload": {
                    "opening_four_components": {
                        "component_counts": {
                            "active_abra": 3,
                            "rare_candy": 2,
                            "alakazam_or_search": 4,
                            "psychic_energy_or_hilda": 3,
                            "all_four": 1,
                        },
                        "sample_games": 4,
                    },
                    "dunsparce_bridge": {"numerator": 1, "denominator": 2},
                    "by_turn_order": {
                        "first": {
                            "games": 2,
                            "bridge_successes": 1,
                            "bridge_opportunities": 1,
                        },
                        "second": {
                            "games": 2,
                            "bridge_successes": 0,
                            "bridge_opportunities": 1,
                        },
                    },
                    "second_turn_draws": {
                        "all_games": {"total": 8, "games": 4, "average": 2.0},
                        "reached_second_turn": {"total": 8, "games": 4, "average": 2.0},
                        "first": {"total": 3, "games": 2, "average": 1.5},
                        "second": {"total": 5, "games": 2, "average": 2.5},
                    },
                },
            },
            "post_ko_relay": {
                "metric_id": "post_ko_relay",
                "numerator": 3,
                "denominator": 6,
                "value": 0.5,
                "payload": {
                    "opportunities": 6,
                    "successes": 3,
                    "success_rate": 0.5,
                    "failure_counts": {"recoverable_discard_miss": 2},
                },
            },
            "attack_quality": {
                "metric_id": "attack_quality",
                "numerator": 2,
                "denominator": 10,
                "value": 0.2,
                "payload": {
                    "attack_submissions": 10,
                    "resolved_attacks": 10,
                    "unresolved_attacks": 0,
                    "unknown_prize_attacks": 0,
                    "non_prize_attacks": {"numerator": 2, "denominator": 10, "rate": 0.2},
                    "powerful_hand": {
                        "non_prize_attacks": 1,
                        "resolved_attacks": 8,
                        "unknown_prize_attacks": 0,
                    },
                },
            },
            "library_pressure": {
                "metric_id": "library_pressure",
                "numerator": 7,
                "denominator": 4,
                "value": 1.75,
            },
        }
        manifest = {
            "games": 4,
            "metric_profile": {"id": "auto_iteration_v8_setup_relay", "revision": 2},
            "candidate": {"name": "alakazam_v9", "package_hash": "candidate-hash"},
            "control": None,
            "opponents": [{"name": "fixture_a"}, {"name": "fixture_b"}],
        }
        summary_path = run_root / "summary.json"
        metrics_path = run_root / "metrics.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        (run_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_root / "report.html").write_text("<h1>native report</h1>", encoding="utf-8")
        (run_root / "report.md").write_text("# native report", encoding="utf-8")
        return summary_path, metrics_path

    def test_build_native_iteration_normalizes_metrics_and_writes_artifacts(self) -> None:
        build_native_iteration = getattr(auto_iteration_report, "build_native_iteration", None)
        self.assertTrue(callable(build_native_iteration), "缺少 build_native_iteration")

        with tempfile.TemporaryDirectory() as temporary:
            history_root = Path(temporary)
            summary_path, metrics_path = self._write_native_run(
                history_root, "baseline", wins=2, errors=1
            )
            document = build_native_iteration(
                summary_path,
                metrics_path,
                history_root,
                iteration_id="baseline",
                label="V9 fixed-deck baseline",
                change_summary="固定卡表策略基线",
                decision="observe",
                agent_label="alakazam_v9_baseline",
            )

            iteration_root = history_root / "baseline"
            self.assertEqual(document["metadata"]["sample_type"], "full")
            self.assertIsNone(document["metadata"]["control_id"])
            self.assertEqual(document["sample"]["games"], 4)
            self.assertEqual(document["sample"]["errors"], 1)
            self.assertEqual(document["metrics"]["win_rate"]["first"]["denominator"], 2)
            self.assertEqual(
                document["metrics"]["t2_alakazam"]["overall"]["all_games"]["rate"], 0.5
            )
            self.assertEqual(
                document["metrics"]["post_ko_relay"]["overall"]["success_rate"], 0.5
            )
            self.assertEqual(
                document["metrics"]["non_prize_attacks"]["overall"]
                ["non_prize_attacks"]["denominator"],
                10,
            )
            self.assertEqual(document["metrics"]["library_pressure"]["value"], 1.75)
            self.assertIsNone(document["native_metrics"]["library_pressure"]["payload"])
            self.assertEqual(document["native_metrics"]["correctness"]["denominator"], 4)
            for artifact in ("result.json", "iteration.md", "index.html"):
                self.assertTrue((iteration_root / artifact).is_file())
            self.assertTrue((history_root / "index.html").is_file())

            detail_html = (iteration_root / "index.html").read_text(encoding="utf-8")
            iteration_markdown = (iteration_root / "iteration.md").read_text(
                encoding="utf-8"
            )
            self.assertIn('href="run-baseline/report.html"', detail_html)
            self.assertIn('href="run-baseline/report.md"', detail_html)
            self.assertIn("主要假设", detail_html)
            self.assertIn("固定卡表策略基线", detail_html)
            self.assertIn("observe", detail_html)
            self.assertIn("Revision: 2", detail_html)
            self.assertIn("legacy zero-ready 失败率", detail_html)
            self.assertIn("profile revision：`2`", iteration_markdown)
            self.assertIn("legacy zero-ready 失败率", iteration_markdown)
            self.assertIn("<td>1/4</td><td>未提供</td><td>未提供</td>", detail_html)
            self.assertNotIn("<td>0/0</td>", detail_html)

    def test_native_post_ko_relay_preserves_values_and_declares_semantics(self) -> None:
        build_native_iteration = getattr(auto_iteration_report, "build_native_iteration", None)
        self.assertTrue(callable(build_native_iteration), "缺少 build_native_iteration")

        with tempfile.TemporaryDirectory() as temporary:
            history_root = Path(temporary)
            summary_path, metrics_path = self._write_native_run(
                history_root, "baseline", wins=2, errors=0
            )
            document = build_native_iteration(
                summary_path,
                metrics_path,
                history_root,
                iteration_id="baseline",
                label="baseline",
                change_summary="baseline",
                decision="observe",
                agent_label="alakazam_v9",
            )

        relay = document["native_metrics"]["post_ko_relay"]
        self.assertEqual(relay["numerator"], 3)
        self.assertEqual(relay["denominator"], 6)
        self.assertEqual(relay["value"], 0.5)
        self.assertEqual(
            relay.get("semantics"),
            {
                "value": "legacy_zero_ready_failure_rate",
                "direction": "lower_is_better",
                "numerator": "failure_count",
                "denominator": "opportunity_count",
            },
        )

    def test_build_native_iteration_relativizes_only_whitelisted_run_artifacts(self) -> None:
        build_native_iteration = getattr(auto_iteration_report, "build_native_iteration", None)
        self.assertTrue(callable(build_native_iteration), "缺少 build_native_iteration")

        with tempfile.TemporaryDirectory() as temporary:
            history_root = Path(temporary)
            summary_path, metrics_path = self._write_native_run(
                history_root, "baseline", wins=2, errors=0
            )
            run_root = summary_path.parent
            absolute_trace = f"{run_root.resolve()}/traces/example.json"
            for filename in ("cases.jsonl", "games.jsonl", "report.md", "report.html"):
                (run_root / filename).write_text(absolute_trace, encoding="utf-8")
            untouched = run_root / "unlisted.txt"
            untouched_text = absolute_trace
            untouched.write_text(untouched_text, encoding="utf-8")
            nested = run_root / "nested"
            nested.mkdir()
            nested_file = nested / "report.md"
            nested_file.write_text(absolute_trace, encoding="utf-8")

            build_native_iteration(
                summary_path,
                metrics_path,
                history_root,
                iteration_id="baseline",
                label="baseline",
                change_summary="baseline",
                decision="observe",
                agent_label="alakazam_v9",
            )

            for filename in ("cases.jsonl", "games.jsonl", "report.md", "report.html"):
                content = (run_root / filename).read_text(encoding="utf-8")
                self.assertNotIn(str(run_root.resolve()), content)
                self.assertIn("traces/example.json", content)
            self.assertEqual(untouched.read_text(encoding="utf-8"), untouched_text)
            self.assertEqual(nested_file.read_text(encoding="utf-8"), absolute_trace)

    def test_build_native_iteration_rejects_external_run_before_relativizing(self) -> None:
        build_native_iteration = getattr(auto_iteration_report, "build_native_iteration", None)
        self.assertTrue(callable(build_native_iteration), "缺少 build_native_iteration")

        with (
            tempfile.TemporaryDirectory() as history_temporary,
            tempfile.TemporaryDirectory() as external_temporary,
        ):
            history_root = Path(history_temporary)
            summary_path, metrics_path = self._write_native_run(
                Path(external_temporary), "baseline", wins=2, errors=0
            )
            run_root = summary_path.parent
            absolute_trace = f"{run_root.resolve()}/traces/example.json"
            cases_path = run_root / "cases.jsonl"
            cases_path.write_text(absolute_trace, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must remain under"):
                build_native_iteration(
                    summary_path,
                    metrics_path,
                    history_root,
                    iteration_id="baseline",
                    label="external",
                    change_summary="external",
                    decision="reject",
                    agent_label="alakazam_v9",
                )

            self.assertEqual(cases_path.read_text(encoding="utf-8"), absolute_trace)

    def test_build_native_iteration_rejects_unsafe_iteration_ids(self) -> None:
        build_native_iteration = getattr(auto_iteration_report, "build_native_iteration", None)
        self.assertTrue(callable(build_native_iteration), "缺少 build_native_iteration")

        with tempfile.TemporaryDirectory() as temporary:
            history_root = Path(temporary)
            summary_path, metrics_path = self._write_native_run(
                history_root, "baseline", wins=2, errors=0
            )
            for iteration_id in ("", ".", "..", "/absolute", "nested/iteration", r"nested\iteration"):
                with self.subTest(iteration_id=iteration_id):
                    with self.assertRaisesRegex(ValueError, "iteration_id"):
                        build_native_iteration(
                            summary_path,
                            metrics_path,
                            history_root,
                            iteration_id=iteration_id,
                            label="invalid",
                            change_summary="invalid",
                            decision="reject",
                            agent_label="alakazam_v9",
                        )

    def test_build_native_iteration_rebuilds_cross_iteration_history(self) -> None:
        build_native_iteration = getattr(auto_iteration_report, "build_native_iteration", None)
        self.assertTrue(callable(build_native_iteration), "缺少 build_native_iteration")

        with tempfile.TemporaryDirectory() as temporary:
            history_root = Path(temporary)
            for iteration_id, wins, errors, control_id in (
                ("baseline", 2, 1, None),
                ("iteration-001", 3, 0, "baseline"),
            ):
                summary_path, metrics_path = self._write_native_run(
                    history_root, iteration_id, wins=wins, errors=errors
                )
                build_native_iteration(
                    summary_path,
                    metrics_path,
                    history_root,
                    iteration_id=iteration_id,
                    label=iteration_id,
                    change_summary=f"{iteration_id} hypothesis",
                    decision="promote" if control_id else "observe",
                    agent_label="alakazam_v9",
                    control_id=control_id,
                )

            history_html = (history_root / "index.html").read_text(encoding="utf-8")

        self.assertIn('href="baseline/index.html"', history_html)
        self.assertIn('href="iteration-001/index.html"', history_html)
        self.assertIn("50.0% (2/4)", history_html)
        self.assertIn("75.0% (3/4)", history_html)
        self.assertIn("Correctness errors", history_html)
        self.assertIn("Powerful Hand", history_html)
        self.assertIn("Post-KO", history_html)
        self.assertIn("Decision", history_html)
        self.assertIn("iteration-001 hypothesis", history_html)
        self.assertIn("promote", history_html)

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
