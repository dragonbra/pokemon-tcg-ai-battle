from __future__ import annotations

import unittest
from dataclasses import asdict
import json
import tempfile
from pathlib import Path

from evaluation.metrics import (
    AggregateMetric,
    GameContext,
    GameMetric,
    MetricPresentation,
    MetricPlugin,
    OutcomePlugin,
    CorrectnessPlugin,
    LengthPlugin,
    default_metric_plugins,
)
from evaluation.metrics.registry import (
    CORE_METRIC_IDS,
    MetricRegistry,
    create_metric_registry,
    load_metric_plugin,
)
from evaluation.metrics.powerful_hand import PowerfulHandPlugin
from evaluation.metrics.rare_candy import RareCandyPlugin
from evaluation.metrics.post_ko_relay import PostKORelayPlugin
from evaluation.metrics.run_away_draw import RunAwayDrawPlugin
from evaluation.metrics.library_pressure import LibraryPressurePlugin
from evaluation.metrics.trace_utils import (
    action_list,
    current,
    field_state,
    knockout_is_confirmed,
    option_list,
    selected_attack_id,
)


def context(
    *,
    game_id: str = "game-1",
    candidate_physical_index: int = 0,
    candidate_first: bool = True,
) -> GameContext:
    return GameContext(
        game_id=game_id,
        candidate_name="candidate",
        opponent_name="opponent-a",
        candidate_physical_index=candidate_physical_index,
        candidate_first=candidate_first,
    )


def step(
    *,
    turn: int = 3,
    role: str = "candidate",
    action: list[int] | None = None,
    options: list[dict[str, object]] | None = None,
    observation: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "turn": turn,
        "role": role,
        "action": action or [],
        "select": {"options": options or []},
        "observation": observation or {},
    }


def trace(
    *,
    winner: int | str | None = 0,
    steps: list[dict[str, object]] | None = None,
    finished: bool = True,
    status: str = "",
    error_kind: str = "",
    error: str = "",
    reason: str = "",
    visualization_error: str = "",
) -> dict[str, object]:
    result: dict[str, object] = {
        "winner": winner,
        "finished": finished,
        "status": status,
        "error_kind": error_kind,
        "error": error,
        "reason": reason,
        "trace": steps or [],
        "steps": len(steps or []),
    }
    if visualization_error:
        result["visualization_error"] = visualization_error
    return result


def worker_trace(
    *,
    winner: int | None = 0,
    finished: bool = True,
    status: str = "finished",
    error_kind: str | None = None,
    error: str | None = None,
    steps: list[dict[str, object]] | None = None,
    result_steps: int | None = None,
    candidate_physical_index: int = 0,
) -> dict[str, object]:
    """Match the payload persisted by evaluation.runner.worker.run_game."""
    return {
        "run_id": "run-1",
        "game_id": "game-1",
        "candidate": "candidate",
        "opponent": "opponent-a",
        "candidate_first": True,
        "trace": steps or [],
        "result": {
            "game_id": "game-1",
            "opponent": "opponent-a",
            "candidate_first": True,
            "candidate_physical_index": candidate_physical_index,
            "finished": finished,
            "winner": winner,
            "status": status,
            "error_kind": error_kind,
            "error": error,
            "steps": len(steps or []) if result_steps is None else result_steps,
            "trace_path": "/tmp/game-1.json",
        },
    }


def strategy_step(
    *,
    step_index: int,
    turn: int,
    your_index: int,
    first_player: int,
    action: list[int] | None = None,
    options: list[dict[str, object]] | None = None,
    players: list[dict[str, object]] | None = None,
    logs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    current_state = {
        "turn": turn,
        "yourIndex": your_index,
        "firstPlayer": first_player,
        "players": players or [{}, {}],
        "result": -1,
    }
    return {
        "step": step_index,
        "state": current_state,
        "observation": {
            "current": current_state,
            "select": {"option": options or []},
            "logs": logs or [],
        },
        "action": action or [],
    }


def pokemon(card_id: int, serial: int, *, energies: list[int] | None = None) -> dict[str, object]:
    return {
        "id": card_id,
        "serial": serial,
        "hp": 80,
        "energies": energies or [],
        "preEvolution": [],
    }


def player(
    *,
    active_card: dict[str, object] | None = None,
    bench_cards: list[dict[str, object]] | None = None,
    hand_cards: list[dict[str, object]] | None = None,
    discard_cards: list[dict[str, object]] | None = None,
    deck_count: int = 20,
) -> dict[str, object]:
    return {
        "active": [active_card] if active_card else [],
        "bench": bench_cards or [],
        "hand": hand_cards or [],
        "discard": discard_cards or [],
        "deckCount": deck_count,
    }


class EvaluationMetricsTests(unittest.TestCase):
    def test_metric_contracts_are_immutable_and_plugins_are_defaulted(self) -> None:
        metric = GameMetric("x", "success", 1, 1, 1, (), ())
        aggregate = AggregateMetric("x", 1, 1, 1, {})

        with self.assertRaises(AttributeError):
            metric.value = 2  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            aggregate.value = 2  # type: ignore[misc]

        plugins = default_metric_plugins()
        self.assertEqual(tuple(plugin.metric_id for plugin in plugins), (
            "outcome",
            "length",
            "correctness",
        ))
        self.assertTrue(all(isinstance(plugin, MetricPlugin) for plugin in plugins))

    def test_trace_helpers_normalize_current_and_legacy_option_key(self) -> None:
        legacy = {
            "turn": 4,
            "action": [0, "bad", 1],  # type: ignore[list-item]
            "observation": {
                "current": {
                    "turn": "4",
                    "players": [{"active": [{"id": 1}], "bench": []}],
                },
                "select": {"option": [{"attackId": 1072}, {"type": 7}]},
            },
        }

        self.assertEqual(current(legacy)["turn"], "4")
        self.assertEqual(option_list(legacy), [{"attackId": 1072}, {"type": 7}])
        self.assertEqual(action_list(legacy), [0, 1])
        self.assertEqual(selected_attack_id(legacy), 1072)

    def test_trace_helpers_confirm_knockout_from_damage_or_zero_hp(self) -> None:
        observation = {
            "current": {
                "players": [
                    {"active": [{"serial": 4, "hp": 0}], "bench": []},
                    {},
                ]
            },
            "logs": [
                {
                    "type": 16,
                    "playerIndex": 0,
                    "serial": 4,
                    "putDamageCounter": True,
                }
            ],
        }
        raw_step = step(observation=observation)

        self.assertEqual(field_state(raw_step, 0)[4]["hp"], 0)
        self.assertTrue(
            knockout_is_confirmed(
                raw_step,
                observation["logs"][0],  # type: ignore[index]
                0,
                {},
            )
        )

    def test_trace_helpers_confirm_normal_damage_knockout_after_discard(self) -> None:
        knocked_out = pokemon(140, 4)
        hp_change = {
            "type": 16,
            "playerIndex": 0,
            "cardId": 140,
            "serial": 4,
            "value": -80,
            "putDamageCounter": False,
        }
        movement = {
            "type": 6,
            "playerIndex": 0,
            "cardId": 140,
            "serial": 4,
            "fromArea": 4,
            "toArea": 3,
        }
        raw_step = strategy_step(
            step_index=1,
            turn=4,
            your_index=1,
            first_player=0,
            players=[player(discard_cards=[knocked_out]), player()],
            logs=[hp_change, movement],
        )

        self.assertTrue(
            knockout_is_confirmed(
                raw_step,
                movement,
                0,
                {4: knocked_out},
            )
        )

    def test_outcome_classifies_win_loss_draw_and_error(self) -> None:
        plugin = OutcomePlugin()
        cases = [
            (trace(winner=0), "win"),
            (trace(winner=1), "loss"),
            (trace(winner=None), "draw"),
            (trace(winner=0, error="worker_crash"), "error"),
        ]

        for raw_trace, value in cases:
            with self.subTest(value=value):
                result = plugin.analyze_game(raw_trace, context())
                self.assertEqual(result.status, "success" if value != "error" else "error")
                self.assertEqual(result.value, value)
                self.assertEqual(result.denominator, 1)

    def test_outcome_counts_unfinished_and_preserves_attempt_denominator(self) -> None:
        plugin = OutcomePlugin()
        unfinished = plugin.analyze_game(
            trace(winner=None, finished=False, reason="step_limit"),
            context(),
        )
        aggregate = plugin.aggregate(
            [
                plugin.analyze_game(trace(winner=0), context(game_id="1")),
                plugin.analyze_game(trace(winner=1), context(game_id="2")),
                plugin.analyze_game(trace(winner=None), context(game_id="3")),
                unfinished,
            ]
        )

        self.assertEqual(unfinished.value, "unfinished")
        self.assertEqual(aggregate.numerator, 1)
        self.assertEqual(aggregate.denominator, 4)
        self.assertEqual(aggregate.value, 0.25)
        self.assertEqual(aggregate.by_opponent["opponent-a"]["attempts"], 4)

    def test_outcome_uses_worker_candidate_relative_winner_when_candidate_is_second(self) -> None:
        plugin = OutcomePlugin()
        result = plugin.analyze_game(
            trace(winner=0),
            context(candidate_physical_index=1, candidate_first=False),
        )

        self.assertEqual(result.value, "win")
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 1)

    def test_worker_payload_reads_nested_result_for_candidate_win(self) -> None:
        raw_trace = worker_trace(steps=[step()])
        outcome = OutcomePlugin().analyze_game(raw_trace, context())
        correctness = CorrectnessPlugin().analyze_game(raw_trace, context())

        self.assertEqual(outcome.value, "win")
        self.assertEqual(outcome.status, "success")
        self.assertEqual(correctness.value, "ok")
        self.assertEqual(correctness.status, "success")

    def test_worker_payload_uses_result_steps_and_state_evidence(self) -> None:
        worker_steps = [
            {
                "step": 0,
                "state": {"turn": 10, "yourIndex": 1, "result": -1},
                "observation": {"current": {"turn": 10, "yourIndex": 1, "result": -1}},
                "action": [0],
            },
            {
                "step": 1,
                "state": {"turn": 11, "yourIndex": 0, "result": -1},
                "observation": {"current": {"turn": 11, "yourIndex": 0, "result": -1}},
                "action": [0],
            },
            {
                "step": 2,
                "state": {"turn": 12, "yourIndex": 1, "result": -1},
                "observation": {"current": {"turn": 12, "yourIndex": 1, "result": -1}},
                "action": [0],
            },
            {
                "step": 3,
                "state": {"turn": 13, "yourIndex": 1, "result": 1},
                "observation": {"current": {"turn": 13, "yourIndex": 1, "result": 1}},
            },
        ]
        raw_trace = worker_trace(
            steps=worker_steps,
            result_steps=3,
            candidate_physical_index=1,
        )

        length = LengthPlugin().analyze_game(raw_trace, context(candidate_physical_index=1))
        outcome = OutcomePlugin().analyze_game(raw_trace, context(candidate_physical_index=1))
        correctness = CorrectnessPlugin().analyze_game(
            raw_trace,
            context(candidate_physical_index=1),
        )

        self.assertEqual(length.value, 7)
        self.assertEqual(length.payload["round"], 7)
        self.assertEqual(length.payload["engine_turn"], 13)
        self.assertEqual(length.payload["ending_phase"], "first_player")
        self.assertEqual(length.payload["action_selections"], 3)
        for result in (length, outcome, correctness):
            self.assertEqual(result.evidence[0]["step"], 3)
            self.assertEqual(result.evidence[0]["turn"], 13)
            self.assertEqual(result.evidence[0]["role"], "candidate")
            self.assertEqual(result.evidence[0]["action"], [])

    def test_length_keeps_worker_zero_steps_available(self) -> None:
        raw_trace = worker_trace(
            steps=[
                {
                    "step": 0,
                    "state": {"turn": 0, "yourIndex": 0, "result": 0},
                    "observation": {"current": {"turn": 0, "yourIndex": 0, "result": 0}},
                }
            ],
            result_steps=0,
        )

        result = LengthPlugin().analyze_game(raw_trace, context())

        self.assertEqual(result.status, "success")
        self.assertEqual(result.numerator, 0)
        self.assertEqual(result.denominator, 1)
        self.assertEqual(result.value, 0)

    def test_length_excludes_error_stop_turn_from_completed_game_average(self) -> None:
        completed = LengthPlugin().analyze_game(
            worker_trace(
                steps=[strategy_step(step_index=0, turn=13, your_index=0, first_player=0)],
                result_steps=30,
            ),
            context(game_id="completed"),
        )
        errored = LengthPlugin().analyze_game(
            worker_trace(
                winner=None,
                finished=False,
                status="game_error",
                error_kind="game_error",
                steps=[strategy_step(step_index=0, turn=7, your_index=0, first_player=0)],
                result_steps=20,
            ),
            context(game_id="errored"),
        )

        aggregate = LengthPlugin().aggregate([completed, errored])

        self.assertEqual(errored.status, "error")
        self.assertEqual(errored.denominator, 0)
        self.assertEqual(errored.payload["observed_engine_turn"], 7)
        self.assertEqual(errored.payload["observed_round"], 4)
        self.assertEqual(aggregate.denominator, 1)
        self.assertEqual(aggregate.value, 7.0)
        self.assertEqual(aggregate.payload["action_selections"]["value"], 25.0)

    def test_worker_payload_reads_nested_candidate_error(self) -> None:
        raw_trace = worker_trace(
            winner=None,
            finished=False,
            status="candidate_error",
            error_kind="candidate_error",
            error="RuntimeError: candidate boom",
            steps=[step()],
        )
        outcome = OutcomePlugin().analyze_game(raw_trace, context())
        correctness = CorrectnessPlugin().analyze_game(raw_trace, context())

        self.assertEqual(outcome.value, "error")
        self.assertEqual(outcome.status, "error")
        self.assertEqual(correctness.value, "candidate_error")
        self.assertEqual(correctness.status, "error")

    def test_finished_opponent_forfeit_is_a_candidate_win(self) -> None:
        raw_trace = worker_trace(
            winner=0,
            finished=True,
            status="opponent_error",
            error_kind="opponent_error",
            error="IndexError: invalid selected action",
            steps=[step(role="opponent")],
        )

        outcome = OutcomePlugin().analyze_game(raw_trace, context())

        self.assertEqual(outcome.value, "win")
        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.numerator, 1)

    def test_worker_payload_maps_engine_and_worker_failures(self) -> None:
        plugin = CorrectnessPlugin()
        cases = (
            ("game_error", "engine_error"),
            ("cg_mismatch", "engine_error"),
            ("load_error", "worker_crash"),
        )

        for worker_error, expected in cases:
            with self.subTest(worker_error=worker_error):
                raw_trace = worker_trace(
                    winner=None,
                    finished=False,
                    status=worker_error,
                    error_kind=worker_error,
                    error=f"RuntimeError: {worker_error}",
                )
                result = plugin.analyze_game(
                    raw_trace,
                    context(),
                )
                outcome = OutcomePlugin().analyze_game(raw_trace, context())
                self.assertEqual(result.value, expected)
                self.assertEqual(result.status, "error")
                self.assertEqual(outcome.value, "error")
                self.assertEqual(outcome.status, "error")

    def test_outcome_counts_error_kinds_as_attempts_before_unfinished_fallback(self) -> None:
        plugin = OutcomePlugin()
        results = [
            plugin.analyze_game(
                trace(winner=None, finished=False, error_kind="candidate_error"),
                context(game_id="candidate-error"),
            ),
            plugin.analyze_game(
                trace(winner=None, finished=False, status="opponent_error"),
                context(game_id="opponent-error"),
            ),
            plugin.analyze_game(
                trace(winner=None, finished=False),
                context(game_id="unfinished"),
            ),
        ]

        self.assertEqual([result.value for result in results], ["error", "error", "unfinished"])
        aggregate = plugin.aggregate(results)
        self.assertEqual(aggregate.denominator, 3)
        self.assertEqual(aggregate.by_opponent["opponent-a"]["errors"], 2)
        self.assertEqual(aggregate.by_opponent["opponent-a"]["unfinished"], 1)

    def test_length_reports_engine_turns_and_audits_action_selections(self) -> None:
        plugin = LengthPlugin()
        empty = plugin.analyze_game({}, context())
        aggregate = plugin.aggregate(
            [
                plugin.analyze_game(trace(steps=[step(), step()]), context(game_id="1")),
                plugin.analyze_game(
                    trace(steps=[step(), step(), step(), step()]),
                    context(game_id="2"),
                ),
                empty,
            ]
        )

        self.assertEqual(empty.status, "unavailable")
        self.assertEqual(empty.denominator, 0)
        self.assertTrue({"step", "turn", "role", "action"} <= set(empty.evidence[0]))
        self.assertEqual(aggregate.numerator, 4)
        self.assertEqual(aggregate.denominator, 2)
        self.assertEqual(aggregate.value, 2.0)
        self.assertEqual(aggregate.payload["rounds"]["value"], 2.0)
        self.assertEqual(aggregate.payload["turns"]["value"], 2.0)
        self.assertEqual(
            aggregate.payload["turns"]["by_opponent"]["opponent-a"]["value"],
            2.0,
        )
        self.assertEqual(aggregate.payload["action_selections"]["value"], 3.0)

    def test_length_separates_win_loss_rounds_by_candidate_turn_order(self) -> None:
        plugin = LengthPlugin()

        def finished_game(
            game_id: str,
            winner: int,
            engine_turn: int,
            *,
            candidate_first: bool = True,
        ) -> GameMetric:
            return plugin.analyze_game(
                worker_trace(
                    winner=winner,
                    steps=[
                        strategy_step(
                            step_index=0,
                            turn=engine_turn,
                            your_index=0,
                            first_player=0,
                        )
                    ],
                ),
                context(game_id=game_id, candidate_first=candidate_first),
            )

        aggregate = plugin.aggregate(
            [
                finished_game("win-first-phase", 0, 5),
                finished_game("win-candidate-second", 0, 6, candidate_first=False),
                finished_game("loss-second-phase", 1, 10),
            ]
        )

        wins = aggregate.payload["by_outcome"]["win"]
        losses = aggregate.payload["by_outcome"]["loss"]
        self.assertEqual(wins["average"], 3.0)
        self.assertEqual(wins["distribution"]["3"]["candidate_first"], 1)
        self.assertEqual(wins["distribution"]["3"]["candidate_second"], 1)
        self.assertEqual(losses["average"], 5.0)
        self.assertEqual(losses["distribution"]["5"]["candidate_first"], 1)
        self.assertEqual(losses["distribution"]["5"]["candidate_second"], 0)

    def test_correctness_attributes_candidate_and_opponent_errors(self) -> None:
        plugin = CorrectnessPlugin()
        candidate = plugin.analyze_game(
            trace(
                finished=False,
                error_kind="candidate_error",
                error="candidate selected unavailable option",
                steps=[step()],
            ),
            context(),
        )
        opponent = plugin.analyze_game(
            trace(
                finished=False,
                status="opponent_error",
                error="invalid move",
                steps=[step(role="opponent")],
            ),
            context(),
        )

        self.assertEqual(candidate.value, "candidate_error")
        self.assertEqual(candidate.status, "error")
        self.assertEqual(opponent.value, "opponent_error")
        self.assertEqual(opponent.status, "success")
        self.assertTrue(candidate.evidence)
        self.assertTrue({"step", "turn", "role", "action"} <= set(candidate.evidence[0]))
        aggregate = plugin.aggregate([candidate, opponent])
        self.assertEqual(aggregate.numerator, 1)
        self.assertEqual(aggregate.denominator, 2)

    def test_correctness_handles_visualization_step_limit_and_empty_trace(self) -> None:
        plugin = CorrectnessPlugin()
        for raw_trace, expected in (
            (trace(finished=False, visualization_error="missing frames"), "visualization_error"),
            (trace(finished=False, error_kind="step_limit"), "unfinished"),
            (trace(finished=False, status="engine_error"), "engine_error"),
        ):
            with self.subTest(expected=expected):
                raw_trace["trace"] = [step(action=[0])]
                raw_trace["steps"] = 1
                result = plugin.analyze_game(raw_trace, context())
                self.assertEqual(result.value, expected)
        unavailable = plugin.analyze_game({}, context())
        self.assertEqual(unavailable.status, "unavailable")
        self.assertEqual(unavailable.denominator, 0)
        self.assertTrue({"step", "turn", "role", "action"} <= set(unavailable.evidence[0]))

    def test_registry_registers_and_analyzes_plugins(self) -> None:
        registry = MetricRegistry(default_metric_plugins())
        results = registry.analyze(trace(winner=0, steps=[step()]), context())

        self.assertEqual(set(results), {"outcome", "length", "correctness"})
        aggregates = registry.aggregate(results)
        self.assertEqual(aggregates["outcome"].value, 1.0)

    def test_registry_supports_payload_and_plugin_presentation(self) -> None:
        class PresentedPlugin:
            metric_id = "presented"

            def analyze_game(self, trace, context):
                return GameMetric(
                    self.metric_id,
                    "success",
                    1,
                    2,
                    0.5,
                    (),
                    (),
                    {"seen": True},
                )

            def aggregate(self, results):
                return AggregateMetric(
                    self.metric_id,
                    1,
                    2,
                    0.5,
                    {},
                    {"total": 1},
                )

            def render(self, aggregate, results):
                return MetricPresentation(
                    self.metric_id,
                    "Presented",
                    "## Presented",
                    "<h2>Presented</h2>",
                )

        registry = MetricRegistry((PresentedPlugin(),), trusted_plugins=("presented",))
        results = registry.analyze({}, context())
        aggregates = registry.aggregate(results)
        presentations = registry.present(aggregates, results)

        self.assertEqual(results["presented"].payload["seen"], True)
        self.assertEqual(aggregates["presented"].payload["total"], 1)
        self.assertEqual(presentations["presented"].title, "Presented")
        self.assertIn("Presented", presentations["presented"].markdown)

    def test_registry_uses_generic_presentation_for_legacy_plugins(self) -> None:
        class LegacyPlugin:
            metric_id = "legacy"

            def analyze_game(self, trace, context):
                return GameMetric(self.metric_id, "success", 1, 1, "ok", (), ())

            def aggregate(self, results):
                return AggregateMetric(self.metric_id, 1, 1, 1.0, {})

        registry = MetricRegistry((LegacyPlugin(),))
        results = registry.analyze({}, context())
        presentations = registry.present(registry.aggregate(results), results)

        self.assertEqual(presentations["legacy"].metric_id, "legacy")
        self.assertIn("1", presentations["legacy"].markdown)
        self.assertIn("legacy", presentations["legacy"].html)

    def test_generic_presentation_escapes_all_aggregate_values(self) -> None:
        class UnsafeValuesPlugin:
            metric_id = "unsafe_values"

            def analyze_game(self, trace, context):
                return GameMetric(self.metric_id, "success", 1, 1, "ok", (), ())

            def aggregate(self, results):
                return AggregateMetric(
                    self.metric_id,
                    "<img src=x onerror=alert(1)>",
                    "<svg onload=alert(2)>",
                    "<script>alert(3)</script>",
                    {},
                )

        registry = MetricRegistry((UnsafeValuesPlugin(),))
        results = registry.analyze({}, context())
        presentation = registry.present(registry.aggregate(results), results)["unsafe_values"]

        self.assertNotIn("<img", presentation.html)
        self.assertNotIn("<svg", presentation.html)
        self.assertNotIn("<script", presentation.html)
        self.assertIn("&lt;img", presentation.html)


    def test_dynamic_plugin_cannot_inject_raw_html_presentation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "unsafe_metric.py"
            module_path.write_text(
                """
from evaluation.metrics.base import AggregateMetric, GameMetric, MetricPresentation

class UnsafePlugin:
    metric_id = 'unsafe_metric'
    def analyze_game(self, trace, context):
        return GameMetric(self.metric_id, 'success', 1, 1, 1, (), ())
    def aggregate(self, results):
        return AggregateMetric(self.metric_id, 1, 1, 1, {})
    def render(self, aggregate, results):
        return MetricPresentation(self.metric_id, 'Unsafe', '<script>alert(1)</script>', '<script>alert(1)</script>')
""",
                encoding="utf-8",
            )

            registry = create_metric_registry([str(module_path)])
            results = registry.analyze(trace(winner=0), context())
            presentation = registry.present(registry.aggregate(results), results)["unsafe_metric"]

        self.assertEqual(presentation.title, "指标：unsafe_metric")
        self.assertNotIn("<script", presentation.html.lower())

    def test_registry_plugins_are_untrusted_without_explicit_allowlist(self) -> None:
        class UntrustedPlugin:
            metric_id = "untrusted"

            def analyze_game(self, trace, context):
                return GameMetric(self.metric_id, "success", 1, 1, 1, (), ())

            def aggregate(self, results):
                return AggregateMetric(self.metric_id, 1, 1, 1, {})

            def render(self, aggregate, results):
                return MetricPresentation(
                    self.metric_id,
                    "Untrusted",
                    "# raw",
                    "<script>alert(1)</script>",
                )

        registry = MetricRegistry((UntrustedPlugin(),))
        results = registry.analyze(trace(winner=0), context())
        presentation = registry.present(registry.aggregate(results), results)["untrusted"]

        self.assertEqual(presentation.title, "指标：untrusted")
        self.assertNotIn("<script", presentation.html.lower())

    def test_registry_isolates_metric_analysis_errors(self) -> None:
        class BrokenPlugin:
            metric_id = "broken"

            def analyze_game(self, trace, context):
                raise RuntimeError("fixture metric failure")

            def aggregate(self, results):
                return AggregateMetric(self.metric_id, 0, 0, None, {})

        registry = MetricRegistry((OutcomePlugin(), BrokenPlugin()))
        results = registry.analyze(trace(winner=0), context())

        self.assertEqual(results["outcome"].value, "win")
        self.assertEqual(results["broken"].status, "error")
        self.assertEqual(results["broken"].value, "metric_error")
        self.assertEqual(results["broken"].denominator, 1)
        self.assertEqual(results["broken"].payload["games"], 1)
        self.assertEqual(results["broken"].payload["error_games"], 1)
        self.assertIn("fixture metric failure", results["broken"].diagnostics[0]["error"])

    def test_registry_aggregate_errors_preserve_attempt_audit(self) -> None:
        class BrokenAggregatePlugin:
            metric_id = "broken_aggregate"

            def analyze_game(self, trace, context):
                return GameMetric(self.metric_id, "success", 0, 1, 0, (), ())

            def aggregate(self, results):
                raise RuntimeError("fixture aggregate failure")

        registry = MetricRegistry((BrokenAggregatePlugin(),))
        result = registry.analyze({}, context())["broken_aggregate"]
        aggregate = registry.aggregate({"broken_aggregate": [result]})["broken_aggregate"]

        self.assertEqual(aggregate.denominator, 1)
        self.assertEqual(aggregate.payload["games"], 1)
        self.assertEqual(aggregate.payload["error_games"], 1)
        self.assertIn("fixture aggregate failure", aggregate.payload["error"])

    def test_powerful_hand_uses_candidate_relative_second_turn_and_actual_selection(self) -> None:
        plugin = PowerfulHandPlugin()
        cases = (
            (
                "candidate-first",
                0,
                True,
                [
                    strategy_step(step_index=0, turn=1, your_index=0, first_player=0),
                    strategy_step(
                        step_index=1,
                        turn=3,
                        your_index=0,
                        first_player=0,
                        action=[0],
                        options=[{"type": 13, "attackId": 1072}],
                    ),
                ],
                1,
            ),
            (
                "candidate-second",
                1,
                False,
                [
                    strategy_step(step_index=0, turn=2, your_index=1, first_player=0),
                    strategy_step(
                        step_index=1,
                        turn=4,
                        your_index=1,
                        first_player=0,
                        action=[0],
                        options=[{"type": 13, "attackId": 1072}],
                    ),
                ],
                1,
            ),
            (
                "third-own-turn-does-not-count",
                0,
                True,
                [
                    strategy_step(step_index=0, turn=1, your_index=0, first_player=0),
                    strategy_step(step_index=1, turn=3, your_index=0, first_player=0),
                    strategy_step(
                        step_index=2,
                        turn=5,
                        your_index=0,
                        first_player=0,
                        action=[0],
                        options=[{"type": 13, "attackId": 1072}],
                    ),
                ],
                0,
            ),
        )

        for name, candidate_index, candidate_first, steps, expected in cases:
            with self.subTest(name=name):
                result = plugin.analyze_game(
                    worker_trace(
                        steps=steps,
                        candidate_physical_index=candidate_index,
                    ),
                    context(
                        candidate_physical_index=candidate_index,
                        candidate_first=candidate_first,
                    ),
                )
                self.assertEqual(result.numerator, expected)
                self.assertEqual(result.denominator, 1)
                self.assertEqual(result.diagnostics[0]["all_games_denominator"], 1)
                self.assertEqual(result.diagnostics[0]["reached_second_turn_denominator"], 1)

    def test_powerful_hand_distinguishes_not_selected_from_unavailable(self) -> None:
        plugin = PowerfulHandPlugin()
        first_turn = strategy_step(step_index=0, turn=1, your_index=0, first_player=0)
        legal_not_selected = strategy_step(
            step_index=1,
            turn=3,
            your_index=0,
            first_player=0,
            action=[1],
            options=[
                {"type": 13, "attackId": 1072},
                {"type": 13, "attackId": 999},
            ],
        )
        unavailable = strategy_step(
            step_index=2,
            turn=3,
            your_index=0,
            first_player=0,
            options=[{"type": 14}],
        )

        missed = plugin.analyze_game(
            worker_trace(steps=[first_turn, legal_not_selected]),
            context(),
        )
        absent = plugin.analyze_game(
            worker_trace(steps=[first_turn, unavailable]),
            context(),
        )

        self.assertEqual(missed.value, "not_selected")
        self.assertEqual(missed.status, "success")
        self.assertEqual(absent.value, "unavailable")
        self.assertEqual(absent.status, "unavailable")
        self.assertEqual(missed.evidence[0]["step"], 1)
        self.assertEqual(absent.evidence[0]["step"], 2)

    def test_powerful_hand_records_game_and_reached_turn_denominators(self) -> None:
        plugin = PowerfulHandPlugin()
        reached = plugin.analyze_game(
            worker_trace(
                steps=[
                    strategy_step(step_index=0, turn=1, your_index=0, first_player=0),
                    strategy_step(step_index=1, turn=3, your_index=0, first_player=0),
                ]
            ),
            context(game_id="reached"),
        )
        not_reached = plugin.analyze_game(
            worker_trace(
                finished=False,
                status="unfinished",
                steps=[strategy_step(step_index=0, turn=1, your_index=0, first_player=0)],
            ),
            context(game_id="short"),
        )

        aggregate = plugin.aggregate([reached, not_reached])

        self.assertEqual(aggregate.denominator, 2)
        self.assertEqual(aggregate.by_opponent["opponent-a"]["all_games_denominator"], 2)
        self.assertEqual(
            aggregate.by_opponent["opponent-a"]["reached_second_turn_denominator"],
            1,
        )
        self.assertEqual(not_reached.diagnostics[0]["reason"], "second_turn_not_reached")

    def test_rare_candy_alakazam_attack_requires_ordered_second_turn_chain(self) -> None:
        plugin = RareCandyPlugin()
        abra = pokemon(741, 10, energies=[5])
        alakazam = pokemon(743, 11, energies=[5])
        alakazam["preEvolution"] = [{"id": 741, "serial": 10}]
        steps = [
            strategy_step(step_index=0, turn=1, your_index=0, first_player=0),
            strategy_step(
                step_index=1,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 7, "cardId": 1079}],
                players=[player(active_card=abra), player()],
            ),
            strategy_step(
                step_index=2,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 9, "cardId": 743}],
                players=[player(active_card=alakazam), player()],
            ),
            strategy_step(
                step_index=3,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 13, "attackId": 1072}],
                players=[player(active_card=alakazam), player()],
            ),
        ]

        result = plugin.analyze_game(worker_trace(steps=steps), context())

        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 1)
        self.assertEqual(result.value, "success")
        self.assertIsNone(result.diagnostics[0]["failure_stage"])
        self.assertEqual(result.evidence[0]["step"], 3)

    def test_rare_candy_resolves_official_play_option_from_hand_index(self) -> None:
        plugin = RareCandyPlugin()
        abra = pokemon(741, 10, energies=[5])
        alakazam = pokemon(743, 11, energies=[5])
        alakazam["preEvolution"] = [{"id": 741, "serial": 10}]
        rare_candy = {"id": 1079, "serial": 30}
        filler = {"id": 1152, "serial": 31}
        steps = [
            strategy_step(step_index=0, turn=1, your_index=0, first_player=0),
            strategy_step(
                step_index=1,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 7, "index": 1}],
                players=[
                    player(active_card=abra, hand_cards=[filler, rare_candy]),
                    player(),
                ],
            ),
            strategy_step(
                step_index=2,
                turn=3,
                your_index=0,
                first_player=0,
                players=[player(active_card=alakazam), player()],
            ),
            strategy_step(
                step_index=3,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 13, "attackId": 1072}],
                players=[player(active_card=alakazam), player()],
            ),
        ]

        result = plugin.analyze_game(worker_trace(steps=steps), context())

        self.assertEqual(result.value, "success")

    def test_rare_candy_does_not_reuse_alakazam_present_before_play(self) -> None:
        plugin = RareCandyPlugin()
        abra = pokemon(741, 10)
        existing_alakazam = pokemon(743, 11, energies=[5])
        existing_alakazam["preEvolution"] = [{"id": 741, "serial": 9}]
        unchanged_player = player(
            active_card=existing_alakazam,
            bench_cards=[abra],
        )
        steps = [
            strategy_step(
                step_index=1,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 7, "id": 1079}],
                players=[unchanged_player, player()],
            ),
            strategy_step(
                step_index=2,
                turn=3,
                your_index=0,
                first_player=0,
                players=[unchanged_player, player()],
            ),
            strategy_step(
                step_index=3,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 13, "attackId": 1072}],
                players=[unchanged_player, player()],
            ),
        ]

        result = plugin.analyze_game(worker_trace(steps=steps), context())

        self.assertEqual(result.value, "evolution_not_completed")

    def test_rare_candy_requires_newly_evolved_instance_to_be_active(self) -> None:
        plugin = RareCandyPlugin()
        abra = pokemon(741, 10)
        existing_alakazam = pokemon(743, 11, energies=[5])
        existing_alakazam["preEvolution"] = [{"id": 741, "serial": 9}]
        evolved_alakazam = pokemon(743, 12, energies=[5])
        evolved_alakazam["preEvolution"] = [{"id": 741, "serial": 10}]
        before = player(active_card=existing_alakazam, bench_cards=[abra])
        after = player(active_card=existing_alakazam, bench_cards=[evolved_alakazam])
        steps = [
            strategy_step(
                step_index=1,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 7, "card_id": 1079}],
                players=[before, player()],
            ),
            strategy_step(
                step_index=2,
                turn=3,
                your_index=0,
                first_player=0,
                players=[after, player()],
            ),
            strategy_step(
                step_index=3,
                turn=3,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 13, "attackId": 1072}],
                players=[after, player()],
            ),
        ]

        result = plugin.analyze_game(worker_trace(steps=steps), context())

        self.assertEqual(result.value, "alakazam_not_active")

    def test_rare_candy_failure_stages_are_exact_and_serializable(self) -> None:
        plugin = RareCandyPlugin()
        abra = pokemon(741, 10, energies=[5])
        alakazam = pokemon(743, 11, energies=[5])
        alakazam["preEvolution"] = [{"id": 741, "serial": 10}]
        first_turn = strategy_step(step_index=0, turn=1, your_index=0, first_player=0)
        rare = strategy_step(
            step_index=1,
            turn=3,
            your_index=0,
            first_player=0,
            action=[0],
            options=[{"type": 7, "cardId": 1079}],
            players=[player(active_card=abra), player()],
        )
        evolved_bench = strategy_step(
            step_index=2,
            turn=3,
            your_index=0,
            first_player=0,
            players=[player(active_card=abra, bench_cards=[alakazam]), player()],
        )
        evolved_active = strategy_step(
            step_index=2,
            turn=3,
            your_index=0,
            first_player=0,
            players=[player(active_card=alakazam), player()],
        )
        attack_not_selected = strategy_step(
            step_index=3,
            turn=3,
            your_index=0,
            first_player=0,
            action=[1],
            options=[{"type": 13, "attackId": 1072}, {"type": 14}],
            players=[player(active_card=alakazam), player()],
        )
        cases = (
            ("rare_candy_not_played", [first_turn, evolved_active]),
            ("evolution_not_completed", [first_turn, rare]),
            ("alakazam_not_active", [first_turn, rare, evolved_bench]),
            ("no_legal_attack", [first_turn, rare, evolved_active]),
            (
                "attack_not_declared",
                [first_turn, rare, evolved_active, attack_not_selected],
            ),
        )

        for expected_stage, steps in cases:
            with self.subTest(expected_stage=expected_stage):
                result = plugin.analyze_game(worker_trace(steps=steps), context())
                self.assertEqual(result.numerator, 0)
                self.assertEqual(result.diagnostics[0]["failure_stage"], expected_stage)
                json.dumps(asdict(result))
                self.assertIn(
                    result.evidence[0]["step"],
                    {raw_step["step"] for raw_step in steps},
                )

    def test_post_ko_relay_counts_confirmed_events_and_zero_ready_attackers(self) -> None:
        plugin = PostKORelayPlugin()
        knocked_out = pokemon(743, 10, energies=[5])
        ready = pokemon(742, 20, energies=[5])

        def ko_trace(bench_cards: list[dict[str, object]]) -> dict[str, object]:
            return worker_trace(
                steps=[
                    strategy_step(
                        step_index=0,
                        turn=4,
                        your_index=1,
                        first_player=0,
                        players=[player(active_card=knocked_out), player()],
                    ),
                    strategy_step(
                        step_index=1,
                        turn=5,
                        your_index=0,
                        first_player=0,
                        players=[player(bench_cards=bench_cards), player()],
                        logs=[
                            {
                                "type": 16,
                                "playerIndex": 0,
                                "serial": 10,
                                "putDamageCounter": True,
                            },
                            {
                                "type": 6,
                                "playerIndex": 0,
                                "cardId": 743,
                                "serial": 10,
                                "fromArea": 4,
                                "toArea": 3,
                            },
                        ],
                    ),
                ]
            )

        zero_ready = plugin.analyze_game(ko_trace([]), context(game_id="zero"))
        has_relay = plugin.analyze_game(ko_trace([ready]), context(game_id="ready"))
        aggregate = plugin.aggregate([zero_ready, has_relay])

        self.assertEqual((zero_ready.numerator, zero_ready.denominator), (1, 1))
        self.assertEqual((has_relay.numerator, has_relay.denominator), (0, 1))
        self.assertEqual((aggregate.numerator, aggregate.denominator), (1, 2))
        self.assertEqual(zero_ready.diagnostics[0]["games_with_zero_ready"], 1)
        self.assertEqual(zero_ready.evidence[0]["step"], 1)

    def test_post_ko_relay_ignores_discard_movement_without_knockout_confirmation(self) -> None:
        plugin = PostKORelayPlugin()
        raw_trace = worker_trace(
            steps=[
                strategy_step(
                    step_index=0,
                    turn=5,
                    your_index=0,
                    first_player=0,
                    logs=[
                        {
                            "type": 6,
                            "playerIndex": 0,
                            "cardId": 743,
                            "serial": 10,
                            "fromArea": 4,
                            "toArea": 3,
                        }
                    ],
                )
            ]
        )

        result = plugin.analyze_game(raw_trace, context())

        self.assertEqual(result.denominator, 0)
        self.assertEqual(result.status, "unavailable")

    def test_post_ko_relay_counts_confirmed_ko_outside_alakazam_line(self) -> None:
        plugin = PostKORelayPlugin()
        knocked_out = pokemon(140, 10)
        raw_trace = worker_trace(
            steps=[
                strategy_step(
                    step_index=0,
                    turn=4,
                    your_index=1,
                    first_player=0,
                    players=[player(active_card=knocked_out), player()],
                ),
                strategy_step(
                    step_index=1,
                    turn=5,
                    your_index=0,
                    first_player=0,
                    players=[player(discard_cards=[knocked_out]), player()],
                    logs=[
                        {
                            "type": 16,
                            "playerIndex": 0,
                            "cardId": 140,
                            "serial": 10,
                            "value": -210,
                            "putDamageCounter": False,
                        },
                        {
                            "type": 6,
                            "playerIndex": 0,
                            "cardId": 140,
                            "serial": 10,
                            "fromArea": 4,
                            "toArea": 3,
                        },
                    ],
                ),
            ]
        )

        result = plugin.analyze_game(raw_trace, context())

        self.assertEqual((result.numerator, result.denominator), (1, 1))
        self.assertEqual(result.evidence[0]["knocked_out_card_id"], 140)

    def test_run_away_draw_counts_only_actual_empty_bench_selection(self) -> None:
        plugin = RunAwayDrawPlugin()
        dudunsparce = pokemon(66, 12)
        selected = strategy_step(
            step_index=0,
            turn=5,
            your_index=0,
            first_player=0,
            action=[0],
            options=[{"type": 10, "area": 4, "indexInArea": 0}],
            players=[player(active_card=dudunsparce), player()],
        )
        not_selected = strategy_step(
            step_index=1,
            turn=7,
            your_index=0,
            first_player=0,
            action=[1],
            options=[
                {"type": 10, "cardId": 66},
                {"type": 14},
            ],
            players=[player(active_card=dudunsparce), player()],
        )
        bench_not_empty = strategy_step(
            step_index=2,
            turn=9,
            your_index=0,
            first_player=0,
            action=[0],
            options=[{"type": 10, "cardId": 66}],
            players=[player(active_card=dudunsparce, bench_cards=[pokemon(741, 13)]), player()],
        )

        result = plugin.analyze_game(
            worker_trace(steps=[selected, not_selected, bench_not_empty]),
            context(),
        )

        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 1)
        self.assertEqual(result.value, 1)
        self.assertEqual(result.evidence[0]["step"], 0)

    def test_library_pressure_records_consumption_end_turn_and_unfinished(self) -> None:
        plugin = LibraryPressurePlugin()
        steps = [
            strategy_step(
                step_index=0,
                turn=5,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 7, "cardId": 1152}],
                players=[player(deck_count=12), player()],
            ),
            strategy_step(
                step_index=1,
                turn=5,
                your_index=0,
                first_player=0,
                action=[0],
                options=[{"type": 10, "cardId": 66}],
                players=[player(deck_count=9), player()],
            ),
            strategy_step(
                step_index=2,
                turn=6,
                your_index=1,
                first_player=0,
                players=[player(deck_count=7), player()],
            ),
        ]

        result = plugin.analyze_game(
            worker_trace(
                winner=None,
                finished=False,
                status="unfinished",
                error_kind="step_limit",
                steps=steps,
            ),
            context(),
        )

        self.assertEqual(result.numerator, 5)
        self.assertEqual(result.denominator, 1)
        self.assertEqual(result.diagnostics[0]["at_or_below_15_consumption"], 5)
        self.assertEqual(result.diagnostics[0]["at_or_below_10_consumption"], 2)
        self.assertEqual(
            result.diagnostics[0]["end_turn_deck_counts"],
            [{"turn": 5, "deck_count": 7}],
        )
        self.assertFalse(result.diagnostics[0]["deck_out"])
        self.assertTrue(result.diagnostics[0]["unfinished"])
        self.assertEqual([item["step"] for item in result.evidence], [0, 1])

    def test_library_pressure_identifies_candidate_deck_out_from_result_log(self) -> None:
        plugin = LibraryPressurePlugin()
        deck_out_step = strategy_step(
            step_index=0,
            turn=8,
            your_index=1,
            first_player=0,
            players=[player(), player(deck_count=0)],
            logs=[{"type": 23, "result": 0, "reason": 2}],
        )

        result = plugin.analyze_game(
            worker_trace(
                winner=1,
                candidate_physical_index=1,
                steps=[deck_out_step],
            ),
            context(candidate_physical_index=1, candidate_first=False),
        )

        self.assertTrue(result.diagnostics[0]["deck_out"])
        self.assertFalse(result.diagnostics[0]["unfinished"])
        json.dumps(asdict(result))

    def test_library_pressure_identifies_nested_result_deck_out(self) -> None:
        plugin = LibraryPressurePlugin()
        raw_trace = worker_trace(
            winner=1,
            candidate_physical_index=1,
            steps=[
                strategy_step(
                    step_index=0,
                    turn=8,
                    your_index=1,
                    first_player=0,
                    players=[player(), player(deck_count=0)],
                )
            ],
        )
        raw_trace["result"]["reason"] = 2  # type: ignore[index]

        result = plugin.analyze_game(
            raw_trace,
            context(candidate_physical_index=1, candidate_first=False),
        )

        self.assertTrue(result.diagnostics[0]["deck_out"])

    def test_registry_core_factory_has_unique_core_ids(self) -> None:
        registry = create_metric_registry()
        metric_ids = [plugin.metric_id for plugin in registry.plugins]

        self.assertEqual(len(metric_ids), len(set(metric_ids)))
        self.assertEqual(set(metric_ids), set(CORE_METRIC_IDS))

    def test_registry_loads_extra_module_without_overriding_core_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "extra_metric.py"
            module_path.write_text(
                """
from evaluation.metrics.base import AggregateMetric, GameMetric

class Helper:
    pass

class ExtraPlugin:
    metric_id = 'extra_metric'
    def analyze_game(self, trace, context):
        return GameMetric(self.metric_id, 'success', 1, 1, 'extra', (), ())
    def aggregate(self, results):
        return AggregateMetric(self.metric_id, 1, 1, 1.0, {})
""",
                encoding="utf-8",
            )
            loaded = load_metric_plugin(str(module_path))
            registry = create_metric_registry([str(module_path)])
            results = registry.analyze(trace(winner=0), context())

        self.assertEqual(loaded.metric_id, "extra_metric")
        self.assertIn("extra_metric", results)
        self.assertEqual(results["outcome"].value, "win")

    def test_registry_rejects_dynamic_plugin_that_uses_core_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "override.py"
            module_path.write_text(
                """
class OverridePlugin:
    metric_id = 'outcome'
    def analyze_game(self, trace, context):
        return None
    def aggregate(self, results):
        return None
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "core metric"):
                create_metric_registry([str(module_path)])


if __name__ == "__main__":
    unittest.main()
