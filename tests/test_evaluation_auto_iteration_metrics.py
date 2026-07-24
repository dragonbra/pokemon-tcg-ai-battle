from __future__ import annotations

import unittest

from evaluation.metrics.outcome import OutcomePlugin
from evaluation.metrics.powerful_hand import PowerfulHandPlugin
from evaluation.metrics.attack_quality import AttackQualityPlugin
from evaluation.metrics.post_ko_relay import PostKORelayPlugin
from evaluation.metrics.setup_relay import SetupRelayPlugin
from tests.test_evaluation_metrics import (
    context,
    player,
    pokemon,
    strategy_step,
    trace,
    worker_trace,
)


def _setup_trace(*, candidate_first: bool, target: bool = True) -> dict[str, object]:
    candidate_index = 0 if candidate_first else 1
    first_turn = 1 if candidate_first else 2
    target_turn = 3 if candidate_first else 4
    first_player = 0
    abra = pokemon(741, 10)
    dunsparce = pokemon(305, 11)
    alakazam = pokemon(743, 12, energies=[5])
    dudunsparce = pokemon(66, 13)
    opening_player = player(
        active_card=abra if candidate_first else dunsparce,
        bench_cards=[] if candidate_first else [],
        hand_cards=[
            {"id": 1079},
            {"id": 743},
            {"id": 5},
        ],
    )
    target_player = player(
        active_card=alakazam,
        bench_cards=[dudunsparce] if not candidate_first else [],
    )
    players_first = [opening_player, player()] if candidate_first else [player(), opening_player]
    players_target = [target_player, player()] if candidate_first else [player(), target_player]
    steps = [
        strategy_step(
            step_index=0,
            turn=first_turn,
            your_index=candidate_index,
            first_player=first_player,
            players=players_first,
            logs=[{"type": 2, "playerIndex": candidate_index}, {"type": 4, "playerIndex": candidate_index}],
        )
    ]
    if target:
        steps.append(
            strategy_step(
                step_index=1,
                turn=target_turn,
                your_index=candidate_index,
                first_player=first_player,
                action=[0],
                options=[{"type": 13, "attackId": 1072}],
                players=players_target,
                logs=[{"type": 4, "playerIndex": candidate_index}],
            )
        )
    return worker_trace(
        steps=steps,
        candidate_physical_index=candidate_index,
    )


def _bridge_trace(
    *,
    opening_abra: bool = False,
    first_turn_bench_abra: bool = True,
    use_ability: bool = True,
    matching_dunsparce_line: bool = True,
    psychic_energy: bool = True,
) -> dict[str, object]:
    dunsparce = pokemon(305, 11)
    abra = pokemon(741, 12)
    dudunsparce = pokemon(66, 21)
    dudunsparce["preEvolution"] = [pokemon(305, 11 if matching_dunsparce_line else 99)]
    alakazam = pokemon(743, 22, energies=[5] if psychic_energy else [])
    alakazam["preEvolution"] = [pokemon(741, 12)]
    opening_hand = [{"id": 1079}, {"id": 743}, {"id": 5}]
    if opening_abra:
        opening_hand.append({"id": 741})
    first_turn_bench = [abra] if first_turn_bench_abra else []
    ability_options = (
        [{"type": 10, "area": 4, "cardId": 66}]
        if use_ability
        else [{"type": 12}]
    )
    steps = [
        strategy_step(
            step_index=0,
            turn=1,
            your_index=0,
            first_player=0,
            players=[player(active_card=dunsparce, hand_cards=opening_hand), player()],
            logs=[{"type": 2, "playerIndex": 0}, {"type": 4, "playerIndex": 0}],
        ),
        strategy_step(
            step_index=1,
            turn=1,
            your_index=0,
            first_player=0,
            players=[player(active_card=dunsparce, bench_cards=first_turn_bench), player()],
        ),
        strategy_step(
            step_index=2,
            turn=3,
            your_index=0,
            first_player=0,
            players=[player(active_card=dunsparce, bench_cards=first_turn_bench), player()],
        ),
        strategy_step(
            step_index=3,
            turn=3,
            your_index=0,
            first_player=0,
            action=[0],
            options=ability_options,
            players=[player(active_card=dudunsparce, bench_cards=first_turn_bench), player()],
        ),
        strategy_step(
            step_index=4,
            turn=3,
            your_index=0,
            first_player=0,
            players=[player(active_card=abra), player()],
            logs=[{"type": 4, "playerIndex": 0}],
        ),
        strategy_step(
            step_index=5,
            turn=3,
            your_index=0,
            first_player=0,
            action=[0],
            options=[{"type": 13, "attackId": 1072}],
            players=[player(active_card=alakazam), player()],
        ),
    ]
    return worker_trace(steps=steps)


class AutoIterationMetricTests(unittest.TestCase):
    def test_powerful_hand_records_target_turn_and_reached_payload(self) -> None:
        plugin = PowerfulHandPlugin()

        result = plugin.analyze_game(
            _setup_trace(candidate_first=False),
            context(candidate_physical_index=1, candidate_first=False),
        )

        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.diagnostics[0]["target_turn"], 4)
        self.assertTrue(result.payload["reached_second_turn"])
        self.assertEqual(result.payload["target_turn"], 4)

    def test_powerful_hand_exposes_reached_numerator_and_rate(self) -> None:
        plugin = PowerfulHandPlugin()
        reached = plugin.analyze_game(
            _setup_trace(candidate_first=False),
            context(candidate_physical_index=1, candidate_first=False),
        )
        not_reached = plugin.analyze_game(
            _setup_trace(candidate_first=True, target=False),
            context(candidate_first=True),
        )

        aggregate = plugin.aggregate([reached, not_reached])

        self.assertEqual(aggregate.payload["reached_numerator"], 1)
        self.assertEqual(aggregate.payload["reached_denominator"], 1)
        self.assertEqual(aggregate.payload["reached_value"], 1.0)
        self.assertEqual(
            aggregate.payload["by_turn_order"]["second"]["reached_numerator"],
            1,
        )

    def test_powerful_hand_marks_worker_lifecycle_in_metric_status(self) -> None:
        plugin = PowerfulHandPlugin()
        target_step = strategy_step(
            step_index=0,
            turn=3,
            your_index=0,
            first_player=0,
            action=[0],
            options=[{"type": 13, "attackId": 1072}],
        )

        unfinished = plugin.analyze_game(
            worker_trace(
                finished=False,
                winner=None,
                status="unfinished",
                error_kind="step_limit",
                steps=[target_step],
            ),
            context(),
        )
        errored = plugin.analyze_game(
            worker_trace(
                finished=False,
                winner=None,
                status="worker_crash",
                error_kind="worker_crash",
                steps=[target_step],
            ),
            context(game_id="error"),
        )

        self.assertEqual(unfinished.status, "unavailable")
        self.assertEqual(unfinished.payload["lifecycle_status"], "unfinished")
        self.assertEqual(errored.status, "error")
        self.assertEqual(errored.payload["lifecycle_status"], "error")

    def test_outcome_aggregate_separates_actual_turn_order(self) -> None:
        plugin = OutcomePlugin()
        first = plugin.analyze_game(
            trace(winner=0),
            context(game_id="first", candidate_first=True),
        )
        second = plugin.analyze_game(
            trace(winner=1),
            context(game_id="second", candidate_first=False),
        )

        aggregate = plugin.aggregate([first, second])

        self.assertEqual(aggregate.payload["by_turn_order"]["first"]["wins"], 1)
        self.assertEqual(aggregate.payload["by_turn_order"]["second"]["losses"], 1)

    def test_setup_relay_separates_components_bridge_and_draw_types(self) -> None:
        plugin = SetupRelayPlugin()
        result = plugin.analyze_game(
            _bridge_trace(),
            context(),
        )

        self.assertTrue(result.payload["opening_components"]["rare_candy"])
        self.assertTrue(result.payload["bridge_opportunity"])
        self.assertTrue(result.payload["bridge_completed"])
        self.assertTrue(result.payload["bridge_components"]["used_run_away_draw"])
        self.assertTrue(result.payload["bridge_components"]["active_psychic_at_attack"])
        self.assertEqual(result.payload["draws_to_second_turn"]["ability_draw_cards"], 1)
        self.assertEqual(result.payload["draws_to_second_turn"]["normal_draw_cards"], 1)

        missing_target = plugin.analyze_game(
            _setup_trace(candidate_first=True, target=False),
            context(candidate_first=True),
        )
        aggregate = plugin.aggregate([result, missing_target])

        self.assertEqual(
            aggregate.payload["second_turn_draws"]["all_games"]["games"],
            2,
        )
        self.assertEqual(
            aggregate.payload["second_turn_draws"]["normal_draw_cards"]["all_games"]["total"],
            2,
        )

    def test_setup_relay_requires_strict_opportunity_and_ordered_serial_route(self) -> None:
        plugin = SetupRelayPlugin()

        opening_abra = plugin.analyze_game(_bridge_trace(opening_abra=True), context())
        no_bench_abra = plugin.analyze_game(
            _bridge_trace(first_turn_bench_abra=False), context()
        )
        retreat = plugin.analyze_game(_bridge_trace(use_ability=False), context())
        wrong_line = plugin.analyze_game(
            _bridge_trace(matching_dunsparce_line=False), context()
        )
        no_energy = plugin.analyze_game(
            _bridge_trace(psychic_energy=False), context()
        )

        self.assertFalse(opening_abra.payload["bridge_opportunity"])
        self.assertFalse(no_bench_abra.payload["bridge_opportunity"])
        for result in (retreat, wrong_line, no_energy):
            self.assertTrue(result.payload["bridge_opportunity"])
            self.assertFalse(result.payload["bridge_completed"])

    def test_attack_quality_separates_resolution_prize_and_unknown_states(self) -> None:
        plugin = AttackQualityPlugin()

        def attack_player(prize_count: int | None) -> dict[str, object]:
            value = player(active_card=pokemon(743, 10, energies=[5]))
            if prize_count is not None:
                value["prizeCount"] = prize_count
            return value

        def attack_trace(after_prize_count: int | None, resolved: bool) -> dict[str, object]:
            steps = [
                strategy_step(
                    step_index=0,
                    turn=3,
                    your_index=0,
                    first_player=0,
                    action=[0],
                    options=[{"type": 13, "attackId": 1072}],
                    players=[attack_player(6), player()],
                )
            ]
            if resolved:
                steps.extend(
                    [
                        strategy_step(
                            step_index=1,
                            turn=3,
                            your_index=0,
                            first_player=0,
                            players=[attack_player(6), player()],
                            logs=[{"type": 15, "playerIndex": 0, "attackId": 1072}],
                        ),
                        strategy_step(
                            step_index=2,
                            turn=3,
                            your_index=0,
                            first_player=0,
                            players=[attack_player(after_prize_count), player()],
                        ),
                    ]
                )
            return worker_trace(steps=steps)

        prize_taken = plugin.analyze_game(attack_trace(5, True), context())
        no_prize = plugin.analyze_game(attack_trace(6, True), context())
        unknown_prize = plugin.analyze_game(attack_trace(None, True), context())
        unresolved = plugin.analyze_game(attack_trace(None, False), context())
        aggregate = plugin.aggregate([prize_taken, no_prize, unknown_prize, unresolved])

        self.assertEqual(prize_taken.payload["resolved_attacks"], 1)
        self.assertEqual(no_prize.payload["non_prize_attacks"], 1)
        self.assertEqual(unknown_prize.payload["unknown_prize_attacks"], 1)
        self.assertEqual(unresolved.payload["unresolved_attacks"], 1)
        self.assertEqual(aggregate.payload["resolved_attacks"], 3)
        self.assertEqual(aggregate.payload["unknown_prize_attacks"], 1)
        self.assertEqual(aggregate.payload["non_prize_attacks"]["denominator"], 2)

    def test_post_ko_relay_exposes_conservative_failure_classification(self) -> None:
        plugin = PostKORelayPlugin()
        knocked_out = pokemon(743, 10, energies=[5])

        def ko_result(
            *,
            bench_cards: list[dict[str, object]] | None = None,
            hand_cards: list[dict[str, object]] | None = None,
            discard_cards: list[dict[str, object]] | None = None,
            deck_count: int = 20,
        ) -> str:
            after = player(
                bench_cards=bench_cards,
                hand_cards=hand_cards,
                discard_cards=discard_cards,
                deck_count=deck_count,
            )
            result = plugin.analyze_game(
                worker_trace(
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
                            players=[after, player()],
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
                ),
                context(),
            )
            return next(
                name
                for name, count in result.payload["failure_counts"].items()
                if count == 1
            )

        self.assertEqual(ko_result(bench_cards=[pokemon(741, 20)]), "field_route_miss")
        self.assertEqual(ko_result(discard_cards=[pokemon(741, 20)]), "recoverable_discard_miss")
        self.assertEqual(ko_result(hand_cards=[{"id": 741}]), "recoverable_route_incomplete")
        self.assertEqual(ko_result(deck_count=20), "nonterminal_no_field_route")
        self.assertEqual(ko_result(deck_count=0), "terminal_no_resource")

    def test_event_metrics_preserve_error_and_unfinished_audit_counts(self) -> None:
        unfinished = worker_trace(
            finished=False,
            winner=None,
            status="unfinished",
            error_kind="step_limit",
            steps=[],
        )
        errored = worker_trace(
            finished=False,
            winner=None,
            status="worker_crash",
            error_kind="worker_crash",
            error="fixture crash",
            steps=[],
        )

        for plugin in (SetupRelayPlugin(), AttackQualityPlugin(), PostKORelayPlugin()):
            with self.subTest(metric_id=plugin.metric_id):
                results = [
                    plugin.analyze_game(unfinished, context()),
                    plugin.analyze_game(errored, context(game_id="error")),
                ]
                aggregate = plugin.aggregate(results)

                self.assertEqual(aggregate.payload["games"], 2)
                self.assertEqual(aggregate.payload["unfinished_games"], 1)
                self.assertEqual(aggregate.payload["error_games"], 1)


if __name__ == "__main__":
    unittest.main()
