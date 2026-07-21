from __future__ import annotations

import sys
import unittest
from pathlib import Path

from tests.alakazam_v8_fixtures import main_obs, player, pokemon


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / "work" / "alakazam_v8_current"
sys.path.insert(0, str(CANDIDATE_ROOT))

from strategy.cards import (  # noqa: E402
    ABRA,
    ALAKAZAM,
    BASIC_PSYCHIC,
    DUNSPARCE,
    DUDUNSPARCE,
    KADABRA,
    POWERFUL_HAND_ATTACK,
    load_deck,
    make_deck_spec,
)
from strategy.facts import build_turn_facts  # noqa: E402
from strategy.memory import GameMemory  # noqa: E402
from strategy.options import decode_options  # noqa: E402
from strategy.planner import build_turn_plan  # noqa: E402
from strategy.profiles import (  # noqa: E402
    BASELINE_PROFILE,
    AttackPreparation,
    with_variant,
)
from strategy.routes import analyze_routes  # noqa: E402
from strategy.model import PlanKind, RouteCertainty  # noqa: E402


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = GameMemory()
        self.deck_spec = make_deck_spec(load_deck())

    def _plan(self, me: dict, opponent: dict, options: list[dict], *, turn: int = 5):
        obs = main_obs(me, opponent, options, turn=turn)
        facts = build_turn_facts(obs, self.memory, self.deck_spec)
        decoded = decode_options(obs["select"], facts)
        routes = analyze_routes(facts, decoded)
        return build_turn_plan(facts, routes, BASELINE_PROFILE)

    def test_active_alakazam_nonterminal_ko_keeps_bench_kadabra_preparation(self) -> None:
        plan = self._plan(
            player(
                active=pokemon(ALAKAZAM, 1, energies=[BASIC_PSYCHIC]),
                bench=[pokemon(KADABRA, 2)],
                hand=[ALAKAZAM],
            ),
            player(active=pokemon(900, 3, hp=40)),
            [
                {"type": 9, "cardId": ALAKAZAM, "inPlayArea": 5, "inPlayIndex": 0},
                {"type": 13, "attackId": POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        )

        self.assertEqual(plan.kind, PlanKind.ATTACK)
        self.assertTrue(any(goal.rule_id == "evolution.bench_kadabra" for goal in plan.must_goals))
        self.assertIn("must_goals", plan.attack_blockers)

    def test_active_dunsparce_needs_confirmed_ready_handoff_before_switch_route(self) -> None:
        plan = self._plan(
            player(active=pokemon(DUNSPARCE, 1), bench=[pokemon(ABRA, 2)]),
            player(active=pokemon(900, 3, hp=220)),
            [{"type": 14}],
        )

        self.assertEqual(plan.handoff.certainty, RouteCertainty.BLOCKED)
        self.assertFalse(plan.handoff.via_dudunsparce)
        self.assertFalse(any("dudunsparce" in goal.rule_id for goal in plan.must_goals))

    def test_active_dunsparce_can_switch_to_ready_alakazam(self) -> None:
        plan = self._plan(
            player(
                active=pokemon(DUNSPARCE, 1),
                bench=[pokemon(ALAKAZAM, 2, energies=[BASIC_PSYCHIC])],
                hand=[DUDUNSPARCE],
            ),
            player(active=pokemon(900, 3, hp=220)),
            [
                {"type": 9, "cardId": DUDUNSPARCE, "inPlayArea": 4, "inPlayIndex": 0},
                {"type": 14},
            ],
        )

        self.assertEqual(plan.handoff.certainty, RouteCertainty.CONFIRMED)
        self.assertTrue(plan.handoff.via_dudunsparce)
        self.assertEqual(plan.handoff.attacker.card_id, ALAKAZAM)

    def test_unknown_dudunsparce_is_only_a_possible_search_handoff(self) -> None:
        obs = main_obs(
            player(
                active=pokemon(DUNSPARCE, 1),
                bench=[pokemon(ALAKAZAM, 2, energies=[BASIC_PSYCHIC])],
            ),
            player(active=pokemon(900, 3)),
            [{"type": 14}],
        )
        facts = build_turn_facts(obs, self.memory, self.deck_spec)
        routes = analyze_routes(facts, decode_options(obs["select"], facts))

        self.assertEqual(routes.handoff.certainty, RouteCertainty.POSSIBLE)
        self.assertTrue(routes.handoff.needs_search)
        self.assertTrue(routes.handoff.via_dudunsparce)

    def test_last_prize_may_skip_future_bench_preparation(self) -> None:
        plan = self._plan(
            player(
                active=pokemon(ALAKAZAM, 1, energies=[BASIC_PSYCHIC]),
                bench=[pokemon(KADABRA, 2)],
                hand=[900, 901, 902],
                prize_count=1,
            ),
            player(active=pokemon(900, 3, hp=40)),
            [{"type": 13, "attackId": POWERFUL_HAND_ATTACK}, {"type": 14}],
        )

        self.assertEqual(plan.kind, PlanKind.VICTORY)
        self.assertFalse(plan.must_goals)

    def test_profile_variant_changes_only_attack_preparation_policy(self) -> None:
        me = player(
            active=pokemon(ALAKAZAM, 1, energies=[BASIC_PSYCHIC]),
            bench=[pokemon(KADABRA, 2)],
            hand=[ALAKAZAM, 900, 901],
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"type": 9, "cardId": ALAKAZAM, "inPlayArea": 5, "inPlayIndex": 0},
            {"type": 13, "attackId": POWERFUL_HAND_ATTACK},
        ]
        obs = main_obs(me, opponent, options)
        facts = build_turn_facts(obs, self.memory, self.deck_spec)
        decoded = decode_options(obs["select"], facts)
        routes = analyze_routes(facts, decoded)
        variant = with_variant(
            BASELINE_PROFILE,
            attack_preparation=AttackPreparation.ATTACK_WHEN_LETHAL,
        )

        plan = build_turn_plan(facts, routes, variant)

        self.assertFalse(plan.must_goals)
        self.assertEqual(variant.dudunsparce, BASELINE_PROFILE.dudunsparce)


if __name__ == "__main__":
    unittest.main()
