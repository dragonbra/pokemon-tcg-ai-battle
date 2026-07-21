from __future__ import annotations

import sys
import unittest
from pathlib import Path

from tests.alakazam_v8_fixtures import main_obs, player, pokemon


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / "work" / "alakazam_v8_current"
sys.path.insert(0, str(CANDIDATE_ROOT))

from strategy.cards import (  # noqa: E402
    ALAKAZAM,
    DUDUNSPARCE,
    KADABRA,
    load_deck,
    make_deck_spec,
)
from strategy.facts import build_turn_facts  # noqa: E402
from strategy.memory import GameMemory  # noqa: E402
from strategy.model import ActionIntent, ActionKind, DecisionPhase  # noqa: E402
from strategy.options import decode_options, match_intent  # noqa: E402


class DeckSpecTests(unittest.TestCase):
    def test_counts_are_derived_from_the_v8_deck(self) -> None:
        spec = make_deck_spec(load_deck())

        self.assertEqual(spec.count(DUDUNSPARCE), 2)
        self.assertNotIn(1146, spec.card_ids)
        self.assertEqual(sum(spec.counts.values()), 60)


class TurnFactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = GameMemory()
        self.deck_spec = make_deck_spec(load_deck())

    def test_unknown_prize_is_not_reported_as_known_deck(self) -> None:
        me = player(
            active=pokemon(ALAKAZAM, 1),
            hand=[KADABRA],
            deck_count=20,
            prize_count=6,
        )
        opponent = player(active=pokemon(900, 2))
        obs = main_obs(me, opponent, [{"type": 14}])

        facts = build_turn_facts(obs, self.memory, self.deck_spec)

        self.assertEqual(facts.resources.known_in_deck.get(KADABRA, 0), 0)
        self.assertGreater(facts.resources.unknown_deck_or_prize.get(KADABRA, 0), 0)

    def test_turn_budget_tracks_stadium_independently(self) -> None:
        me = player(active=pokemon(ALAKAZAM, 1))
        opponent = player(active=pokemon(900, 2))
        obs = main_obs(
            me,
            opponent,
            [{"type": 14}],
            supporter_played=False,
            stadium_played=True,
            energy_attached=False,
            retreated=False,
        )

        facts = build_turn_facts(obs, self.memory, self.deck_spec)

        self.assertTrue(facts.budget.stadium_used)
        self.assertFalse(facts.budget.supporter_used)
        self.assertFalse(facts.budget.energy_used)
        self.assertFalse(facts.budget.retreat_used)

    def test_only_the_immediately_previous_opponent_turn_triggers_post_ko(self) -> None:
        me = player(active=pokemon(ALAKAZAM, 1))
        opponent = player(active=pokemon(900, 2))
        first_logs = [
            {"type": 2, "playerIndex": 1},
            {"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3, "serial": 77},
            {"type": 2, "playerIndex": 0},
        ]
        first = main_obs(me, opponent, [{"type": 14}], turn=5, logs=first_logs)

        first_facts = build_turn_facts(first, self.memory, self.deck_spec)

        later_logs = [
            *first_logs,
            {"type": 3, "playerIndex": 0},
            {"type": 2, "playerIndex": 1},
            {"type": 3, "playerIndex": 1},
            {"type": 2, "playerIndex": 0},
        ]
        later = main_obs(me, opponent, [{"type": 14}], turn=7, logs=later_logs)
        later_facts = build_turn_facts(later, self.memory, self.deck_spec)

        self.assertTrue(first_facts.previous_opponent_turn_had_ko)
        self.assertFalse(later_facts.previous_opponent_turn_had_ko)

    def test_evolution_clock_uses_the_turn_start_snapshot(self) -> None:
        active = pokemon(KADABRA, 11)
        me = player(active=active)
        opponent = player(active=pokemon(900, 2))
        obs = main_obs(me, opponent, [{"type": 14}], turn=5)

        facts = build_turn_facts(obs, self.memory, self.deck_spec)

        self.assertTrue(facts.yours.active.can_evolve)


class SemanticOptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = GameMemory()
        self.deck_spec = make_deck_spec(load_deck())

    def test_play_and_evolve_options_resolve_card_and_target(self) -> None:
        me = player(
            active=pokemon(KADABRA, 11),
            hand=[ALAKAZAM],
        )
        opponent = player(active=pokemon(900, 2))
        raw_options = [
            {"type": 7, "index": 0},
            {"type": 9, "cardId": ALAKAZAM, "inPlayArea": 4, "inPlayIndex": 0},
        ]
        obs = main_obs(me, opponent, raw_options)
        facts = build_turn_facts(obs, self.memory, self.deck_spec)

        options = decode_options(obs["select"], facts)

        self.assertEqual(options[0].action_kind, ActionKind.PLAY)
        self.assertEqual(options[0].card_id, ALAKAZAM)
        self.assertEqual(options[1].action_kind, ActionKind.EVOLVE)
        self.assertEqual(options[1].card_id, ALAKAZAM)
        self.assertEqual(options[1].target, facts.yours.active)

    def test_attached_energy_selection_resolves_opponent_target(self) -> None:
        me = player(active=pokemon(ALAKAZAM, 1))
        opponent = player(
            active=pokemon(900, 2),
            bench=[pokemon(901, 3, energy_cards=[11])],
        )
        raw_options = [
            {
                "type": 6,
                "playerIndex": 1,
                "area": 5,
                "indexInArea": 0,
                "energyIndex": 0,
            }
        ]
        obs = main_obs(me, opponent, raw_options)
        obs["select"]["type"] = 4
        obs["select"]["context"] = 30
        facts = build_turn_facts(obs, self.memory, self.deck_spec)

        option = decode_options(obs["select"], facts)[0]

        self.assertEqual(option.action_kind, ActionKind.SELECT)
        self.assertEqual(option.energy_id, 11)
        self.assertEqual(option.target, facts.opponent.bench[0])

    def test_match_intent_uses_semantics_instead_of_option_order(self) -> None:
        me = player(active=pokemon(KADABRA, 11), hand=[ALAKAZAM])
        opponent = player(active=pokemon(900, 2))
        raw_options = [
            {"type": 14},
            {"type": 9, "cardId": ALAKAZAM, "inPlayArea": 4, "inPlayIndex": 0},
        ]
        obs = main_obs(me, opponent, raw_options)
        facts = build_turn_facts(obs, self.memory, self.deck_spec)
        options = decode_options(obs["select"], facts)
        intent = ActionIntent(
            rule_id="evolution.active_kadabra",
            phase=DecisionPhase.CURRENT_ATTACKER,
            action_kind=ActionKind.EVOLVE,
            purpose="prepare_current_attacker",
            card_id=ALAKAZAM,
            target_key=facts.yours.active.key,
        )

        matched = match_intent(intent, options)

        self.assertIsNotNone(matched)
        self.assertEqual(matched.index, 1)


if __name__ == "__main__":
    unittest.main()
