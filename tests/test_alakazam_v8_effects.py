from __future__ import annotations

import sys
import unittest
from pathlib import Path

from tests.alakazam_v8_fixtures import effect_obs, player, pokemon


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / "work" / "alakazam_v8_current"
sys.path.insert(0, str(CANDIDATE_ROOT))

from strategy.cards import (  # noqa: E402
    ABRA,
    ALAKAZAM,
    BASIC_PSYCHIC,
    DAWN,
    DUNSPARCE,
    DUDUNSPARCE,
    ENHANCED_HAMMER,
    HILDA,
    KADABRA,
    LANAS_AID,
    MIST_ENERGY,
    NIGHT_STRETCHER,
    POKE_PAD,
    SACRED_ASH,
    TELEPATH_ENERGY,
    load_deck,
    make_deck_spec,
)
from strategy.effects.dispatcher import select_effect  # noqa: E402
from strategy.facts import build_turn_facts  # noqa: E402
from strategy.memory import GameMemory  # noqa: E402
from strategy.options import decode_options  # noqa: E402
from strategy.planner import build_turn_plan  # noqa: E402
from strategy.profiles import BASELINE_PROFILE  # noqa: E402
from strategy.routes import analyze_routes  # noqa: E402


class EffectSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = GameMemory()
        self.deck_spec = make_deck_spec(load_deck())

    def _choose(self, obs: dict):
        facts = build_turn_facts(obs, self.memory, self.deck_spec)
        options = decode_options(obs["select"], facts)
        routes = analyze_routes(facts, options)
        plan = build_turn_plan(facts, routes, BASELINE_PROFILE)
        return select_effect(obs, facts, plan, self.memory, BASELINE_PROFILE)

    def test_poke_pad_does_not_find_dudunsparce_without_ready_handoff(self) -> None:
        me = player(active=pokemon(DUNSPARCE, 1), bench=[pokemon(ABRA, 2)])
        obs = effect_obs(
            me,
            player(active=pokemon(900, 3)),
            [{"type": 1, "cardId": DUDUNSPARCE}, {"type": 1, "cardId": KADABRA}],
            effect_id=POKE_PAD,
            context=7,
        )

        self.assertEqual(self._choose(obs).option_indexes, (1,))

    def test_poke_pad_finds_dudunsparce_for_ready_alakazam_handoff(self) -> None:
        me = player(
            active=pokemon(DUNSPARCE, 1),
            bench=[pokemon(ALAKAZAM, 2, energies=[BASIC_PSYCHIC])],
        )
        obs = effect_obs(
            me,
            player(active=pokemon(900, 3)),
            [{"type": 1, "cardId": KADABRA}, {"type": 1, "cardId": DUDUNSPARCE}],
            effect_id=POKE_PAD,
            context=7,
        )

        self.assertEqual(self._choose(obs).option_indexes, (1,))

    def test_sacred_ash_recovers_complete_lines_before_duplicates(self) -> None:
        discard = [ABRA, ABRA, KADABRA, KADABRA, ALAKAZAM, ALAKAZAM]
        obs = effect_obs(
            player(active=pokemon(DUNSPARCE, 1), discard=discard),
            player(active=pokemon(900, 2)),
            [{"type": 3, "area": 3, "index": index} for index in range(6)],
            effect_id=SACRED_ASH,
            context=9,
            max_count=5,
        )

        decision = self._choose(obs)
        self.assertEqual(
            [discard[index] for index in decision.option_indexes],
            [ABRA, KADABRA, ALAKAZAM, ABRA, KADABRA],
        )

    def test_lanas_aid_fills_remaining_slot_with_basic_energy(self) -> None:
        discard = [ABRA, BASIC_PSYCHIC]
        obs = effect_obs(
            player(active=pokemon(DUNSPARCE, 1), discard=discard),
            player(active=pokemon(900, 2)),
            [{"type": 3, "area": 3, "index": index} for index in range(2)],
            effect_id=LANAS_AID,
            context=7,
            max_count=3,
        )

        self.assertEqual(self._choose(obs).option_indexes, (0, 1))

    def test_hammer_prefers_benched_protective_energy_over_active_other(self) -> None:
        opponent = player(
            active=pokemon(900, 2, energy_cards=[TELEPATH_ENERGY]),
            bench=[pokemon(901, 3, energy_cards=[MIST_ENERGY])],
        )
        obs = effect_obs(
            player(active=pokemon(ALAKAZAM, 1, energies=[BASIC_PSYCHIC])),
            opponent,
            [
                {
                    "type": 6,
                    "playerIndex": 1,
                    "area": 4,
                    "indexInArea": 0,
                    "energyIndex": 0,
                },
                {
                    "type": 6,
                    "playerIndex": 1,
                    "area": 5,
                    "indexInArea": 0,
                    "energyIndex": 0,
                },
            ],
            effect_id=ENHANCED_HAMMER,
            context=30,
        )

        self.assertEqual(self._choose(obs).option_indexes, (1,))

    def test_dawn_prefers_dunsparce_when_none_is_in_play(self) -> None:
        obs = effect_obs(
            player(active=pokemon(ALAKAZAM, 1)),
            player(active=pokemon(900, 2)),
            [{"type": 1, "cardId": ABRA}, {"type": 1, "cardId": DUNSPARCE}],
            effect_id=DAWN,
            context=7,
        )

        self.assertEqual(self._choose(obs).option_indexes, (1,))

    def test_hilda_prefers_dudunsparce_for_alakazam_handoff_engine(self) -> None:
        obs = effect_obs(
            player(
                active=pokemon(ALAKAZAM, 1),
                bench=[pokemon(KADABRA, 2), pokemon(DUNSPARCE, 3)],
            ),
            player(active=pokemon(900, 4)),
            [{"type": 1, "cardId": KADABRA}, {"type": 1, "cardId": DUDUNSPARCE}],
            effect_id=HILDA,
            context=7,
        )

        self.assertEqual(self._choose(obs).option_indexes, (1,))

    def test_night_stretcher_can_rank_kadabra_without_dispatch_name_error(self) -> None:
        discard = [KADABRA]
        obs = effect_obs(
            player(active=pokemon(DUNSPARCE, 1), discard=discard),
            player(active=pokemon(900, 2)),
            [{"type": 3, "area": 3, "index": 0}],
            effect_id=NIGHT_STRETCHER,
            context=7,
        )

        self.assertEqual(self._choose(obs).option_indexes, (0,))

    def test_dudunsparce_run_away_draw_accepts_an_unready_bench_successor(self) -> None:
        obs = effect_obs(
            player(
                active=pokemon(DUDUNSPARCE, 1),
                bench=[pokemon(ABRA, 2)],
                deck_count=30,
            ),
            player(active=pokemon(900, 2)),
            [{"type": 1}, {"type": 2}],
            effect_id=DUDUNSPARCE,
            context=43,
        )

        self.assertEqual(self._choose(obs).option_indexes, (0,))


if __name__ == "__main__":
    unittest.main()
