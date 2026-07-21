from __future__ import annotations

import unittest

from tests.alakazam_v8_fixtures import effect_obs, load_current_module, main_obs, player, pokemon


MODULE = load_current_module()


class V8RefactorRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        MODULE.agent({"select": None})

    def test_nonterminal_ko_waits_for_bench_kadabra_evolution(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ALAKAZAM,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.ALAKAZAM, 900, 901],
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {
                "type": 9,
                "cardId": MODULE.ALAKAZAM,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_dudunsparce_search_requires_ready_bench_attacker(self) -> None:
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.POKE_PAD],
        )
        opponent = player(active=pokemon(900, 3, hp=220))
        options = [
            {"type": 1, "cardId": MODULE.KADABRA},
            {"type": 1, "cardId": MODULE.DUDUNSPARCE},
        ]
        obs = effect_obs(
            me,
            opponent,
            options,
            effect_id=MODULE.POKE_PAD,
            context=7,
        )

        self.assertEqual(MODULE.agent(obs), [0])

    def test_sacred_ash_recovers_complete_lines_before_duplicate_stages(self) -> None:
        discard = [
            MODULE.ABRA,
            MODULE.ABRA,
            MODULE.KADABRA,
            MODULE.KADABRA,
            MODULE.ALAKAZAM,
            MODULE.ALAKAZAM,
        ]
        me = player(active=pokemon(MODULE.DUNSPARCE, 1), discard=discard)
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [{"type": 3, "area": 3, "index": index} for index in range(6)]
        obs = effect_obs(
            me,
            opponent,
            options,
            effect_id=MODULE.SACRED_ASH,
            context=9,
            min_count=1,
            max_count=5,
        )

        chosen = MODULE.agent(obs)
        chosen_ids = [discard[index] for index in chosen]
        self.assertEqual(
            chosen_ids,
            [MODULE.ABRA, MODULE.KADABRA, MODULE.ALAKAZAM, MODULE.ABRA, MODULE.KADABRA],
        )

    def test_enhanced_hammer_is_not_used_against_basic_energy_only(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ALAKAZAM,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            hand=[MODULE.ENHANCED_HAMMER, 900, 901],
        )
        opponent = player(
            active=pokemon(900, 2, hp=200, energy_cards=[MODULE.BASIC_PSYCHIC])
        )
        options = [
            {"type": 7, "cardId": MODULE.ENHANCED_HAMMER},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [1])


if __name__ == "__main__":
    unittest.main()
