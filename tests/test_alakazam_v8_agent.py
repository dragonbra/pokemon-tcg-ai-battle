from __future__ import annotations

import unittest

from tests.alakazam_v8_fixtures import effect_obs, load_current_module, main_obs, player, pokemon


MODULE = load_current_module()


class V8RefactorRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        MODULE.agent({"select": None})

    def test_active_abra_does_not_submit_teleportation_as_terminal_attack(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ABRA,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            hand=[MODULE.ALAKAZAM, 900, 901],
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {"type": 13, "attackId": MODULE.TELEPORTATION_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [1])

    def test_active_abra_evolves_to_kadabra_before_teleportation(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ABRA,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            hand=[MODULE.KADABRA, 900, 901],
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {
                "type": 9,
                "cardId": MODULE.KADABRA,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"type": 13, "attackId": MODULE.TELEPORTATION_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_active_abra_uses_hilda_to_start_missing_evolution_route(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ABRA,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            hand=[MODULE.HILDA, 900, 901],
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {"type": 7, "indexInArea": 0},
            {"type": 13, "attackId": MODULE.TELEPORTATION_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_active_alakazam_uses_hilda_to_complete_dunsparce_handoff_engine(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ALAKAZAM,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.HILDA, 900, 901],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"type": 7, "indexInArea": 0},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_unenergized_active_alakazam_uses_hilda_for_psychic_energy(self) -> None:
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.HILDA],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"type": 7, "indexInArea": 0},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_enriching_energy_draws_from_non_attack_active(self) -> None:
        me = player(
            active=pokemon(MODULE.FEZANDIPITI_EX, 1),
            hand=[MODULE.ENRICHING_ENERGY],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "type": 8,
                "cardId": MODULE.ENRICHING_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_enriching_energy_respects_low_deck_protection(self) -> None:
        me = player(
            active=pokemon(MODULE.FEZANDIPITI_EX, 1),
            hand=[MODULE.ENRICHING_ENERGY],
            deck_count=10,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "type": 8,
                "cardId": MODULE.ENRICHING_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [1])

    def test_ready_alakazam_builds_bench_insurance_after_initial_setup(self) -> None:
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[
                pokemon(MODULE.ABRA, 2),
                pokemon(MODULE.ABRA, 3),
                pokemon(MODULE.DUNSPARCE, 4),
            ],
            hand=[MODULE.POFFIN],
        )
        opponent = player(active=pokemon(900, 5, hp=200))
        options = [
            {"type": 7, "cardId": MODULE.POFFIN},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_current_attacker_energy_precedes_optional_poffin_setup(self) -> None:
        me = player(
            active=pokemon(MODULE.ABRA, 1),
            hand=[MODULE.POFFIN, MODULE.TELEPATH_ENERGY],
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {"type": 7, "cardId": MODULE.POFFIN},
            {
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 1,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [1])

    def test_dunsparce_uses_telepath_energy_to_open_basic_psychic_search(self) -> None:
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            hand=[MODULE.TELEPATH_ENERGY],
        )
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [
            {
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_active_dunsparce_uses_hilda_to_start_dudunsparce_route(self) -> None:
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            hand=[MODULE.HILDA, 900, 901],
        )
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [
            {"type": 7, "indexInArea": 0},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_active_alakazam_evolves_bench_abra_before_powerful_hand(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ALAKAZAM,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.KADABRA, 900, 901],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "type": 9,
                "cardId": MODULE.KADABRA,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_active_kadabra_evolves_bench_abra_before_kadabra_attack(self) -> None:
        active = pokemon(
            MODULE.KADABRA,
            1,
            energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            appear_this_turn=True,
        )
        me = player(
            active=active,
            bench=[pokemon(MODULE.ABRA, 2, energies=[])],
            hand=[MODULE.ALAKAZAM, MODULE.KADABRA, 900, 901],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "type": 9,
                "cardId": MODULE.KADABRA,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"type": 13, "attackId": 1071},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_active_dunsparce_evolves_bench_abra_before_ending(self) -> None:
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.KADABRA],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "type": 9,
                "cardId": MODULE.KADABRA,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

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

    def test_active_dunsparce_evolves_to_visible_dudunsparce_before_ending(self) -> None:
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.DUDUNSPARCE],
        )
        opponent = player(active=pokemon(900, 3, hp=220))
        options = [
            {
                "type": 9,
                "cardId": MODULE.DUDUNSPARCE,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_ready_alakazam_evolves_bench_dunsparce_before_powerful_hand(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ALAKAZAM,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.DUDUNSPARCE],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "type": 9,
                "cardId": MODULE.DUDUNSPARCE,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_non_attacking_active_preserves_visible_dudunsparce_engine(self) -> None:
        me = player(
            active=pokemon(MODULE.FEZANDIPITI_EX, 1, appear_this_turn=True),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.DUDUNSPARCE],
        )
        opponent = player(active=pokemon(900, 3, hp=220))
        options = [
            {
                "type": 9,
                "cardId": MODULE.DUDUNSPARCE,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_active_dudunsparce_attaches_enriching_energy_before_run_away_draw(self) -> None:
        me = player(
            active=pokemon(MODULE.DUDUNSPARCE, 1),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.ENRICHING_ENERGY],
        )
        opponent = player(active=pokemon(900, 3, hp=220))
        options = [
            {
                "type": 8,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"type": 10, "inPlayArea": 4, "inPlayIndex": 0},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_active_dudunsparce_uses_run_away_draw_with_any_bench_successor(self) -> None:
        me = player(
            active=pokemon(MODULE.DUDUNSPARCE, 1),
            bench=[pokemon(MODULE.ABRA, 2)],
        )
        opponent = player(active=pokemon(900, 3, hp=220))
        options = [
            {"type": 10, "inPlayArea": 4, "inPlayIndex": 0},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

    def test_bench_dudunsparce_draws_before_active_alakazam_attacks(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ALAKAZAM,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            bench=[pokemon(MODULE.DUDUNSPARCE, 2)],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"type": 10, "inPlayArea": 5, "inPlayIndex": 0},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

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

    def test_nighttime_mine_is_played_before_a_legal_attack(self) -> None:
        me = player(
            active=pokemon(
                MODULE.ALAKAZAM,
                1,
                energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            ),
            hand=[MODULE.NIGHTTIME_MINE],
        )
        opponent = player(active=pokemon(900, 2, hp=20))
        options = [
            {"type": 7, "cardId": MODULE.NIGHTTIME_MINE},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

if __name__ == "__main__":
    unittest.main()
