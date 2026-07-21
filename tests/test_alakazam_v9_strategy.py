import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "work" / "alakazam_v9" / "main.py"


def load_main():
    module_name = "alakazam_v9_strategy_main"
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


main = load_main()

ABRA = 741
KADABRA = 742
ALAKAZAM = 743
DUNSPARCE = 305
DUDUNSPARCE = 66
FEZANDIPITI_EX = 140
SHAYMIN = 343
BASIC_PSYCHIC = 5
TELEPATH_ENERGY = 1266
RARE_CANDY = 1079
LANAS_AID = 1184
BOSS_ORDERS = 1182
POFFIN = 1086
NIGHT_STRETCHER = 1097
HILDA = 1225
NIGHTTIME_MINE = 1266
POWERFUL_HAND = 1072


def card(card_id, *, serial=None, energies=None, energy_cards=None, pre_evolution=None):
    attached_energies = list(energies or [])
    attached_cards = list(energy_cards if energy_cards is not None else attached_energies)
    result = {
        "id": card_id,
        "energies": attached_energies,
        "energyCards": [{"id": item} for item in attached_cards],
        "preEvolution": [{"id": item} for item in (pre_evolution or [])],
        "tools": [],
        "appearThisTurn": False,
    }
    if serial is not None:
        result["serial"] = serial
    return result


def state(*, active, bench=None, hand=None, discard=None, deck_count=30, prize_count=4):
    hand_cards = [card(item) for item in (hand or [])]
    return {
        "active": [active] if active else [],
        "bench": list(bench or []),
        "benchMax": 5,
        "hand": hand_cards,
        "handCount": len(hand_cards),
        "deck": [],
        "deckCount": deck_count,
        "discard": [card(item) for item in (discard or [])],
        "prize": [card(1) for _ in range(prize_count)],
    }


def observation(*, player, opponent=None, options=None, logs=None, turn=3, select=None):
    opponent = opponent or state(active=card(30, serial=90), prize_count=6)
    current = {
        "turn": turn,
        "yourIndex": 0,
        "firstPlayer": 0,
        "players": [player, opponent],
        "supporterPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "logs": list(logs or []),
    }
    return {
        "current": current,
        "select": select
        or {
            "type": 0,
            "context": 0,
            "option": list(options or []),
            "minCount": 1,
            "maxCount": 1,
        },
        "logs": list(logs or []),
    }


def play_option(card_id):
    return {"type": 7, "cardId": card_id}


def attack_option(attack_id=POWERFUL_HAND):
    return {"type": 13, "attackId": attack_id}


def energy_option(energy_id, *, area, index):
    return {
        "type": 8,
        "cardId": energy_id,
        "area": 2,
        "inPlayArea": area,
        "inPlayIndex": index,
    }


def recovery_select(effect_id, cards, *, min_count, max_count):
    return {
        "type": 1,
        "context": 7,
        "effect": {"id": effect_id, "serial": 100},
        "option": [
            {"type": 3, "area": 3, "index": index}
            for index, _ in enumerate(cards)
        ],
        "minCount": min_count,
        "maxCount": max_count,
    }


def deck_search_select(effect_id, cards, *, min_count=1, max_count=1):
    return {
        "type": 1,
        "context": 7,
        "effect": {"id": effect_id, "serial": 101},
        "deck": [card(item) for item in cards],
        "option": [
            {"type": 3, "area": 1, "index": index}
            for index, _ in enumerate(cards)
        ],
        "minCount": min_count,
        "maxCount": max_count,
    }


class V9StrategyTest(unittest.TestCase):
    def setUp(self):
        main._TURN_MEMORY.reset()
        main._EFFECT_PROGRESS.clear()

    def test_field_counters(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(ABRA, serial=2), card(DUNSPARCE, serial=3)],
        )

        self.assertEqual(main._abra_series_count(player), 2)
        self.assertEqual(main._dunsparce_series_count(player), 1)
        self.assertEqual(main._field_gaps(player), (0, 0))
        self.assertEqual(main._reserve_bench_slots(player), 0)

    def test_dudunsparce_satisfies_the_dunsparce_series_gap(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(DUDUNSPARCE, serial=2)],
        )

        self.assertEqual(main._abra_series_count(player), 1)
        self.assertEqual(main._dunsparce_series_count(player), 1)
        self.assertEqual(main._field_gaps(player), (1, 0))
        self.assertEqual(main._reserve_bench_slots(player), 1)

    def test_turn_plan_detects_direct_active_alakazam_attack_route(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(ABRA, serial=2), card(DUNSPARCE, serial=3)],
        )
        options = [attack_option(), {"type": 14}]
        obs = observation(player=player, options=options)

        plan = main._build_turn_plan(obs["current"], player, options, obs["select"])

        self.assertTrue(plan.attack_now)
        self.assertIsNotNone(plan.attack_route)
        self.assertIsNone(plan.reserved_energy_id)
        self.assertIsNone(plan.reserved_energy_target_serial)
        self.assertEqual(plan.abra_count, 2)
        self.assertEqual(plan.dunsparce_count, 1)
        self.assertEqual(plan.reserved_bench_slots, 0)

    def test_terminal_boss_route_beats_direct_non_knockout_attack(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            hand=[BOSS_ORDERS, 900, 901],
            prize_count=2,
        )
        opponent = state(
            active=card(30, serial=90),
            bench=[card(FEZANDIPITI_EX, serial=91)],
            prize_count=6,
        )
        opponent["active"][0]["hp"] = 300
        opponent["bench"][0]["hp"] = 40
        options = [attack_option(), play_option(BOSS_ORDERS), {"type": 14}]
        obs = observation(player=player, opponent=opponent, options=options)

        plan = main._build_turn_plan(obs["current"], player, options, obs["select"])

        self.assertTrue(plan.terminal_attack)
        self.assertEqual(plan.reserved_supporter_id, BOSS_ORDERS)
        self.assertEqual(main._main_action(obs), [1])

    def test_switch_selection_chooses_ready_bench_alakazam(self):
        player = state(
            active=card(FEZANDIPITI_EX, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[
                card(ALAKAZAM, serial=2),
                card(ALAKAZAM, serial=3, energies=[BASIC_PSYCHIC]),
            ],
        )
        obs = observation(player=player)
        options = [
            {"area": 5, "inPlayArea": 5, "inPlayIndex": 0},
            {"area": 5, "inPlayArea": 5, "inPlayIndex": 1},
        ]

        self.assertEqual(main._choose_switch_option(options, obs["current"]), [1])

    def test_turn_plan_reserves_attach_before_retreat_option_exists(self):
        player = state(
            active=card(FEZANDIPITI_EX, serial=1),
            bench=[card(ALAKAZAM, serial=2, energies=[BASIC_PSYCHIC])],
            hand=[BASIC_PSYCHIC],
        )
        options = [
            energy_option(BASIC_PSYCHIC, area=4, index=0),
            {"type": 14},
        ]
        obs = observation(player=player, options=options)

        plan = main._build_turn_plan(obs["current"], player, options, obs["select"])

        self.assertEqual(plan.attack_route, "retreat_after_attach")
        self.assertEqual(plan.reserved_energy_id, BASIC_PSYCHIC)
        self.assertEqual(plan.reserved_energy_target_serial, 1)
        self.assertEqual(main._main_action(obs), [0])

    def test_turn_plan_does_not_attach_for_retreat_after_retreat_was_used(self):
        player = state(
            active=card(FEZANDIPITI_EX, serial=1),
            bench=[card(ALAKAZAM, serial=2, energies=[BASIC_PSYCHIC])],
            hand=[BASIC_PSYCHIC],
        )
        options = [energy_option(BASIC_PSYCHIC, area=4, index=0), {"type": 14}]
        obs = observation(player=player, options=options)
        obs["current"]["retreated"] = True

        plan = main._build_turn_plan(obs["current"], player, options, obs["select"])

        self.assertNotEqual(plan.attack_route, "retreat_after_attach")
        self.assertEqual(main._main_action(obs), [1])

    def test_turn_plan_does_not_attach_telepath_after_retreat_was_used(self):
        player = state(
            active=card(FEZANDIPITI_EX, serial=1),
            bench=[card(ALAKAZAM, serial=2, energies=[BASIC_PSYCHIC])],
            hand=[TELEPATH_ENERGY],
        )
        options = [energy_option(TELEPATH_ENERGY, area=4, index=0), {"type": 14}]
        obs = observation(player=player, options=options)
        obs["current"]["retreated"] = True

        self.assertEqual(main._main_action(obs), [1])

    def test_energy_selector_follows_turn_plan_active_target(self):
        player = state(
            active=card(ALAKAZAM, serial=1),
            bench=[card(KADABRA, serial=2)],
            hand=[BASIC_PSYCHIC],
        )
        energy_options = [
            energy_option(BASIC_PSYCHIC, area=4, index=0),
            energy_option(BASIC_PSYCHIC, area=5, index=0),
        ]
        options = [*energy_options, {"type": 14}]
        obs = observation(player=player, options=options)
        main._TURN_MEMORY.sync(obs["current"], player)
        plan = main._build_turn_plan(obs["current"], player, options, obs["select"])

        self.assertEqual(plan.attack_route, "attach_active")
        self.assertTrue(
            main._energy_option_matches_plan(energy_options[0], plan, obs["current"])
        )
        self.assertFalse(
            main._energy_option_matches_plan(energy_options[1], plan, obs["current"])
        )
        self.assertEqual(
            main._choose_energy_option(energy_options, obs["current"], player, plan),
            [0],
        )

    def test_terminal_powerful_hand_beats_future_bench_attachment(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(ABRA, serial=2)],
            hand=[BASIC_PSYCHIC],
            prize_count=1,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 20
        options = [
            energy_option(BASIC_PSYCHIC, area=5, index=0),
            attack_option(),
            {"type": 14},
        ]

        self.assertEqual(
            main._main_action(observation(player=player, opponent=opponent, options=options)),
            [1],
        )

    def test_exact_knockout_beats_anchor_play_that_reduces_powerful_hand_damage(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            hand=[ABRA, 900, 901],
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 60
        options = [play_option(ABRA), attack_option(), {"type": 14}]

        self.assertEqual(
            main._main_action(observation(player=player, opponent=opponent, options=options)),
            [1],
        )

    def test_exact_knockout_beats_poffin_that_reduces_powerful_hand_damage(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            hand=[POFFIN, 900, 901],
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 60
        options = [play_option(POFFIN), attack_option(), {"type": 14}]

        self.assertEqual(
            main._main_action(observation(player=player, opponent=opponent, options=options)),
            [1],
        )

    def test_exact_knockout_beats_nighttime_mine_that_reduces_powerful_hand_damage(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            hand=[NIGHTTIME_MINE, 900, 901],
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 60
        options = [play_option(NIGHTTIME_MINE), attack_option(), {"type": 14}]

        self.assertEqual(
            main._main_action(observation(player=player, opponent=opponent, options=options)),
            [1],
        )

    def test_retreat_without_ready_alakazam_stays_below_end(self):
        player = state(
            active=card(DUNSPARCE, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(ABRA, serial=2)],
        )
        options = [{"type": 12}, {"type": 14}]

        self.assertEqual(main._main_action(observation(player=player, options=options)), [1])

    def test_active_fezandipiti_gets_energy_for_ready_bench_alakazam_handoff(self):
        player = state(
            active=card(FEZANDIPITI_EX, serial=1),
            bench=[card(ALAKAZAM, serial=2, energies=[BASIC_PSYCHIC])],
            hand=[BASIC_PSYCHIC],
        )
        options = [
            energy_option(BASIC_PSYCHIC, area=4, index=0),
            energy_option(BASIC_PSYCHIC, area=5, index=0),
            attack_option(),
            {"type": 14},
        ]

        self.assertEqual(main._main_action(observation(player=player, options=options)), [0])

    def test_active_abra_attack_route_beats_bench_kadabra_attachment(self):
        player = state(
            active=card(ABRA, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(KADABRA, serial=2)],
            hand=[RARE_CANDY, ALAKAZAM, BASIC_PSYCHIC],
        )
        options = [
            energy_option(BASIC_PSYCHIC, area=5, index=0),
            {
                "type": 9,
                "cardId": ALAKAZAM,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            attack_option(),
            {"type": 14},
        ]

        self.assertEqual(main._main_action(observation(player=player, options=options)), [1])

    def test_lanas_aid_selects_abra_and_basic_psychic_for_missing_successor(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            discard=[ABRA, BASIC_PSYCHIC],
        )
        select = recovery_select(
            LANAS_AID,
            [ABRA, BASIC_PSYCHIC],
            min_count=1,
            max_count=2,
        )

        self.assertEqual(main._select_effect(observation(player=player, select=select)), [0, 1])

    def test_lanas_aid_does_not_fill_selection_with_unrelated_cards(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            discard=[ABRA, 999],
        )
        select = recovery_select(
            LANAS_AID,
            [ABRA, 999],
            min_count=1,
            max_count=2,
        )

        self.assertEqual(main._select_effect(observation(player=player, select=select)), [0])

    def test_lanas_aid_is_played_for_visible_abra_energy_successor(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            hand=[LANAS_AID],
            discard=[ABRA, BASIC_PSYCHIC],
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 300
        options = [play_option(LANAS_AID), attack_option(), {"type": 14}]

        self.assertEqual(
            main._main_action(observation(player=player, opponent=opponent, options=options)),
            [0],
        )

    def test_hilda_attack_route_reserves_supporter_over_lanas_aid(self):
        player = state(
            active=card(ABRA, serial=1, energies=[BASIC_PSYCHIC]),
            hand=[HILDA, RARE_CANDY, LANAS_AID],
            discard=[ABRA, BASIC_PSYCHIC],
        )
        options = [play_option(LANAS_AID), play_option(HILDA), {"type": 14}]

        self.assertEqual(main._main_action(observation(player=player, options=options)), [1])

    def test_field_gap_prioritizes_abra_before_dunsparce_and_other_pokemon(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            hand=[ABRA, DUNSPARCE, FEZANDIPITI_EX],
        )
        options = [
            play_option(DUNSPARCE),
            play_option(ABRA),
            play_option(FEZANDIPITI_EX),
            {"type": 14},
        ]

        self.assertEqual(main._main_action(observation(player=player, options=options)), [1])

    def test_field_gap_prioritizes_dunsparce_after_two_abra_series(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(ABRA, serial=2)],
            hand=[ABRA, DUNSPARCE, FEZANDIPITI_EX],
        )
        options = [
            play_option(ABRA),
            play_option(DUNSPARCE),
            play_option(FEZANDIPITI_EX),
            {"type": 14},
        ]

        self.assertEqual(main._main_action(observation(player=player, options=options)), [1])

    def test_poffin_searches_dunsparce_once_two_abra_series_are_in_play(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(ABRA, serial=2)],
        )
        select = deck_search_select(POFFIN, [ABRA, DUNSPARCE])

        self.assertEqual(main._select_effect(observation(player=player, select=select)), [1])

    def test_poffin_batch_updates_field_gaps_between_two_selections(self):
        player = state(active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]))
        select = deck_search_select(
            POFFIN,
            [ABRA, ABRA, DUNSPARCE],
            min_count=1,
            max_count=2,
        )

        self.assertEqual(main._select_effect(observation(player=player, select=select)), [0, 2])

    def test_night_stretcher_recovers_abra_before_dunsparce_for_field_gap(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            discard=[DUNSPARCE, ABRA],
        )
        select = recovery_select(
            NIGHT_STRETCHER,
            [DUNSPARCE, ABRA],
            min_count=1,
            max_count=1,
        )

        self.assertEqual(main._select_effect(observation(player=player, select=select)), [1])

    def test_night_stretcher_recovers_energy_for_current_active_attack_route(self):
        player = state(
            active=card(ALAKAZAM, serial=1),
            discard=[ABRA, BASIC_PSYCHIC],
        )
        select = recovery_select(
            NIGHT_STRETCHER,
            [ABRA, BASIC_PSYCHIC],
            min_count=1,
            max_count=1,
        )

        self.assertEqual(main._select_effect(observation(player=player, select=select)), [1])

    def test_night_stretcher_keeps_abra_first_when_active_is_already_ready(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            discard=[ABRA, BASIC_PSYCHIC],
        )
        select = recovery_select(
            NIGHT_STRETCHER,
            [ABRA, BASIC_PSYCHIC],
            min_count=1,
            max_count=1,
        )

        self.assertEqual(main._select_effect(observation(player=player, select=select)), [0])

    def test_night_stretcher_recovers_dunsparce_before_future_energy(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(ABRA, serial=2)],
            discard=[DUNSPARCE, BASIC_PSYCHIC],
        )
        select = recovery_select(
            NIGHT_STRETCHER,
            [DUNSPARCE, BASIC_PSYCHIC],
            min_count=1,
            max_count=1,
        )

        self.assertEqual(main._select_effect(observation(player=player, select=select)), [0])

    def test_legal_fezandipiti_ability_is_used_without_local_ko_log(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[
                card(ABRA, serial=2),
                card(DUNSPARCE, serial=3),
                card(FEZANDIPITI_EX, serial=4),
            ],
            hand=[900, 901],
            deck_count=30,
        )
        options = [
            {
                "type": 10,
                "cardId": FEZANDIPITI_EX,
                "inPlayArea": 5,
                "inPlayIndex": 2,
            },
            attack_option(),
            {"type": 14},
        ]

        self.assertEqual(main._main_action(observation(player=player, options=options)), [0])

    def test_fezandipiti_ability_is_blocked_by_low_deck_guard(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[
                card(ABRA, serial=2),
                card(DUNSPARCE, serial=3),
                card(FEZANDIPITI_EX, serial=4),
            ],
            hand=[900, 901],
            deck_count=12,
        )
        options = [
            {
                "type": 10,
                "cardId": FEZANDIPITI_EX,
                "inPlayArea": 5,
                "inPlayIndex": 2,
            },
            attack_option(),
            {"type": 14},
        ]

        self.assertEqual(main._main_action(observation(player=player, options=options)), [1])

    def test_terminal_attack_beats_legal_fezandipiti_ability(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[
                card(ABRA, serial=2),
                card(DUNSPARCE, serial=3),
                card(FEZANDIPITI_EX, serial=4),
            ],
            hand=[900, 901],
            deck_count=30,
            prize_count=1,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 40
        options = [
            {
                "type": 10,
                "cardId": FEZANDIPITI_EX,
                "inPlayArea": 5,
                "inPlayIndex": 2,
            },
            attack_option(),
            {"type": 14},
        ]

        self.assertEqual(
            main._main_action(observation(player=player, opponent=opponent, options=options)),
            [1],
        )

    def test_nonterminal_knockout_allows_safe_fezandipiti_draw_before_attack(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[
                card(ABRA, serial=2),
                card(DUNSPARCE, serial=3),
                card(FEZANDIPITI_EX, serial=4),
            ],
            hand=[900, 901],
            deck_count=30,
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 40
        options = [
            {
                "type": 10,
                "cardId": FEZANDIPITI_EX,
                "inPlayArea": 5,
                "inPlayIndex": 2,
            },
            attack_option(),
            {"type": 14},
        ]

        self.assertEqual(
            main._main_action(observation(player=player, opponent=opponent, options=options)),
            [0],
        )

    def test_fezandipiti_hand_play_does_not_consume_reserved_last_bench_slot(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[
                card(SHAYMIN, serial=2),
                card(FEZANDIPITI_EX, serial=3),
                card(SHAYMIN, serial=4),
                card(FEZANDIPITI_EX, serial=5),
            ],
            hand=[FEZANDIPITI_EX],
        )
        logs = [{"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3}]
        options = [play_option(FEZANDIPITI_EX), {"type": 14}]

        self.assertEqual(
            main._main_action(observation(player=player, options=options, logs=logs)),
            [1],
        )

    def test_fezandipiti_hand_play_uses_real_ko_trigger_when_anchors_are_safe(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
            bench=[card(ABRA, serial=2), card(DUNSPARCE, serial=3)],
            hand=[FEZANDIPITI_EX],
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 300
        logs = [{"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3}]
        options = [play_option(FEZANDIPITI_EX), attack_option(), {"type": 14}]

        self.assertEqual(
            main._main_action(
                observation(player=player, opponent=opponent, options=options, logs=logs)
            ),
            [0],
        )


if __name__ == "__main__":
    unittest.main()
