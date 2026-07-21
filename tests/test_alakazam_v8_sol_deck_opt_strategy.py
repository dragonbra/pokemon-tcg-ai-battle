import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "work" / "alakazam_v8_current" / "main.py"


def load_main():
    spec = importlib.util.spec_from_file_location("alakazam_v8_sol_deck_opt_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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
TELEPATH_ENERGY = 19
MIST_ENERGY = 11
ENHANCED_HAMMER = 1081
LANAS_AID = 1184
HILDA = 1225
POKE_PAD = 1152
SACRED_ASH = 1129
NIGHTTIME_MINE = 1266
POWERFUL_HAND = 1072
TRADING_PLACES = 423


def card(card_id, *, serial=None, energies=None, energy_cards=None, pre_evolution=None):
    result = {"id": card_id}
    if serial is not None:
        result["serial"] = serial
    if energies is not None:
        result["energies"] = list(energies)
    if energy_cards is not None:
        result["energyCards"] = [card(item) for item in energy_cards]
    if pre_evolution is not None:
        result["preEvolution"] = [card(item) for item in pre_evolution]
    return result


def state(*, active, bench=None, hand=None, discard=None, deck_count=30, prize_count=4):
    hand = list(hand or [])
    return {
        "active": [active] if active else [],
        "bench": list(bench or []),
        "hand": [card(item) for item in hand],
        "handCount": len(hand),
        "deck": [],
        "deckCount": deck_count,
        "discard": [card(item) for item in (discard or [])],
        "prize": [card(1) for _ in range(prize_count)],
        "benchMax": 5,
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
        },
        "logs": list(logs or []),
    }


def play_option(card_id):
    return {"type": 7, "cardId": card_id}


def attack_option(attack_id=POWERFUL_HAND):
    return {"type": 13, "attackId": attack_id}


class V8SolDeckOptStrategyTest(unittest.TestCase):
    def setUp(self):
        main._TURN_MEMORY.reset()
        main._EFFECT_PROGRESS.clear()

    def test_fezandipiti_draw_is_selected_after_previous_turn_knockout(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[5]),
            hand=[FEZANDIPITI_EX, 1000, 1001, 1002, 1003, 1004, 1005],
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 100
        logs = [{"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3}]
        obs = observation(
            player=player,
            opponent=opponent,
            options=[play_option(FEZANDIPITI_EX), attack_option()],
            logs=logs,
        )

        self.assertEqual(main._main_action(obs), [0])

    def test_fezandipiti_draw_is_not_suppressed_by_rule_box_prize_risk(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[5]),
            bench=[card(ALAKAZAM, serial=2, energies=[5])],
            hand=[FEZANDIPITI_EX, 1000, 1001, 1002, 1003, 1004, 1005],
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=2)
        opponent["active"][0]["hp"] = 100
        logs = [{"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3}]
        obs = observation(
            player=player,
            opponent=opponent,
            options=[play_option(FEZANDIPITI_EX), attack_option()],
            logs=logs,
        )

        self.assertEqual(main._main_action(obs), [0])

    def test_nighttime_mine_is_played_before_a_nonterminal_attack(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[5]),
            hand=[NIGHTTIME_MINE],
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 100
        obs = observation(
            player=player,
            opponent=opponent,
            options=[play_option(NIGHTTIME_MINE), attack_option()],
        )

        self.assertEqual(main._main_action(obs), [0])

    def test_enhanced_hammer_is_used_for_any_legal_special_energy_target(self):
        player = state(
            active=card(ALAKAZAM, serial=1, energies=[5]),
            hand=[ENHANCED_HAMMER],
            prize_count=4,
        )
        opponent = state(
            active=card(30, serial=90, energies=[6], energy_cards=[TELEPATH_ENERGY]),
            prize_count=6,
        )
        obs = observation(
            player=player,
            opponent=opponent,
            options=[play_option(ENHANCED_HAMMER), attack_option()],
        )

        self.assertEqual(main._main_action(obs), [0])

    def test_dudunsparce_draw_is_used_as_a_handoff_before_an_attack(self):
        player = state(
            active=card(DUDUNSPARCE, serial=1, energies=[5, 5, 5]),
            bench=[card(ALAKAZAM, serial=2, energies=[5])],
            hand=[],
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 80
        options = [
            {"type": 10, "cardId": DUDUNSPARCE, "inPlayArea": 4, "index": 0},
            attack_option(),
            {"type": 14},
        ]
        obs = observation(player=player, opponent=opponent, options=options)

        self.assertEqual(main._main_action(obs), [0])

    def test_trading_places_is_never_selected_as_a_switch(self):
        player = state(
            active=card(DUNSPARCE, serial=1),
            bench=[card(ALAKAZAM, serial=2, energies=[5])],
            prize_count=4,
        )
        obs = observation(
            player=player,
            options=[attack_option(TRADING_PLACES), {"type": 14}],
        )

        self.assertEqual(main._main_action(obs), [1])

    def test_active_abra_does_not_submit_teleportation_attack_when_end_is_available(self):
        player = state(
            active=card(ABRA, serial=1, energies=[5]),
            prize_count=4,
        )
        obs = observation(
            player=player,
            options=[attack_option(1070), {"type": 14}],
        )

        self.assertEqual(main._main_action(obs), [1])

    def test_active_dunsparce_keeps_a_dudunsparce_bridge_over_a_low_value_ko(self):
        player = state(
            active=card(DUNSPARCE, serial=1),
            bench=[card(ALAKAZAM, serial=2, energies=[5])],
            hand=[HILDA],
            prize_count=4,
        )
        opponent = state(active=card(30, serial=90), prize_count=6)
        opponent["active"][0]["hp"] = 10
        obs = observation(
            player=player,
            opponent=opponent,
            options=[play_option(HILDA), attack_option(424)],
        )

        self.assertEqual(main._main_action(obs), [0])

    def test_first_turn_poke_pad_is_preserved_when_the_board_has_setup_basics(self):
        player = state(
            active=card(ABRA, serial=1),
            bench=[card(ABRA, serial=2), card(DUNSPARCE, serial=3)],
            hand=[1152],
            prize_count=4,
        )
        obs = observation(
            player=player,
            options=[play_option(1152), {"type": 14}],
            turn=1,
        )

        self.assertEqual(main._main_action(obs), [1])

    def test_first_turn_poke_pad_builds_an_abra_anchor_when_active_is_dunsparce(self):
        player = state(
            active=card(DUNSPARCE, serial=1),
            hand=[POKE_PAD],
            prize_count=4,
        )
        select = {
            "type": 0,
            "context": 9,
            "effect": {"id": POKE_PAD},
            "deck": [card(DUNSPARCE), card(ABRA)],
            "option": [
                {"area": 1, "index": 0},
                {"area": 1, "index": 1},
            ],
            "minCount": 1,
            "maxCount": 1,
        }
        obs = observation(player=player, select=select, turn=1)

        self.assertEqual(main._select_effect(obs), [1])

    def test_sacred_ash_selects_attack_line_in_stage_order_and_fills_selection(self):
        discard = [ALAKAZAM, KADABRA, ABRA, DUDUNSPARCE, DUNSPARCE, SHAYMIN]
        player = state(active=card(SHAYMIN, serial=1), discard=discard, prize_count=4)
        select = {
            "type": 0,
            "context": 9,
            "effect": {"id": SACRED_ASH},
            "option": [
                {"area": 3, "index": index}
                for index in range(len(discard))
            ],
            "minCount": 1,
            "maxCount": 5,
        }
        obs = observation(player=player, select=select)

        self.assertEqual(main._select_effect(obs), [2, 1, 0, 3, 4])

    def test_lanas_aid_recovers_all_available_attack_line_pokemon_before_energy(self):
        discard = [ABRA, KADABRA, ALAKAZAM, BASIC_PSYCHIC]
        player = state(active=card(SHAYMIN, serial=1), discard=discard, prize_count=4)
        select = {
            "type": 0,
            "context": 9,
            "effect": {"id": LANAS_AID},
            "option": [
                {"area": 3, "index": index}
                for index in range(len(discard))
            ],
            "minCount": 1,
            "maxCount": 3,
        }
        obs = observation(player=player, select=select)

        self.assertEqual(main._select_effect(obs), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
