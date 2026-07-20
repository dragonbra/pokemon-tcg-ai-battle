import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "work" / "alakazam_v8_current" / "main.py"
sys.path.insert(0, str(MAIN_PATH.parent))
SPEC = importlib.util.spec_from_file_location("alakazam_v8_luna_deck_opt_main", MAIN_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {MAIN_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

NIGHTTIME_MINE = 1266


def pokemon(card_id, serial, *, hp=100, energies=None, energy_cards=None):
    energy_ids = list(energy_cards or [])
    return {
        "id": card_id,
        "serial": serial,
        "hp": hp,
        "maxHp": hp,
        "energies": list(energies or []),
        "energyCards": [
            {"id": energy_id, "serial": serial * 100 + index}
            for index, energy_id in enumerate(energy_ids)
        ],
        "appearThisTurn": False,
        "preEvolution": [],
        "tools": [],
    }


def player(*, active=None, bench=None, hand=None, deck_count=30, discard=None):
    hand = list(hand or [])
    discard = list(discard or [])
    return {
        "active": [active] if active else [],
        "bench": list(bench or []),
        "benchMax": 5,
        "hand": [{"id": card_id, "serial": 9000 + index} for index, card_id in enumerate(hand)],
        "handCount": len(hand),
        "deckCount": deck_count,
        "discard": [
            {"id": card_id, "serial": 7000 + index}
            for index, card_id in enumerate(discard)
        ],
        "prize": [{} for _ in range(6)],
    }


def base_obs(me, opponent, options, *, turn=5, logs=None):
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [me, opponent],
            "supporterPlayed": False,
            "energyAttached": False,
            "retreated": False,
        },
        "logs": list(logs or []),
        "select": {
            "type": 0,
            "context": 0,
            "option": options,
            "minCount": 1,
            "maxCount": 1,
        },
    }


class V8CurrentStrategyTests(unittest.TestCase):
    def setUp(self):
        MODULE._EFFECT_PROGRESS.clear()
        MODULE._TURN_MEMORY.reset()

    def test_fezandipiti_draw_precedes_nonterminal_powerful_hand_after_ko(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.FEZANDIPITI_EX, 900, 901, 902],
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"type": 7, "cardId": MODULE.FEZANDIPITI_EX},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]
        logs = [{"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3}]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options, logs=logs)), [0])

    def test_fezandipiti_does_not_delay_the_last_prize_attack(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.FEZANDIPITI_EX, 900, 901],
        )
        me["prize"] = [{}]
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"type": 7, "cardId": MODULE.FEZANDIPITI_EX},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]
        logs = [{"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3}]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options, logs=logs)), [1])

    def test_dawn_precedes_nonterminal_powerful_hand_for_bench_abra_evolution(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.DAWN, MODULE.KADABRA, 900, 901],
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"type": 7, "cardId": MODULE.DAWN},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_trading_places_is_not_used_when_only_forbidden_attack_remains(self):
        me = player(active=pokemon(MODULE.DUNSPARCE, 1))
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [{"type": 13, "attackId": MODULE.TRADING_PLACES_ATTACK}]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [])

    def test_run_away_draw_hands_active_position_to_ready_bench_alakazam(self):
        me = player(
            active=pokemon(MODULE.DUDUNSPARCE, 1),
            bench=[pokemon(MODULE.ALAKAZAM, 2, energies=[MODULE.PSYCHIC_ENERGY_TYPE])],
            hand=[900, 901, 902],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "type": 10,
                "cardId": None,
                "area": 4,
                "indexInArea": 0,
            },
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_enhanced_hammer_precedes_nonterminal_powerful_hand(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.ENHANCED_HAMMER, 900, 901],
        )
        opponent = player(
            active=pokemon(900, 2, hp=200, energy_cards=[MODULE.TELEPATH_ENERGY])
        )
        options = [
            {"type": 7, "cardId": MODULE.ENHANCED_HAMMER},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_hammer_prefers_benched_protective_energy_over_active_other_special_energy(self):
        me = player(active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]))
        opponent_active = pokemon(900, 2, energy_cards=[MODULE.TELEPATH_ENERGY])
        opponent_bench = pokemon(901, 3, energy_cards=[MODULE.MIST_ENERGY])
        opponent = player(active=opponent_active, bench=[opponent_bench])
        current = base_obs(me, opponent, [])['current']
        options = [
            {
                "type": 3,
                "playerIndex": 1,
                "area": 4,
                "indexInArea": 0,
                "energyIndex": 0,
            },
            {
                "type": 3,
                "playerIndex": 1,
                "area": 5,
                "indexInArea": 0,
                "energyIndex": 0,
            },
        ]

        self.assertEqual(MODULE._choose_energy_option(options, current, me), [1])

    def test_lanas_aid_prefers_multiple_abra_line_pokemon_over_energy(self):
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            hand=[MODULE.LANAS_AID, 900],
            discard=[MODULE.ABRA, MODULE.KADABRA, MODULE.ALAKAZAM, MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [
            {"type": 3, "area": 3, "index": 0},
            {"type": 3, "area": 3, "index": 1},
            {"type": 3, "area": 3, "index": 2},
            {"type": 3, "area": 3, "index": 3},
        ]
        obs = base_obs(me, opponent, options)
        obs["select"] = {
            "type": 1,
            "context": 7,
            "effect": {"id": MODULE.LANAS_AID, "serial": 1},
            "minCount": 1,
            "maxCount": 3,
            "option": options,
        }

        chosen = MODULE._select_effect(obs)

        self.assertEqual(len(chosen), 3)
        self.assertEqual(
            {
                MODULE._option_card_id(options[index], obs["select"], obs["current"])
                for index in chosen
            },
            {MODULE.ABRA, MODULE.KADABRA, MODULE.ALAKAZAM},
        )

    def test_lanas_aid_recovers_pokemon_even_without_discard_energy(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.LANAS_AID, 900, 901],
            discard=[MODULE.ABRA],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"type": 7, "cardId": MODULE.LANAS_AID},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_sacred_ash_uses_all_available_slots_for_recovery_chain(self):
        me = player(active=pokemon(MODULE.DUNSPARCE, 1), discard=[
            MODULE.ABRA,
            MODULE.KADABRA,
            MODULE.ALAKAZAM,
            MODULE.ABRA,
            MODULE.DUNSPARCE,
        ])
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [
            {"type": 3, "area": 3, "index": index}
            for index in range(5)
        ]
        obs = base_obs(me, opponent, options)
        obs["select"] = {
            "type": 1,
            "context": 9,
            "effect": {"id": MODULE.SACRED_ASH, "serial": 2},
            "minCount": 1,
            "maxCount": 5,
            "option": options,
        }

        self.assertEqual(len(MODULE._select_effect(obs)), 5)

    def test_sacred_ash_deprioritizes_kadabra_when_rare_candy_is_available(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1),
            hand=[MODULE.RARE_CANDY],
            discard=[MODULE.ABRA, MODULE.KADABRA, MODULE.ALAKAZAM],
        )
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [
            {"type": 3, "area": 3, "index": 0},
            {"type": 3, "area": 3, "index": 1},
            {"type": 3, "area": 3, "index": 2},
        ]
        obs = base_obs(me, opponent, options)
        obs["select"] = {
            "type": 1,
            "context": 9,
            "effect": {"id": MODULE.SACRED_ASH, "serial": 10},
            "minCount": 1,
            "maxCount": 5,
            "option": options,
        }

        chosen = MODULE._select_effect(obs)

        chosen_ids = [
            MODULE._option_card_id(options[index], obs["select"], obs["current"])
            for index in chosen
        ]
        self.assertEqual(chosen_ids[:3], [MODULE.ABRA, MODULE.ALAKAZAM, MODULE.KADABRA])

    def test_first_turn_pokepad_waits_after_two_abra_and_dunsparce_are_ready(self):
        me = player(
            active=pokemon(MODULE.ABRA, 1),
            bench=[pokemon(MODULE.ABRA, 2), pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.POKE_PAD, 900],
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        options = [
            {"type": 7, "cardId": MODULE.POKE_PAD},
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options, turn=1)), [1])

    def test_pokepad_finds_dudunsparce_for_active_dunsparce_handoff(self):
        me = player(active=pokemon(MODULE.DUNSPARCE, 1), hand=[MODULE.POKE_PAD])
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [
            {"type": 1, "cardId": MODULE.KADABRA},
            {"type": 1, "cardId": MODULE.DUDUNSPARCE},
        ]
        obs = base_obs(me, opponent, options)
        obs["select"] = {
            "type": 1,
            "context": 7,
            "effect": {"id": MODULE.POKE_PAD, "serial": 9},
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        }

        self.assertEqual(MODULE._select_effect(obs), [1])

    def test_dawn_finds_dunsparce_when_no_dunsparce_is_in_play(self):
        me = player(active=pokemon(MODULE.ALAKAZAM, 1), hand=[MODULE.DAWN])
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [
            {"type": 1, "cardId": MODULE.ABRA},
            {"type": 1, "cardId": MODULE.DUNSPARCE},
        ]
        obs = base_obs(me, opponent, options)
        obs["select"] = {
            "type": 1,
            "context": 7,
            "effect": {"id": MODULE.DAWN, "serial": 3},
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        }

        self.assertEqual(MODULE._select_effect(obs), [1])

    def test_dawn_finds_dudunsparce_for_active_dunsparce_handoff(self):
        me = player(active=pokemon(MODULE.DUNSPARCE, 1), hand=[MODULE.DAWN])
        opponent = player(active=pokemon(900, 2, hp=220))
        options = [
            {"type": 1, "cardId": MODULE.KADABRA},
            {"type": 1, "cardId": MODULE.DUDUNSPARCE},
        ]
        obs = base_obs(me, opponent, options)
        obs["select"] = {
            "type": 1,
            "context": 7,
            "effect": {"id": MODULE.DAWN, "serial": 4},
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        }
        MODULE._EFFECT_PROGRESS[4] = 1

        self.assertEqual(MODULE._select_effect(obs), [1])

    def test_nighttime_mine_precedes_nonterminal_attack(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[NIGHTTIME_MINE, 900, 901],
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {"type": 7, "cardId": NIGHTTIME_MINE},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_nighttime_mine_precedes_even_bench_insurance_setup(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[NIGHTTIME_MINE, MODULE.POFFIN, 900],
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {"type": 7, "cardId": NIGHTTIME_MINE},
            {"type": 7, "cardId": MODULE.POFFIN},
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])


if __name__ == "__main__":
    unittest.main()
