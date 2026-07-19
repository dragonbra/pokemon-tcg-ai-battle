import importlib.util
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "submission" / "alakazam_v6" / "main.py"
SPEC = importlib.util.spec_from_file_location("alakazam_v6_main", MAIN_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {MAIN_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def pokemon(card_id, serial, *, hp=100, max_hp=None, energies=None, energy_cards=None, appear=False):
    return {
        "id": card_id,
        "serial": serial,
        "hp": hp,
        "maxHp": max_hp if max_hp is not None else hp,
        "energies": list(energies or []),
        "energyCards": [{"id": energy_id, "serial": serial * 100 + i} for i, energy_id in enumerate(energy_cards or [])],
        "appearThisTurn": appear,
        "preEvolution": [],
        "tools": [],
    }


def base_obs(player, opponent, options, *, turn=3):
    current = {
        "turn": turn,
        "yourIndex": 0,
        "players": [player, opponent],
        "supporterPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "result": None,
    }
    return {
        "current": current,
        "logs": [],
        "select": {"type": 0, "context": 0, "option": options, "minCount": 1, "maxCount": 1},
    }


def player(*, active=None, bench=None, hand=None, deck_count=30, prize_count=6, hand_count=None):
    hand = list(hand or [])
    return {
        "active": [active] if active else [],
        "bench": list(bench or []),
        "benchMax": 5,
        "hand": [{"id": card_id, "serial": 9000 + i} for i, card_id in enumerate(hand)],
        "handCount": len(hand) if hand_count is None else hand_count,
        "deckCount": deck_count,
        "discard": [],
        "prize": [None] * prize_count,
    }


class V6StrategyTests(unittest.TestCase):
    def setUp(self):
        MODULE._EFFECT_PROGRESS.clear()
        MODULE._TURN_MEMORY.reset()

    def test_trading_places_is_never_preferred_over_end(self):
        active = pokemon(MODULE.DUNSPARCE, 1, energies=[5], energy_cards=[5])
        ready = pokemon(MODULE.ALAKAZAM, 2, energies=[5], energy_cards=[5])
        me = player(active=active, bench=[ready], hand=[], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 13, "attackId": MODULE.TRADING_PLACES_ATTACK},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_ready_bench_alakazam_uses_retreat_instead_of_trading_places(self):
        active = pokemon(MODULE.DUNSPARCE, 1, energies=[5], energy_cards=[5])
        ready = pokemon(MODULE.ALAKAZAM, 2, energies=[5], energy_cards=[5])
        me = player(active=active, bench=[ready], hand=[], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 13, "attackId": MODULE.TRADING_PLACES_ATTACK},
            {"type": 12},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_retreat_switch_prefers_the_energized_alakazam_target(self):
        active = pokemon(MODULE.DUNSPARCE, 1, energies=[5], energy_cards=[5])
        unready = pokemon(MODULE.ALAKAZAM, 2)
        ready = pokemon(MODULE.ALAKAZAM, 3, energies=[5], energy_cards=[5])
        me = player(active=active, bench=[unready, ready], hand=[], deck_count=30)
        opp = player(active=pokemon(900, 4, hp=200, max_hp=200))
        current = base_obs(me, opp, []).get("current")
        options = [
            {"type": 3, "playerIndex": 0, "area": 5, "index": 0},
            {"type": 3, "playerIndex": 0, "area": 5, "index": 1},
        ]
        self.assertEqual(MODULE._choose_switch_option(options, current), [1])

    def test_active_kadabra_natural_evolution_precedes_non_ko_attack(self):
        active = pokemon(MODULE.KADABRA, 1, hp=80, energies=[5], energy_cards=[5])
        me = player(active=active, hand=[MODULE.ALAKAZAM], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 9, "cardId": MODULE.ALAKAZAM, "inPlayArea": 4, "inPlayIndex": 0},
            {"type": 13, "attackId": 1},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])

    def test_bench_abra_evolves_after_active_alakazam_is_ready(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        bench_abra = pokemon(MODULE.ABRA, 2, hp=50, max_hp=50)
        me = player(active=active, bench=[bench_abra], hand=[], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 9, "cardId": MODULE.KADABRA, "inPlayArea": 5, "inPlayIndex": 0},
            {"type": 13, "attackId": 1},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])

    def test_required_bench_evolution_stays_before_a_lethal_attack(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        bench_abra = pokemon(MODULE.ABRA, 2, hp=50, max_hp=50)
        me = player(active=active, bench=[bench_abra], hand_count=5, deck_count=30)
        opp = player(active=pokemon(900, 3, hp=80, max_hp=80))
        options = [
            {"type": 9, "cardId": MODULE.KADABRA, "inPlayArea": 5, "inPlayIndex": 0},
            {"type": 13, "attackId": 1},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])

    def test_newly_evolved_bench_kadabra_draws_before_a_knockout_attack(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        bench_abra = pokemon(MODULE.ABRA, 2, hp=50, max_hp=50)
        me = player(active=active, bench=[bench_abra], hand=[], deck_count=30, hand_count=5)
        opp = player(active=pokemon(900, 3, hp=80, max_hp=80))
        evolve = [
            {
                "type": 9,
                "cardId": MODULE.KADABRA,
                "inPlayArea": 5,
                "inPlayIndex": 0,
                "serial": 200,
                "serialTarget": 2,
            }
        ]
        MODULE._main_action(base_obs(me, opp, evolve))
        evolved = pokemon(MODULE.KADABRA, 200, hp=80, max_hp=80)
        evolved["preEvolution"] = [{"id": MODULE.ABRA, "serial": 2}]
        me["bench"] = [evolved]
        options = [
            {"type": 10, "cardId": MODULE.KADABRA},
            {"type": 13, "attackId": 1},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])

    def test_enriching_energy_targets_dudunsparce_after_active_attack_is_ready(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        dudunsparce = pokemon(MODULE.DUDUNSPARCE, 2, hp=120, max_hp=120)
        me = player(active=active, bench=[dudunsparce], hand=[MODULE.ENRICHING_ENERGY], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 8, "cardId": MODULE.ENRICHING_ENERGY, "inPlayArea": 5, "inPlayIndex": 0},
            {"type": 13, "attackId": 1},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])

    def test_enriching_energy_models_four_draws_and_three_card_hand_gain(self):
        option = {"type": 8, "cardId": MODULE.ENRICHING_ENERGY}
        self.assertEqual(MODULE._main_draw_gain(option, MODULE.ENRICHING_ENERGY, {}), 3)
        self.assertEqual(MODULE.ENRICHING_DRAW_COUNT, 4)

    def test_xerosic_requires_six_cards_and_wounded_non_ko_active(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        me = player(active=active, hand=[MODULE.XEROSIC], deck_count=30, hand_count=1)
        wounded = pokemon(900, 3, hp=190, max_hp=200)
        opp = player(active=wounded, hand_count=6)
        options = [
            {"type": 7, "cardId": MODULE.XEROSIC, "index": 0},
            {"type": 13, "attackId": 1},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])
        opp["handCount"] = 5
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_boss_switch_prefers_highest_hp_among_visible_ko_targets(self):
        me = player(active=pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5]), hand_count=6)
        low = pokemon(900, 3, hp=70, max_hp=70)
        high = pokemon(901, 4, hp=100, max_hp=100)
        opp = player(active=pokemon(902, 5, hp=200, max_hp=200), bench=[low, high])
        current = base_obs(me, opp, []).get("current")
        options = [
            {"type": 3, "playerIndex": 1, "area": 5, "index": 0},
            {"type": 3, "playerIndex": 1, "area": 5, "index": 1},
        ]
        self.assertEqual(MODULE._choose_switch_option(options, current), [1])

    def test_boss_damage_uses_post_boss_hand_count(self):
        me = player(active=pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5]), hand_count=5)
        low = pokemon(900, 3, hp=70, max_hp=70)
        high = pokemon(901, 4, hp=100, max_hp=100)
        opp = player(active=pokemon(902, 5, hp=200, max_hp=200), bench=[low, high])
        current = base_obs(me, opp, []).get("current")
        self.assertEqual(MODULE._boss_ko_targets(current, me), [low])

    def test_deck_protection_blocks_non_terminal_draw_at_ten_cards(self):
        me = player(active=pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5]), deck_count=10)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        current = base_obs(me, opp, []).get("current")
        self.assertTrue(MODULE._v6_draw_is_blocked(current, me, gain=3, deck_delta=3))

    def test_enriching_energy_respects_deck_protection(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        dudunsparce = pokemon(MODULE.DUDUNSPARCE, 2, hp=120, max_hp=120)
        me = player(
            active=active,
            bench=[dudunsparce],
            hand=[MODULE.ENRICHING_ENERGY],
            deck_count=10,
        )
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 8, "cardId": MODULE.ENRICHING_ENERGY, "inPlayArea": 5, "inPlayIndex": 0},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_low_deck_terminal_closure_requires_an_actual_attack_option(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        dudunsparce = pokemon(MODULE.DUDUNSPARCE, 2, hp=120, max_hp=120)
        me = player(
            active=active,
            bench=[dudunsparce],
            deck_count=10,
            hand_count=10,
            prize_count=2,
        )
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [{"type": 10, "cardId": MODULE.DUDUNSPARCE}, {"type": 14}]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_nonterminal_knockout_still_completes_dudunsparce_preparation(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        dudunsparce = pokemon(MODULE.DUDUNSPARCE, 2, hp=120, max_hp=120)
        me = player(
            active=active,
            bench=[dudunsparce],
            deck_count=30,
            hand_count=5,
            prize_count=2,
        )
        opp = player(active=pokemon(900, 3, hp=80, max_hp=80))
        options = [
            {"type": 10, "cardId": MODULE.DUDUNSPARCE},
            {"type": 13, "attackId": 1},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])

    def test_low_deck_ordinary_knockout_does_not_override_terminal_prize_gate(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        dudunsparce = pokemon(MODULE.DUDUNSPARCE, 2, hp=120, max_hp=120)
        me = player(
            active=active,
            bench=[dudunsparce],
            deck_count=9,
            hand_count=3,
            prize_count=2,
        )
        opp = player(active=pokemon(900, 3, hp=80, max_hp=80))
        options = [
            {"type": 10, "cardId": MODULE.DUDUNSPARCE},
            {"type": 13, "attackId": 1},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_low_deck_fezzandipiti_ordinary_knockout_does_not_bypass_gate(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        bench_abra = pokemon(MODULE.ABRA, 2, hp=50, max_hp=50)
        me = player(
            active=active,
            bench=[bench_abra],
            hand=[MODULE.FEZANDIPITI_EX],
            deck_count=9,
            hand_count=3,
            prize_count=2,
        )
        opp = player(active=pokemon(900, 3, hp=80, max_hp=80))
        logs = [{"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3}]
        options = [
            {"type": 7, "cardId": MODULE.FEZANDIPITI_EX, "index": 0},
            {"type": 13, "attackId": 1},
        ]
        obs = base_obs(me, opp, options)
        obs["logs"] = logs
        self.assertEqual(MODULE._main_action(obs), [1])

    def test_low_deck_fezzandipiti_needs_an_attack_option_for_final_prize(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        bench_abra = pokemon(MODULE.ABRA, 2, hp=50, max_hp=50)
        me = player(
            active=active,
            bench=[bench_abra],
            hand=[MODULE.FEZANDIPITI_EX],
            deck_count=9,
            hand_count=3,
            prize_count=1,
        )
        opp = player(active=pokemon(900, 3, hp=80, max_hp=80))
        logs = [{"type": 6, "playerIndex": 0, "fromArea": 4, "toArea": 3}]
        options = [
            {"type": 7, "cardId": MODULE.FEZANDIPITI_EX, "index": 0},
            {"type": 14},
        ]
        obs = base_obs(me, opp, options)
        obs["logs"] = logs
        self.assertEqual(MODULE._main_action(obs), [1])

    def test_low_deck_psychic_draw_declines_ability_without_terminal_attack(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        me = player(active=active, deck_count=9, hand_count=5, prize_count=2)
        opp = player(active=pokemon(900, 3, hp=80, max_hp=80))
        MODULE._TURN_MEMORY.last_main_options = [{"type": 10, "cardId": MODULE.ALAKAZAM}]
        obs = base_obs(me, opp, [], turn=3)
        obs["select"] = {
            "type": 9,
            "context": 43,
            "contextCard": {"id": MODULE.ALAKAZAM},
            "effect": {"id": MODULE.ALAKAZAM, "serial": 12},
            "option": [{"type": 1}, {"type": 2}],
            "minCount": 1,
            "maxCount": 1,
        }
        self.assertEqual(MODULE._select_effect(obs), [1])

    def test_used_supporter_is_not_selected_again(self):
        active = pokemon(MODULE.KADABRA, 1, energies=[5], energy_cards=[5])
        me = player(active=active, hand=[MODULE.DAWN], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        obs = base_obs(me, opp, [], turn=3)
        obs["current"]["supporterPlayed"] = True
        options = [
            {"type": 7, "cardId": MODULE.DAWN, "index": 0},
            {"type": 13, "attackId": 1},
            {"type": 14},
        ]
        self.assertIn(MODULE._main_action({**obs, "select": {**obs["select"], "option": options}}), ([1], [2]))

    def test_used_manual_energy_is_not_selected_again(self):
        active = pokemon(MODULE.ABRA, 1)
        me = player(active=active, hand=[MODULE.BASIC_PSYCHIC], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        obs = base_obs(
            me,
            opp,
            [
                {"type": 8, "cardId": MODULE.BASIC_PSYCHIC, "inPlayArea": 4, "inPlayIndex": 0},
                {"type": 14},
            ],
        )
        obs["current"]["energyAttached"] = True
        self.assertEqual(MODULE._main_action(obs), [1])

    def test_used_retreat_is_not_selected_again(self):
        active = pokemon(MODULE.DUNSPARCE, 1, energies=[5], energy_cards=[5])
        ready = pokemon(MODULE.ALAKAZAM, 2, energies=[5], energy_cards=[5])
        me = player(active=active, bench=[ready], hand=[], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        obs = base_obs(me, opp, [{"type": 12}, {"type": 14}])
        obs["current"]["retreated"] = True
        self.assertEqual(MODULE._main_action(obs), [1])

    def test_low_deck_dudunsparce_draw_yields_to_end_without_final_prize(self):
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[5], energy_cards=[5])
        dudunsparce = pokemon(MODULE.DUDUNSPARCE, 2, hp=120, max_hp=120)
        me = player(active=active, bench=[dudunsparce], hand=[], deck_count=9)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 10, "cardId": MODULE.DUDUNSPARCE},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_night_stretcher_does_not_recover_alakazam_without_basic_route(self):
        active = pokemon(MODULE.DUNSPARCE, 1, hp=70, max_hp=70)
        me = player(active=active, hand=[MODULE.NIGHT_STRETCHER], deck_count=30)
        me["discard"] = [{"id": MODULE.ALAKAZAM}]
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 7, "cardId": MODULE.NIGHT_STRETCHER, "index": 0},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_night_stretcher_can_recover_basic_energy_for_field_abra(self):
        active = pokemon(MODULE.ABRA, 1, hp=50, max_hp=50)
        me = player(active=active, hand=[MODULE.NIGHT_STRETCHER], deck_count=30)
        me["discard"] = [{"id": MODULE.BASIC_PSYCHIC}]
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 7, "cardId": MODULE.NIGHT_STRETCHER, "index": 0},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])

    def test_night_stretcher_rejects_stage_two_without_a_basic_route(self):
        active = pokemon(MODULE.DUNSPARCE, 1, hp=70, max_hp=70)
        me = player(
            active=active,
            hand=[MODULE.NIGHT_STRETCHER, MODULE.BASIC_PSYCHIC],
            deck_count=30,
        )
        me["discard"] = [{"id": MODULE.ALAKAZAM}]
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 7, "cardId": MODULE.NIGHT_STRETCHER, "index": 0},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [1])

    def test_lanas_aid_precedes_night_stretcher_for_pokemon_and_energy_recovery(self):
        active = pokemon(MODULE.DUNSPARCE, 1, hp=70, max_hp=70)
        me = player(
            active=active,
            hand=[MODULE.LANAS_AID, MODULE.NIGHT_STRETCHER],
            deck_count=30,
        )
        me["discard"] = [{"id": MODULE.ABRA}, {"id": MODULE.BASIC_PSYCHIC}]
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 7, "cardId": MODULE.LANAS_AID, "index": 0},
            {"type": 7, "cardId": MODULE.NIGHT_STRETCHER, "index": 1},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options)), [0])

    def test_lanas_aid_selection_reserves_basic_energy_when_route_needs_both(self):
        active = pokemon(MODULE.DUNSPARCE, 1, hp=70, max_hp=70)
        me = player(active=active, deck_count=30)
        me["discard"] = [
            {"id": MODULE.ABRA},
            {"id": MODULE.KADABRA},
            {"id": MODULE.ALAKAZAM},
            {"id": MODULE.BASIC_PSYCHIC},
        ]
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 3, "area": 3, "index": 0, "playerIndex": 0},
            {"type": 3, "area": 3, "index": 1, "playerIndex": 0},
            {"type": 3, "area": 3, "index": 2, "playerIndex": 0},
            {"type": 3, "area": 3, "index": 3, "playerIndex": 0},
        ]
        obs = base_obs(me, opp, [])
        obs["select"] = {
            "type": 1,
            "context": 7,
            "effect": {"id": MODULE.LANAS_AID, "serial": 11},
            "option": options,
            "minCount": 0,
            "maxCount": 3,
        }
        chosen = MODULE._select_effect(obs)
        self.assertIn(3, chosen)

    def test_new_game_clears_v6_turn_memory(self):
        active = pokemon(MODULE.KADABRA, 1, energies=[5], energy_cards=[5])
        me = player(active=active, hand=[MODULE.ALAKAZAM], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        first = base_obs(me, opp, [{"type": 14}], turn=3)
        MODULE._main_action(first)
        MODULE._TURN_MEMORY.record_evolution(1)
        MODULE.agent({"select": None})
        options = [{"type": 9, "cardId": MODULE.ALAKAZAM, "inPlayArea": 4, "inPlayIndex": 0}, {"type": 14}]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options, turn=3)), [0])

    def test_dudunsparce_net_deck_change_accounts_for_attachments(self):
        self.assertEqual(MODULE._dudunsparce_net_deck_change(0), 2)
        self.assertEqual(MODULE._dudunsparce_net_deck_change(1), 1)
        self.assertEqual(MODULE._dudunsparce_net_deck_change(2), 0)

    def test_turn_memory_records_turn_start_and_action_budget(self):
        active = pokemon(MODULE.KADABRA, 1, energies=[5], energy_cards=[5])
        me = player(active=active, hand=[MODULE.XEROSIC], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        obs = base_obs(me, opp, [], turn=3)
        memory = MODULE.TurnMemory()
        memory.sync(obs["current"], me)
        self.assertIn(1, memory.turn_start_serials)
        memory.record_main_action({"type": 7, "cardId": MODULE.XEROSIC}, obs["current"], me)
        self.assertTrue(memory.supporter_used)
        memory.record_evolution(1)
        self.assertIn(1, memory.evolved_this_turn)
        self.assertTrue(memory.was_in_play_at_turn_start(1))

    def test_resource_ledger_tracks_unknown_counts_for_all_fixed_resources(self):
        me = player(active=pokemon(MODULE.ALAKAZAM, 1), deck_count=30)
        me["hand"] = [{"id": MODULE.ALAKAZAM, "serial": 10}]
        me["handCount"] = 1
        opp = player(active=pokemon(900, 3))
        current = base_obs(me, opp, []) ["current"]
        memory = MODULE.TurnMemory()
        memory.sync(current, me)
        self.assertEqual(memory.resource_unknown[MODULE.BASIC_PSYCHIC], 2)
        self.assertEqual(memory.resource_unknown[MODULE.ALAKAZAM], 2)

    def test_illegal_first_turn_evolution_is_not_selected(self):
        active = pokemon(MODULE.ABRA, 1, energies=[5], energy_cards=[5])
        me = player(active=active, hand=[MODULE.ALAKAZAM, MODULE.RARE_CANDY], deck_count=30)
        opp = player(active=pokemon(900, 3, hp=200, max_hp=200))
        options = [
            {"type": 9, "cardId": MODULE.ALAKAZAM, "inPlayArea": 4, "inPlayIndex": 0},
            {"type": 14},
        ]
        self.assertEqual(MODULE._main_action(base_obs(me, opp, options, turn=1)), [1])

    def test_main_supports_raw_exec_loader_without_sys_modules_registration(self):
        namespace = {"__name__": "alakazam_v6_raw_exec_test", "__file__": str(MAIN_PATH)}
        source = MAIN_PATH.read_text(encoding="utf-8")
        exec(compile(source, str(MAIN_PATH), "exec"), namespace)
        self.assertIn("agent", namespace)

    def test_main_supports_raw_exec_loader_without_file(self):
        namespace = {"__name__": "alakazam_v6_raw_exec_without_file"}
        source = MAIN_PATH.read_text(encoding="utf-8")
        previous_cwd = os.getcwd()
        try:
            os.chdir(MAIN_PATH.parent)
            exec(compile(source, "main.py", "exec"), namespace)
        finally:
            os.chdir(previous_cwd)
        self.assertIn("agent", namespace)

    def test_turn_memory_ignores_null_active_placeholder(self):
        me = player(active=None, hand=[], deck_count=30)
        me["active"] = [None]
        opp = player(active=None, hand=[], deck_count=30)
        current = base_obs(me, opp, []) ["current"]
        memory = MODULE.TurnMemory()
        memory.sync(current, me)
        self.assertEqual(memory.resource_ledger["active"], {})


if __name__ == "__main__":
    unittest.main()
