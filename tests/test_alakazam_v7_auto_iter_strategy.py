import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "submission" / "alakazam_v7_auto_iter" / "main.py"
sys.path.insert(0, str(MAIN_PATH.parent))
SPEC = importlib.util.spec_from_file_location("alakazam_v7_auto_iter_main", MAIN_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {MAIN_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MIST_ENERGY = 11
ROCK_FIGHTING_ENERGY = 20


def pokemon(card_id, serial, *, hp=100, energies=None, energy_cards=None):
    return {
        "id": card_id,
        "serial": serial,
        "hp": hp,
        "maxHp": hp,
        "energies": list(energies or []),
        "energyCards": [
            {"id": energy_id, "serial": serial * 100 + index}
            for index, energy_id in enumerate(energy_cards or [])
        ],
        "appearThisTurn": False,
        "preEvolution": [],
        "tools": [],
    }


def player(*, active=None, bench=None, hand=None, deck_count=30, discard=None):
    hand = list(hand or [])
    return {
        "active": [active] if active else [],
        "bench": list(bench or []),
        "benchMax": 5,
        "hand": [{"id": card_id, "serial": 9000 + index} for index, card_id in enumerate(hand)],
        "handCount": len(hand),
        "deckCount": deck_count,
        "discard": [{"id": card_id, "serial": 7000 + index} for index, card_id in enumerate(discard or [])],
        "prize": [{} for _ in range(6)],
    }


def base_obs(me, opponent, options):
    return {
        "current": {
            "turn": 5,
            "yourIndex": 0,
            "players": [me, opponent],
            "supporterPlayed": False,
            "energyAttached": False,
            "retreated": False,
        },
        "logs": [],
        "select": {
            "type": 0,
            "context": 0,
            "option": options,
            "minCount": 1,
            "maxCount": 1,
        },
    }


class V7AutoIterStrategyTests(unittest.TestCase):
    def setUp(self):
        MODULE._EFFECT_PROGRESS.clear()
        MODULE._TURN_MEMORY.reset()

    def test_empty_bench_dudunsparce_does_not_use_run_away_draw(self):
        me = player(active=pokemon(MODULE.DUDUNSPARCE, 1), deck_count=30)
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {
                "index": 0,
                "type": 10,
                "cardId": None,
                "area": 4,
                "indexInArea": 0,
            },
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_ability_option_resolves_field_target_when_in_play_index_is_null(self):
        """Run Away Draw options use indexInArea when inPlayIndex is null."""
        active = pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE])
        bench = pokemon(MODULE.DUDUNSPARCE, 2)
        me = player(active=active, bench=[bench], hand=[900, 901])
        opponent = player(active=pokemon(900, 3, hp=200))
        option = {
            "index": 0,
            "type": 10,
            "cardId": None,
            "area": 5,
            "indexInArea": 0,
            "inPlayArea": None,
            "inPlayIndex": None,
            "attackId": None,
        }
        current = base_obs(me, opponent, [option, {"type": 14}])["current"]

        self.assertEqual(
            MODULE._option_card_id(option, {"type": 0}, current),
            MODULE.DUDUNSPARCE,
        )
        self.assertEqual(MODULE._pokemon_from_option(option, current), bench)

    def test_second_own_turn_after_swap_allows_active_abra_to_evolve(self):
        opponent = player(active=pokemon(900, 2, hp=200))
        me = player(
            active=pokemon(MODULE.ABRA, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.KADABRA],
        )
        observation = {
            "current": {
                "turn": 3,
                "yourIndex": 1,
                "firstPlayer": 1,
                "players": [opponent, me],
                "supporterPlayed": False,
                "energyAttached": False,
                "retreated": False,
            },
            "logs": [],
            "select": {
                "type": 0,
                "context": 0,
                "option": [
                    {
                        "index": 0,
                        "type": 9,
                        "area": 2,
                        "indexInArea": 0,
                        "inPlayArea": 4,
                        "inPlayIndex": 0,
                    },
                    {"index": 1, "type": 13, "attackId": 1070},
                    {"index": 2, "type": 14},
                ],
                "minCount": 1,
                "maxCount": 1,
            },
        }

        self.assertEqual(MODULE._main_action(observation), [0])

    def test_item_lock_active_abra_evolves_before_bench_abra_setup_for_ko(self):
        opponent = player(active=pokemon(235, 2, hp=30))
        me = player(
            active=pokemon(MODULE.ABRA, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ABRA, 3), pokemon(MODULE.DUNSPARCE, 4)],
            hand=[MODULE.ALAKAZAM, MODULE.RARE_CANDY, MODULE.KADABRA],
        )
        observation = {
            "current": {
                "turn": 3,
                "yourIndex": 1,
                "firstPlayer": 1,
                "players": [opponent, me],
                "supporterPlayed": True,
                "energyAttached": False,
                "retreated": False,
            },
            "logs": [
                {
                    "type": 15,
                    "playerIndex": 0,
                    "cardId": MODULE.BUDEW,
                    "attackId": MODULE.ITCHY_POLLEN_ATTACK,
                    "serial": 20,
                }
            ],
            "select": {
                "type": 0,
                "context": 0,
                "option": [
                    {
                        "index": 2,
                        "type": 9,
                        "area": 2,
                        "indexInArea": 2,
                        "inPlayArea": 4,
                        "inPlayIndex": 0,
                    },
                    {
                        "index": 2,
                        "type": 9,
                        "area": 2,
                        "indexInArea": 2,
                        "inPlayArea": 5,
                        "inPlayIndex": 0,
                    },
                    {"index": 2, "type": 13, "attackId": 1070},
                    {"index": 3, "type": 14},
                ],
                "minCount": 1,
                "maxCount": 1,
            },
        }

        self.assertEqual(MODULE._main_action(observation), [0])

    def test_nonterminal_ko_prepares_energized_bench_kadabra_before_attack(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2, energies=[MODULE.PSYCHIC_ENERGY_TYPE])],
            hand=[MODULE.ALAKAZAM],
            deck_count=24,
        )
        me["handCount"] = 13
        opponent = player(
            active=pokemon(900, 3, hp=110),
            bench=[pokemon(901, 4, hp=200)],
        )
        options = [
            {
                "index": 0,
                "type": 9,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_final_prize_attack_precedes_direct_bench_energy_handoff(self):
        """A direct Energy handoff must not delay an attack that wins immediately."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.BASIC_PSYCHIC, 900, 901],
            deck_count=24,
        )
        me["prize"] = [{}]
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.BASIC_PSYCHIC,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_pokepad_route_makes_telepath_bench_abra_handoff_concrete(self):
        """Poké Pad can find Kadabra, so Telepath belongs on the Bench Abra."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[
                pokemon(MODULE.ABRA, 2),
                pokemon(MODULE.ABRA, 3),
                pokemon(MODULE.DUNSPARCE, 4),
            ],
            hand=[MODULE.POKE_PAD, MODULE.RARE_CANDY, MODULE.BASIC_PSYCHIC, MODULE.TELEPATH_ENERGY],
            deck_count=30,
        )
        me["prize"] = [{} for _ in range(4)]
        opponent = player(active=pokemon(900, 5, hp=320))
        options = [
            {"type": 7, "indexInArea": 0},
            {
                "type": 8,
                "index": 3,
                "area": 2,
                "indexInArea": 3,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "type": 8,
                "index": 3,
                "area": 2,
                "indexInArea": 3,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [2])

    def test_empty_bench_alakazam_uses_poffin_before_non_final_knockout(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.POFFIN, MODULE.ABRA, MODULE.DUNSPARCE],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 2, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_existing_bench_kadabra_allows_active_powerful_hand_without_attach_option(self):
        """A present Bench attack line need not block a legal Active attack."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[
                pokemon(MODULE.KADABRA, 2, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
                pokemon(MODULE.DUNSPARCE, 3),
            ],
            hand=[MODULE.POFFIN, 900, 901],
            deck_count=30,
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]
        observation = base_obs(me, opponent, options)
        observation["current"]["turn"] = 3
        self.assertEqual(MODULE._main_action(observation), [1])

    def test_second_turn_boss_precedes_powerful_hand_for_higher_prize_ko(self):
        """A confirmed higher-Prize Boss KO outranks the speed metric."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.BOSS_ORDERS, 900, 901, 902, 903, 904],
            deck_count=24,
        )
        opponent = player(
            active=pokemon(900, 3, hp=200),
            bench=[pokemon(678, 4, hp=100)],
        )
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.BOSS_ORDERS},
            {"index": 1, "type": 13, "attackId": 1072},
        ]
        observation = base_obs(me, opponent, options)
        observation["current"]["turn"] = 3

        self.assertEqual(MODULE._main_action(observation), [0])

    def test_empty_bench_alakazam_keeps_final_knockout_attack_priority(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.POFFIN, MODULE.ABRA, MODULE.DUNSPARCE],
            deck_count=24,
        )
        me["prize"] = [{}]
        opponent = player(active=pokemon(900, 2, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_terminal_powerful_hand_precedes_fezandipiti_on_empty_bench(self):
        """Flip the Script must not outrank an attack that wins the last Prize."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.FEZANDIPITI_EX, 900, 901],
            deck_count=24,
        )
        me["prize"] = [{}]
        me["handCount"] = 27
        opponent = player(active=pokemon(678, 2, hp=340))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.FEZANDIPITI_EX},
            {"index": 1, "type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
        ]

        self.assertTrue(
            MODULE._v7_terminal_prize_closure(
                base_obs(me, opponent, options)["current"],
                me,
                options=options,
            )
        )
        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_empty_bench_alakazam_does_not_skip_poffin_for_potential_boss_route(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.POFFIN, MODULE.ABRA, MODULE.DUNSPARCE, MODULE.BOSS_ORDERS],
            deck_count=24,
        )
        me["prize"] = [{}, {}]
        me["handCount"] = 20
        opponent = player(
            active=pokemon(678, 2, hp=440),
            bench=[pokemon(678, 3, hp=340)],
        )
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 3, "type": 7, "cardId": MODULE.BOSS_ORDERS},
            {"index": 2, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_empty_bench_poffin_does_not_require_basic_in_hand(self):
        """Poffin searches the deck, so an empty hand need not contain Abra."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.POFFIN, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 2, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_dunsparce_only_bench_builds_abra_before_nonterminal_knockout(self):
        """A Dunsparce buffer is not an Abra-line handoff."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.POFFIN, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_unenergized_bench_kadabra_uses_poffin_before_nonterminal_knockout(self):
        """An unenergized Stage 1 is present, but it is not ready continuity."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.POFFIN, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_unenergized_bench_kadabra_without_anchor_keeps_attack(self):
        """Do not delay an attack when no legal continuity action is visible."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [{"index": 0, "type": 13, "attackId": 1072}]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_second_turn_existing_attack_line_preserves_powerful_hand(self):
        """Second-turn speed wins when the Bench already has a ready attacker."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2, energies=[MODULE.PSYCHIC_ENERGY_TYPE])],
            hand=[MODULE.POFFIN, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 1, "type": 13, "attackId": 1072},
        ]
        observation = base_obs(me, opponent, options)
        observation["current"]["turn"] = 3

        self.assertEqual(MODULE._main_action(observation), [1])

    def test_second_turn_unenergized_bench_line_builds_poffin_before_attack(self):
        """A bare Bench Stage 1 is not enough; establish another Bench Basic first."""
        fresh_kadabra = pokemon(MODULE.KADABRA, 2)
        fresh_kadabra["appearThisTurn"] = True
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[fresh_kadabra],
            hand=[MODULE.POFFIN, MODULE.ABRA, MODULE.DUNSPARCE],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.POFFIN},
            {"index": 1, "type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
        ]
        observation = base_obs(me, opponent, options)
        observation["current"].update({"turn": 3, "firstPlayer": 0})

        self.assertEqual(MODULE._main_action(observation), [0])

    def test_bench_abra_receives_psychic_before_enriching_dudunsparce(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ABRA, 2), pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.KADABRA, MODULE.BASIC_PSYCHIC, MODULE.ENRICHING_ENERGY],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.ENRICHING_ENERGY,
                "area": 2,
                "indexInArea": 2,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.BASIC_PSYCHIC,
                "area": 2,
                "indexInArea": 1,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_empty_bench_with_isolated_resources_keeps_powerful_hand(self):
        """Hand resources alone do not create a Bench target for handoff."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[
                MODULE.WONDROUS_PATCH,
                MODULE.BASIC_PSYCHIC,
                MODULE.LANAS_AID,
                MODULE.KADABRA,
                MODULE.DUDUNSPARCE,
                MODULE.ALAKAZAM,
            ],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {"index": 0, "type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
            {"index": 1, "type": 14},
        ]

        observation = base_obs(me, opponent, options)
        current, current_player = MODULE._your_state(observation)
        self.assertFalse(
            MODULE._bench_handoff_preparation_due(
                current, current_player, options, observation["select"]
            )
        )
        self.assertFalse(
            MODULE._bench_insurance_due(
                current, current_player, options, observation["select"]
            )
        )
        self.assertEqual(MODULE._main_action(observation), [0])

    def test_unrouted_bench_abra_does_not_consume_psychic_before_attack(self):
        """Do not treat a Bench Abra as a concrete handoff without evolution evidence."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ABRA, 2), pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.BASIC_PSYCHIC, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.BASIC_PSYCHIC,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_telepath_does_not_treat_unrouted_bench_abra_as_handoff(self):
        """Telepath cannot make an isolated Bench Abra ready by itself."""
        me = player(
            active=pokemon(MODULE.KADABRA, 1),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.TELEPATH_ENERGY, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_telepath_routes_to_bench_abra_when_kadabra_is_visible(self):
        """A visible Kadabra keeps Telepath→Bench Abra as a concrete route."""
        me = player(
            active=pokemon(MODULE.KADABRA, 1),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.TELEPATH_ENERGY, MODULE.KADABRA, 900],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_hand_abra_is_benched_before_nonterminal_knockout(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.ABRA, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 2, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.ABRA},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_telepath_energy_builds_bench_before_nonterminal_knockout(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.TELEPATH_ENERGY, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 2, hp=40))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_active_telepath_prepares_bench_anchor_when_basic_is_also_available(self):
        """Use Telepath's Bench search when Active Kadabra also needs Energy."""
        me = player(
            active=pokemon(MODULE.KADABRA, 1),
            bench=[pokemon(MODULE.ALAKAZAM, 2)],
            hand=[MODULE.BASIC_PSYCHIC, MODULE.TELEPATH_ENERGY],
            deck_count=18,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.BASIC_PSYCHIC,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 1,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_unenergized_active_kadabra_prefers_bench_kadabra_psychic_handoff(self):
        """A Bench Kadabra that can attack next turn beats charging a disposable Active."""
        me = player(
            active=pokemon(MODULE.KADABRA, 1),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.BASIC_PSYCHIC, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.BASIC_PSYCHIC,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.BASIC_PSYCHIC,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_second_turn_alakazam_keeps_psychic_for_current_attack(self):
        """A just-evolved Active Alakazam must attack before Bench handoff."""
        active = pokemon(MODULE.ALAKAZAM, 1)
        active["appearThisTurn"] = True
        me = player(
            active=active,
            bench=[
                pokemon(MODULE.KADABRA, 2),
                pokemon(MODULE.KADABRA, 3),
                pokemon(MODULE.KADABRA, 4),
            ],
            hand=[MODULE.BASIC_PSYCHIC, 900, 901],
            deck_count=26,
        )
        opponent = player(active=pokemon(900, 5, hp=200))
        observation = base_obs(
            me,
            opponent,
            [
                {
                    "index": 0,
                    "type": 8,
                    "area": 2,
                    "indexInArea": 0,
                    "inPlayArea": 4,
                    "inPlayIndex": 0,
                },
                {
                    "index": 0,
                    "type": 8,
                    "area": 2,
                    "indexInArea": 0,
                    "inPlayArea": 5,
                    "inPlayIndex": 0,
                },
                {"type": 14},
            ],
        )
        observation["current"]["turn"] = 3

        self.assertEqual(MODULE._main_action(observation), [0])

    def test_unenergized_active_kadabra_keeps_energy_when_no_bench_attacker_exists(self):
        """Without a real Bench successor, the only Psychic stays on Active."""
        me = player(
            active=pokemon(MODULE.KADABRA, 1),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.BASIC_PSYCHIC, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.BASIC_PSYCHIC,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"index": 1, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_unenergized_active_abra_prefers_telepath_on_bench_kadabra(self):
        """Telepath should charge the existing Bench Kadabra, not the disposable Abra."""
        me = player(
            active=pokemon(MODULE.ABRA, 1),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.TELEPATH_ENERGY, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_rock_fighting_energy_blocks_powerful_hand_like_mist(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ABRA, 3, energies=[MODULE.PSYCHIC_ENERGY_TYPE])],
            hand=[MODULE.ENHANCED_HAMMER, 900, 901],
            deck_count=24,
        )
        opponent = player(
            active=pokemon(678, 2, hp=40, energy_cards=[ROCK_FIGHTING_ENERGY])
        )
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.ENHANCED_HAMMER},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_bench_anchor_does_not_attach_telepath_to_ready_active_first(self):
        """A redundant Active attachment must not outrank putting down Abra."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.TELEPATH_ENERGY, MODULE.ABRA],
            deck_count=21,
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 7, "cardId": MODULE.ABRA},
            {"index": 3, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [2])

    def test_rock_fighting_energy_does_not_protect_non_fighting_pokemon(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ABRA, 3, energies=[MODULE.PSYCHIC_ENERGY_TYPE])],
            hand=[MODULE.ENHANCED_HAMMER, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 2, hp=40, energy_cards=[ROCK_FIGHTING_ENERGY]))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.ENHANCED_HAMMER},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_mist_energy_uses_the_same_protective_damage_model(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ABRA, 3, energies=[MODULE.PSYCHIC_ENERGY_TYPE])],
            hand=[MODULE.ENHANCED_HAMMER, 900, 901],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 2, hp=40, energy_cards=[MIST_ENERGY]))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.ENHANCED_HAMMER},
            {"index": 1, "type": 13, "attackId": 1072},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_boss_higher_prize_ko_precedes_hammering_protective_active(self):
        """Do not spend Hammer when Boss still takes a strictly better Prize."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            hand=[MODULE.ENHANCED_HAMMER, MODULE.BOSS_ORDERS] + list(range(900, 910)),
            deck_count=24,
        )
        opponent = player(
            active=pokemon(900, 2, hp=120, energy_cards=[MIST_ENERGY]),
            bench=[pokemon(MODULE.FEZANDIPITI_EX, 3, hp=210)],
        )
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.ENHANCED_HAMMER},
            {"index": 1, "type": 7, "cardId": MODULE.BOSS_ORDERS},
            {"index": 2, "type": 13, "attackId": 1072},
            {"index": 3, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_active_kadabra_moves_telepath_to_bench_handoff(self):
        """Telepath must prepare the Bench instead of reattaching to ready Kadabra."""
        me = player(
            active=pokemon(MODULE.KADABRA, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2), pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.ALAKAZAM, MODULE.ENRICHING_ENERGY, MODULE.BOSS_ORDERS,
                  MODULE.ALAKAZAM, MODULE.TELEPATH_ENERGY, MODULE.KADABRA],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 4, hp=50))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {
                "index": 2,
                "type": 8,
                "cardId": MODULE.ENRICHING_ENERGY,
                "area": 2,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {
                "index": 3,
                "type": 8,
                "cardId": MODULE.BASIC_PSYCHIC,
                "area": 2,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 4, "type": 13, "attackId": 1071},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_normalized_energy_option_uses_index_in_area_for_bench_handoff(self):
        """Normalized options must resolve the hand card from indexInArea."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.TELEPATH_ENERGY, MODULE.KADABRA, MODULE.ALAKAZAM],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        # The evaluator's normalized trace uses ``index`` for the local
        # option position and ``indexInArea`` for the original hand index.
        # There is deliberately no cardId so the resolver must honor the
        # area index rather than accidentally reading hand[option_index].
        options = [
            {
                "index": 0,
                "type": 8,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 13, "attackId": 1072},
        ]

        observation = base_obs(me, opponent, options)
        current, current_player = MODULE._your_state(observation)
        self.assertEqual(
            MODULE._option_card_id(options[1], observation["select"], current),
            MODULE.TELEPATH_ENERGY,
        )
        self.assertIs(
            MODULE._pokemon_from_option(options[1], current),
            current_player["bench"][0],
        )
        self.assertEqual(MODULE._main_action(observation), [1])

    def test_normalized_discard_option_uses_index_in_area(self):
        """Discard choices must resolve the source card's actual discard index."""
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.LANAS_AID],
            discard=[MODULE.ALAKAZAM, MODULE.ABRA, MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        observation = base_obs(
            me,
            opponent,
            [
                {
                    "index": 0,
                    "type": 3,
                    "area": 3,
                    "indexInArea": 1,
                    "playerIndex": 0,
                },
                {
                    "index": 1,
                    "type": 3,
                    "area": 3,
                    "indexInArea": 2,
                    "playerIndex": 0,
                },
            ],
        )

        options = observation["select"]["option"]
        self.assertEqual(
            [MODULE._option_card_id(option, observation["select"], observation["current"])
             for option in options],
            [MODULE.ABRA, MODULE.BASIC_PSYCHIC],
        )

    def test_dudunsparce_deck_delta_counts_pre_evolution_stack(self):
        """Run Away Draw must count the Dunsparce under Dudunsparce too."""
        active = pokemon(
            MODULE.DUDUNSPARCE,
            1,
            energies=[MODULE.PSYCHIC_ENERGY_TYPE],
            energy_cards=[MODULE.ENRICHING_ENERGY],
        )
        active["preEvolution"] = [{"id": MODULE.DUNSPARCE, "serial": 2}]
        me = player(active=active, deck_count=27)
        opponent = player(active=pokemon(900, 3, hp=200))
        observation = base_obs(
            me,
            opponent,
            [{"type": 10, "area": 4, "indexInArea": 0}],
        )

        self.assertEqual(
            MODULE._dudunsparce_deck_delta(
                me,
                observation["select"]["option"][0],
                observation["current"],
            ),
            0,
        )

    def test_active_kadabra_attaches_to_bench_kadabra_played_this_turn(self):
        """A same-turn Bench Kadabra can receive Energy for the next turn."""
        bench_kadabra = pokemon(MODULE.KADABRA, 2)
        bench_kadabra["appearThisTurn"] = True
        me = player(
            active=pokemon(MODULE.KADABRA, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[bench_kadabra, pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.TELEPATH_ENERGY, MODULE.ENRICHING_ENERGY, MODULE.ALAKAZAM],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 13, "attackId": 1071},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_active_kadabra_recovers_energy_for_same_turn_bench_abra(self):
        """A same-turn Bench Abra may receive Energy even though it cannot evolve yet."""
        bench_abra = pokemon(MODULE.ABRA, 2)
        bench_abra["appearThisTurn"] = True
        me = player(
            active=pokemon(MODULE.KADABRA, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[bench_abra, pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.LANAS_AID, MODULE.ENRICHING_ENERGY, MODULE.ALAKAZAM],
            deck_count=24,
            discard=[MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.LANAS_AID},
            {"index": 1, "type": 13, "attackId": 1071},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_resolves_field_target_using_index_in_area(self):
        me = player(active=pokemon(MODULE.DUDUNSPARCE, 1), deck_count=30)
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {"index": 0, "type": 14},
            {
                "index": 1,
                "type": 10,
                "cardId": None,
                "area": 4,
                "indexInArea": 0,
            },
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_enriching_energy_precedes_bench_setup_for_active_dunsparce_route(self):
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.DUDUNSPARCE, MODULE.ENRICHING_ENERGY],
            deck_count=30,
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {
                "index": 0,
                "type": 9,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "area": 2,
                "indexInArea": 1,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 2,
                "type": 9,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 3, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_new_active_dunsparce_prepares_enriching_for_next_turn_evolution(self):
        """A just-played Dunsparce may receive Enriching before it can evolve."""
        active = pokemon(MODULE.DUNSPARCE, 1)
        active["appearThisTurn"] = True
        me = player(
            active=active,
            bench=[pokemon(MODULE.ABRA, 2), pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.DUDUNSPARCE, MODULE.ENRICHING_ENERGY],
            deck_count=30,
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.ENRICHING_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "cardId": MODULE.ENRICHING_ENERGY,
                "area": 2,
                "indexInArea": 1,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_dudunsparce_draw_gate_uses_the_selected_instance(self):
        """Run Away Draw counts the selected Dudunsparce's attachments only."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[
                pokemon(MODULE.DUDUNSPARCE, 2),
                pokemon(MODULE.DUDUNSPARCE, 3, energy_cards=[MODULE.ENRICHING_ENERGY]),
            ],
            hand=[900, 901, 902, 903, 904],
            deck_count=12,
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        options = [
            {
                "index": 0,
                "type": 10,
                "cardId": None,
                "area": 5,
                "indexInArea": 1,
            },
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_recovery_does_not_treat_lone_stage_two_in_hand_as_basic_source(self):
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            hand=[MODULE.ALAKAZAM, MODULE.NIGHT_STRETCHER],
            discard=[MODULE.BASIC_PSYCHIC],
            deck_count=20,
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        options = [
            {"index": 1, "type": 7},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_recovery_does_not_use_lanas_aid_for_unrouted_abra(self):
        """A lone Abra without an evolution route is not a real recovery target."""
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            bench=[pokemon(MODULE.ABRA, 2)],
            hand=[MODULE.LANAS_AID, MODULE.BATTLE_CAGE],
            deck_count=35,
            discard=[MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.LANAS_AID},
            {"index": 1, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_ready_active_preserves_psychic_attachment_for_bench_kadabra(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2), pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.BASIC_PSYCHIC, MODULE.ENRICHING_ENERGY, MODULE.ALAKAZAM],
            deck_count=24,
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {
                "index": 0,
                "type": 8,
                "area": 2,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {
                "index": 1,
                "type": 8,
                "area": 2,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {"index": 2, "type": 13, "attackId": 1072},
            {"index": 3, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_ready_bench_alakazam_recovers_psychic_before_attack(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ALAKAZAM, 2)],
            hand=[MODULE.LANAS_AID, MODULE.ENRICHING_ENERGY, 900, 901, 902, 903, 904, 905],
            deck_count=20,
            discard=[MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7},
            {
                "index": 1,
                "type": 8,
                "area": 2,
                "inPlayArea": 5,
                "inPlayIndex": 0,
            },
            {"index": 2, "type": 13, "attackId": 1072},
            {"index": 3, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_wondrous_patch_prepares_bench_before_nonterminal_knockout(self):
        """Patch is a free visible handoff resource and should precede attack."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.WONDROUS_PATCH, 900, 901],
            deck_count=24,
            discard=[MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.WONDROUS_PATCH},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_wondrous_patch_skips_abra_without_visible_next_evolution(self):
        """Patch must not spend Psychic on an Abra with no visible conversion route."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[
                pokemon(MODULE.ABRA, 2, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
                pokemon(MODULE.ABRA, 3),
            ],
            hand=[MODULE.WONDROUS_PATCH, MODULE.ALAKAZAM, MODULE.ALAKAZAM],
            deck_count=20,
            discard=[MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.WONDROUS_PATCH},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_wondrous_patch_can_charge_same_turn_bench_abra_for_next_turn(self):
        """Patch may charge a new Abra even though evolution must wait."""
        bench_abra = pokemon(MODULE.ABRA, 2)
        bench_abra["appearThisTurn"] = True
        second_bench_abra = pokemon(MODULE.ABRA, 3)
        second_bench_abra["appearThisTurn"] = True
        me = player(
            active=pokemon(MODULE.KADABRA, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 4), bench_abra, second_bench_abra],
            hand=[
                MODULE.SACRED_ASH,
                MODULE.WONDROUS_PATCH,
                MODULE.BOSS_ORDERS,
                MODULE.RARE_CANDY,
                MODULE.ALAKAZAM,
                MODULE.HILDA,
                MODULE.DUNSPARCE,
                MODULE.ALAKAZAM,
                MODULE.ALAKAZAM,
                MODULE.POKE_PAD,
            ],
            deck_count=24,
            discard=[MODULE.BASIC_PSYCHIC, MODULE.ABRA, MODULE.KADABRA],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7},
            {"index": 1, "type": 7},
            {"index": 9, "type": 7},
            {"type": 13, "attackId": 1071},
            {"type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_wondrous_patch_precedes_hilda_when_it_completes_handoff(self):
        """A direct handoff resource beats a general search Supporter."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.WONDROUS_PATCH, MODULE.HILDA, 900, 901],
            deck_count=24,
            discard=[MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.WONDROUS_PATCH},
            {"index": 1, "type": 7, "cardId": MODULE.HILDA},
            {"index": 2, "type": 13, "attackId": 1072},
            {"index": 3, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_wondrous_patch_does_not_delay_final_knockout(self):
        """A final Prize closure remains more valuable than future setup."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.KADABRA, 2)],
            hand=[MODULE.WONDROUS_PATCH, 900, 901],
            deck_count=24,
            discard=[MODULE.BASIC_PSYCHIC],
        )
        me["prize"] = [{}]
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.WONDROUS_PATCH},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_lanas_aid_recovers_abra_and_psychic_when_both_are_missing(self):
        """Lana's Aid should fill both visible recovery gaps in one Supporter."""
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            bench=[pokemon(MODULE.DUNSPARCE, 2), pokemon(MODULE.DUNSPARCE, 3)],
            hand=[MODULE.LANAS_AID, MODULE.ALAKAZAM, MODULE.BATTLE_CAGE],
            deck_count=35,
            discard=[MODULE.ABRA, MODULE.KADABRA, MODULE.ABRA, MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 4, hp=220))
        discard_options = [
            {"type": 3, "area": 3, "index": 0},
            {"type": 3, "area": 3, "index": 1},
            {"type": 3, "area": 3, "index": 2},
            {"type": 3, "area": 3, "index": 3},
        ]
        obs = {
            "current": {
                "turn": 10,
                "yourIndex": 0,
                "players": [me, opponent],
                "supporterPlayed": True,
                "energyAttached": False,
                "retreated": False,
            },
            "logs": [],
            "select": {
                "type": 1,
                "context": 7,
                "effect": {"id": MODULE.LANAS_AID, "serial": 95},
                "minCount": 1,
                "maxCount": 3,
                "option": discard_options,
            },
        }

        chosen = MODULE._select_effect(obs)

        self.assertEqual(len(chosen), 2)
        self.assertEqual(
            {MODULE._option_card_id(discard_options[index], obs["select"], obs["current"])
             for index in chosen},
            {MODULE.ABRA, MODULE.BASIC_PSYCHIC},
        )

    def test_lanas_aid_keeps_single_resource_recovery_minimal(self):
        """One visible gap must not make Lana's Aid take an unrelated card."""
        me = player(
            active=pokemon(MODULE.DUNSPARCE, 1),
            bench=[pokemon(MODULE.ABRA, 2, energies=[MODULE.PSYCHIC_ENERGY_TYPE])],
            hand=[MODULE.LANAS_AID, MODULE.BATTLE_CAGE],
            deck_count=35,
            discard=[MODULE.BASIC_PSYCHIC, MODULE.ALAKAZAM],
        )
        opponent = player(active=pokemon(900, 4, hp=220))
        discard_options = [
            {"type": 3, "area": 3, "index": 0},
            {"type": 3, "area": 3, "index": 1},
        ]
        obs = {
            "current": {
                "turn": 10,
                "yourIndex": 0,
                "players": [me, opponent],
                "supporterPlayed": True,
                "energyAttached": False,
                "retreated": False,
            },
            "logs": [],
            "select": {
                "type": 1,
                "context": 7,
                "effect": {"id": MODULE.LANAS_AID, "serial": 96},
                "minCount": 1,
                "maxCount": 3,
                "option": discard_options,
            },
        }

        chosen = MODULE._select_effect(obs)

        self.assertEqual(len(chosen), 1)
        self.assertEqual(
            MODULE._option_card_id(discard_options[chosen[0]], obs["select"], obs["current"]),
            MODULE.BASIC_PSYCHIC,
        )

    def test_poffin_prefers_dunsparce_only_with_explicit_enriching_route(self):
        """A complete visible attack line may reserve Poffin for Enriching draw."""
        me = player(
            active=pokemon(MODULE.ABRA, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.ABRA, 2), pokemon(MODULE.KADABRA, 3)],
            hand=[MODULE.POFFIN, MODULE.ALAKAZAM, MODULE.RARE_CANDY, MODULE.ENRICHING_ENERGY],
            deck_count=30,
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        deck_options = [
            {"index": 0, "type": 3, "area": 1, "indexInArea": 0},
            {"index": 1, "type": 3, "area": 1, "indexInArea": 1},
        ]
        obs = {
            "current": {
                "turn": 3,
                "yourIndex": 0,
                "firstPlayer": 0,
                "players": [me, opponent],
                "supporterPlayed": False,
                "energyAttached": False,
                "retreated": False,
            },
            "logs": [],
            "select": {
                "type": 1,
                "context": 7,
                "effect": {"id": MODULE.POFFIN, "serial": 97},
                "minCount": 1,
                "maxCount": 2,
                "option": deck_options,
                "deck": [{"id": MODULE.ABRA}, {"id": MODULE.DUNSPARCE}],
            },
        }

        chosen = MODULE._select_effect(obs)

        self.assertEqual(chosen, [1])

    def test_poffin_keeps_abra_when_energized_bench_abra_has_no_visible_evolution(self):
        """Energy alone does not make an unrouted Bench Abra a ready handoff."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[
                pokemon(MODULE.ABRA, 2, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
                pokemon(MODULE.ABRA, 3, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            ],
            hand=[MODULE.POFFIN, 900, 901],
            deck_count=30,
        )
        opponent = player(active=pokemon(900, 4, hp=200))
        deck_options = [
            {"index": 0, "type": 3, "area": 1, "indexInArea": 0},
            {"index": 1, "type": 3, "area": 1, "indexInArea": 1},
        ]
        obs = {
            "current": {
                "turn": 7,
                "yourIndex": 0,
                "firstPlayer": 0,
                "players": [me, opponent],
                "supporterPlayed": False,
                "energyAttached": False,
                "retreated": False,
            },
            "logs": [],
            "select": {
                "type": 1,
                "context": 7,
                "effect": {"id": MODULE.POFFIN, "serial": 98},
                "minCount": 1,
                "maxCount": 1,
                "option": deck_options,
                "deck": [{"id": MODULE.ABRA}, {"id": MODULE.DUNSPARCE}],
            },
        }

        chosen = MODULE._select_effect(obs)

        self.assertEqual(chosen, [0])

    def test_lanas_aid_builds_discarded_abra_successor_when_bench_has_only_dunsparce(self):
        """Lana's Aid should start a concrete Abra+Psychic Bench handoff before attack."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.LANAS_AID, 900, 901],
            deck_count=24,
            discard=[MODULE.ABRA, MODULE.BASIC_PSYCHIC],
        )
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.LANAS_AID},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])
        discard_options = [
            {"index": 0, "type": 3, "area": 3},
            {"index": 1, "type": 3, "area": 3},
        ]
        recovery_obs = {
            "current": {
                "turn": 5,
                "yourIndex": 0,
                "players": [me, opponent],
                "supporterPlayed": True,
                "energyAttached": False,
                "retreated": False,
            },
            "logs": [],
            "select": {
                "type": 1,
                "context": 7,
                "effect": {"id": MODULE.LANAS_AID, "serial": 98},
                "minCount": 1,
                "maxCount": 3,
                "option": discard_options,
            },
        }

        chosen = MODULE._select_effect(recovery_obs)

        self.assertEqual(
            {
                MODULE._option_card_id(discard_options[index], recovery_obs["select"], recovery_obs["current"])
                for index in chosen
            },
            {MODULE.ABRA, MODULE.BASIC_PSYCHIC},
        )

    def test_lanas_aid_successor_route_does_not_delay_final_knockout(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.LANAS_AID, 900, 901],
            deck_count=24,
            discard=[MODULE.ABRA, MODULE.BASIC_PSYCHIC],
        )
        me["prize"] = [{}]
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.LANAS_AID},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_night_stretcher_builds_discarded_abra_successor_when_no_better_anchor(self):
        """Night Stretcher may retrieve Abra when hand Psychic completes the next turn route."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.NIGHT_STRETCHER, MODULE.BASIC_PSYCHIC, 900],
            deck_count=24,
            discard=[MODULE.ABRA],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.NIGHT_STRETCHER},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

        discard_options = [
            {"index": 0, "type": 3, "area": 3},
        ]
        recovery_obs = {
            "current": {
                "turn": 5,
                "yourIndex": 0,
                "players": [me, opponent],
                "supporterPlayed": False,
                "energyAttached": False,
                "retreated": False,
            },
            "logs": [],
            "select": {
                "type": 1,
                "context": 7,
                "effect": {"id": MODULE.NIGHT_STRETCHER, "serial": 99},
                "minCount": 1,
                "maxCount": 1,
                "option": discard_options,
            },
        }

        chosen = MODULE._select_effect(recovery_obs)

        self.assertEqual(chosen, [0])

    def test_night_stretcher_beats_unusable_lanas_aid_for_abra_handoff(self):
        """Lana's Aid should not suppress Night Stretcher without discard Energy."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[
                MODULE.NIGHT_STRETCHER,
                MODULE.BASIC_PSYCHIC,
                MODULE.LANAS_AID,
                900,
            ],
            deck_count=24,
            discard=[MODULE.ABRA],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.NIGHT_STRETCHER},
            {"index": 2, "type": 7, "cardId": MODULE.LANAS_AID},
            {"index": 3, "type": 13, "attackId": 1072},
            {"index": 4, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [0])

    def test_night_stretcher_successor_route_does_not_delay_final_knockout(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.NIGHT_STRETCHER, MODULE.BASIC_PSYCHIC, 900],
            deck_count=24,
            discard=[MODULE.ABRA],
        )
        me["prize"] = [{}]
        opponent = player(active=pokemon(900, 3, hp=40))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.NIGHT_STRETCHER},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]

        self.assertEqual(MODULE._main_action(base_obs(me, opponent, options)), [1])

    def test_item_lock_rejects_night_stretcher_successor_route(self):
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[pokemon(MODULE.DUNSPARCE, 2)],
            hand=[MODULE.NIGHT_STRETCHER, MODULE.BASIC_PSYCHIC],
            deck_count=24,
            discard=[MODULE.ABRA],
        )
        opponent = player(active=pokemon(235, 3, hp=200))
        options = [
            {"index": 0, "type": 7, "cardId": MODULE.NIGHT_STRETCHER},
            {"index": 1, "type": 13, "attackId": 1072},
            {"index": 2, "type": 14},
        ]
        obs = base_obs(me, opponent, options)
        obs["logs"] = [{"type": 15, "playerIndex": 1, "cardId": 235, "attackId": 323}]

        self.assertEqual(MODULE._main_action(obs), [1])

    def test_dudunsparce_draw_precedes_handoff_when_it_creates_terminal_ko(self):
        """A draw that creates the final KO must still precede a handoff."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[
                pokemon(343, 2),
                pokemon(MODULE.ALAKAZAM, 3),
                pokemon(MODULE.DUDUNSPARCE, 4),
                pokemon(MODULE.DUDUNSPARCE, 5),
            ],
            hand=[MODULE.TELEPATH_ENERGY],
            deck_count=10,
        )
        me["prize"] = [{}, {}]
        me["handCount"] = 14
        opponent = player(active=pokemon(678, 6, hp=340))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {
                "index": 1,
                "type": 10,
                "area": 5,
                "indexInArea": 2,
            },
            {"index": 2, "type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
        ]
        observation = base_obs(me, opponent, options)
        observation["current"].update({"turn": 13, "firstPlayer": 0})

        self.assertEqual(MODULE._main_action(observation), [1])

    def test_nonterminal_bench_telepath_handoff_precedes_dudunsparce_draw(self):
        """A visible Bench Alakazam handoff beats a draw with no immediate KO."""
        me = player(
            active=pokemon(MODULE.ALAKAZAM, 1, energies=[MODULE.PSYCHIC_ENERGY_TYPE]),
            bench=[
                pokemon(343, 2),
                pokemon(MODULE.ALAKAZAM, 3),
                pokemon(MODULE.DUDUNSPARCE, 4),
                pokemon(MODULE.DUDUNSPARCE, 5),
            ],
            hand=[MODULE.TELEPATH_ENERGY],
            deck_count=10,
        )
        me["prize"] = [{}, {}]
        me["handCount"] = 14
        opponent = player(active=pokemon(678, 6, hp=360))
        options = [
            {
                "index": 0,
                "type": 8,
                "cardId": MODULE.TELEPATH_ENERGY,
                "area": 2,
                "indexInArea": 0,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            },
            {
                "index": 1,
                "type": 10,
                "area": 5,
                "indexInArea": 2,
            },
            {"index": 2, "type": 13, "attackId": MODULE.POWERFUL_HAND_ATTACK},
        ]
        observation = base_obs(me, opponent, options)
        observation["current"].update({"turn": 13, "firstPlayer": 0})

        self.assertEqual(MODULE._main_action(observation), [0])


if __name__ == "__main__":
    unittest.main()
