import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "submission" / "alakazam_v7_auto_iter" / "main.py"
SPEC = importlib.util.spec_from_file_location("alakazam_v7_auto_iter_main", MAIN_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {MAIN_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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


def player(*, active=None, bench=None, hand=None, deck_count=30):
    hand = list(hand or [])
    return {
        "active": [active] if active else [],
        "bench": list(bench or []),
        "benchMax": 5,
        "hand": [{"id": card_id, "serial": 9000 + index} for index, card_id in enumerate(hand)],
        "handCount": len(hand),
        "deckCount": deck_count,
        "discard": [],
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


if __name__ == "__main__":
    unittest.main()
