from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

from tests.alakazam_v8_fixtures import load_current_module, main_obs, player, pokemon


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MAIN = ROOT / "submission" / "alakazam_v8" / "main.py"


def load_legacy_module() -> ModuleType:
    name = "alakazam_v8_legacy_parity_main"
    spec = importlib.util.spec_from_file_location(name, LEGACY_MAIN)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {LEGACY_MAIN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy_module()
CURRENT = load_current_module("alakazam_v8_current_parity_main")


class SemanticParityTests(unittest.TestCase):
    def setUp(self) -> None:
        LEGACY.agent({"select": None})
        CURRENT.agent({"select": None})

    def test_item_lock_still_allows_basic_abra_to_be_played(self) -> None:
        me = player(
            active=pokemon(CURRENT.DUNSPARCE, 1),
            hand=[CURRENT.ABRA],
        )
        opponent = player(active=pokemon(CURRENT.BUDEW, 2))
        logs = [
            {"type": 2, "playerIndex": 1},
            {
                "type": 15,
                "playerIndex": 1,
                "cardId": CURRENT.BUDEW,
                "attackId": CURRENT.ITCHY_POLLEN_ATTACK,
            },
            {"type": 2, "playerIndex": 0},
        ]
        observation = main_obs(
            me,
            opponent,
            [
                {"type": 7, "index": 0},
                {"type": 14},
            ],
            turn=3,
            logs=logs,
        )

        self.assertEqual(LEGACY.agent(observation), [0])
        self.assertEqual(CURRENT.agent(observation), [0])

    def test_bench_abra_evolves_before_preserving_active_rare_candy_route(self) -> None:
        me = player(
            active=pokemon(
                CURRENT.ABRA,
                1,
                energies=[CURRENT.PSYCHIC_ENERGY_TYPE],
            ),
            bench=[pokemon(CURRENT.ABRA, 2)],
            hand=[CURRENT.RARE_CANDY, CURRENT.KADABRA, CURRENT.ALAKAZAM],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        observation = main_obs(
            me,
            opponent,
            [
                {
                    "type": 9,
                    "area": 2,
                    "index": 1,
                    "inPlayArea": 4,
                    "inPlayIndex": 0,
                },
                {
                    "type": 9,
                    "area": 2,
                    "index": 1,
                    "inPlayArea": 5,
                    "inPlayIndex": 0,
                },
                {"type": 7, "index": 0},
                {"type": 14},
            ],
            turn=3,
        )

        self.assertEqual(LEGACY.agent(observation), [1])
        self.assertEqual(CURRENT.agent(observation), [1])

    def test_optional_setup_bench_selection_uses_available_dunsparce(self) -> None:
        me = player(hand=[CURRENT.DUDUNSPARCE, CURRENT.DUNSPARCE])
        opponent = player()
        observation = main_obs(me, opponent, [], turn=0)
        observation["select"] = {
            "type": 1,
            "context": 2,
            "minCount": 0,
            "maxCount": 1,
            "option": [
                {
                    "type": 3,
                    "area": 2,
                    "index": 1,
                    "playerIndex": 0,
                }
            ],
        }

        self.assertEqual(LEGACY.agent(observation), [0])
        self.assertEqual(CURRENT.agent(observation), [0])

    def test_temporary_active_attaches_telepath_before_optional_stadium(self) -> None:
        me = player(
            active=pokemon(CURRENT.FEZANDIPITI_EX, 1),
            hand=[CURRENT.TELEPATH_ENERGY, CURRENT.NIGHTTIME_MINE],
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        observation = main_obs(
            me,
            opponent,
            [
                {
                    "type": 8,
                    "area": 2,
                    "index": 0,
                    "inPlayArea": 4,
                    "inPlayIndex": 0,
                },
                {"type": 7, "index": 1},
                {"type": 14},
            ],
            turn=1,
        )

        self.assertEqual(LEGACY.agent(observation), [0])
        self.assertEqual(CURRENT.agent(observation), [0])

    def test_temporary_active_uses_poffin_before_retreat_energy(self) -> None:
        me = player(
            active=pokemon(CURRENT.SHAYMIN, 1),
            hand=[CURRENT.TELEPATH_ENERGY, CURRENT.POFFIN],
        )
        opponent = player(active=pokemon(900, 2, hp=200))
        observation = main_obs(
            me,
            opponent,
            [
                {
                    "type": 8,
                    "area": 2,
                    "index": 0,
                    "inPlayArea": 4,
                    "inPlayIndex": 0,
                },
                {"type": 7, "index": 1},
                {"type": 14},
            ],
            turn=1,
        )

        self.assertEqual(LEGACY.agent(observation), [1])
        self.assertEqual(CURRENT.agent(observation), [1])

    def test_first_turn_poke_pad_searches_dunsparce_before_kadabra(self) -> None:
        me = player(
            active=pokemon(CURRENT.ABRA, 1),
            bench=[pokemon(CURRENT.ABRA, 2)],
        )
        opponent = player(active=pokemon(900, 3, hp=200))
        observation = main_obs(me, opponent, [], turn=1)
        observation["select"] = {
            "type": 1,
            "context": 7,
            "minCount": 0,
            "maxCount": 1,
            "effect": {"id": CURRENT.POKE_PAD, "serial": 5000},
            "deck": [
                {"id": CURRENT.KADABRA, "serial": 8000},
                {"id": CURRENT.DUNSPARCE, "serial": 8001},
            ],
            "option": [
                {"type": 3, "area": 1, "index": 0, "playerIndex": 0},
                {"type": 3, "area": 1, "index": 1, "playerIndex": 0},
            ],
        }

        self.assertEqual(LEGACY.agent(observation), [1])
        self.assertEqual(CURRENT.agent(observation), [1])


if __name__ == "__main__":
    unittest.main()
