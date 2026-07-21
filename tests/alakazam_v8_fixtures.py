from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "work" / "alakazam_v8_current" / "main.py"


def load_current_module(name: str = "alakazam_v8_current_main") -> ModuleType:
    sys.path.insert(0, str(MAIN_PATH.parent))
    spec = importlib.util.spec_from_file_location(name, MAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pokemon(
    card_id: int,
    serial: int,
    *,
    hp: int = 100,
    max_hp: int | None = None,
    energies: list[int] | None = None,
    energy_cards: list[int] | None = None,
    appear_this_turn: bool = False,
    pre_evolution: list[dict] | None = None,
) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "hp": hp,
        "maxHp": hp if max_hp is None else max_hp,
        "energies": list(energies or []),
        "energyCards": [
            {"id": energy_id, "serial": serial * 100 + index}
            for index, energy_id in enumerate(energy_cards or [])
        ],
        "appearThisTurn": appear_this_turn,
        "preEvolution": list(pre_evolution or []),
        "tools": [],
    }


def player(
    *,
    active: dict | None = None,
    bench: list[dict] | None = None,
    hand: list[int] | None = None,
    deck: list[int] | None = None,
    deck_count: int = 30,
    discard: list[int] | None = None,
    prize_count: int = 6,
    prize: list[dict] | None = None,
) -> dict:
    hand_ids = list(hand or [])
    return {
        "active": [active] if active else [],
        "bench": list(bench or []),
        "benchMax": 5,
        "hand": [
            {"id": card_id, "serial": 9000 + index}
            for index, card_id in enumerate(hand_ids)
        ],
        "handCount": len(hand_ids),
        "deck": [
            {"id": card_id, "serial": 8000 + index}
            for index, card_id in enumerate(deck or [])
        ],
        "deckCount": deck_count,
        "discard": [
            {"id": card_id, "serial": 7000 + index}
            for index, card_id in enumerate(discard or [])
        ],
        "prize": list(prize) if prize is not None else [{} for _ in range(prize_count)],
    }


def main_obs(
    me: dict,
    opponent: dict,
    options: list[dict],
    *,
    turn: int = 5,
    logs: list[dict] | None = None,
    supporter_played: bool = False,
    stadium_played: bool = False,
    energy_attached: bool = False,
    retreated: bool = False,
) -> dict:
    return {
        "current": {
            "turn": turn,
            "turnActionCount": 0,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [me, opponent],
            "supporterPlayed": supporter_played,
            "stadiumPlayed": stadium_played,
            "energyAttached": energy_attached,
            "retreated": retreated,
            "stadium": [],
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


def effect_obs(
    me: dict,
    opponent: dict,
    options: list[dict],
    *,
    effect_id: int,
    context: int,
    effect_serial: int = 5000,
    effect_step: int | None = None,
    min_count: int = 1,
    max_count: int = 1,
    turn: int = 5,
) -> dict:
    obs = main_obs(me, opponent, options, turn=turn)
    obs["select"] = {
        "type": 1,
        "context": context,
        "effect": {"id": effect_id, "serial": effect_serial},
        "minCount": min_count,
        "maxCount": max_count,
        "option": options,
    }
    if effect_step is not None:
        obs["select"]["effectStep"] = effect_step
    return obs
