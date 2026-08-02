from __future__ import annotations

import importlib


BASE = "train.0026_raging_bolt_canonical_decoder_rl"


def module(name: str):
    return importlib.import_module(f"{BASE}.{name}")


def pokemon(card_id: int, player: int) -> dict:
    return {
        "id": card_id,
        "playerIndex": player,
        "serial": card_id + player * 10000,
        "hp": 200,
        "maxHp": 200,
        "appearThisTurn": False,
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    }


def observation(actor: int = 0) -> dict:
    players = []
    for player, card_id in ((0, 96), (1, 63)):
        players.append(
            {
                "active": [pokemon(card_id, player)],
                "bench": [],
                "hand": [],
                "discard": [],
                "prize": [None] * 6,
                "deckCount": 40,
                "handCount": 0,
                "asleep": False,
                "burned": False,
                "confused": False,
                "paralyzed": False,
                "poisoned": False,
            }
        )
    return {
        "current": {
            "yourIndex": actor,
            "firstPlayer": 0,
            "turn": 3,
            "turnActionCount": 1,
            "players": players,
            "stadium": [],
            "looking": [],
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
        },
        "select": {
            "type": 0,
            "context": 0,
            "contextCard": None,
            "effect": None,
            "deck": [],
            "option": [{"type": 13, "attackId": 120}, {"type": 14}],
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
        },
        "logs": [],
    }
