"""Deterministic dense table derived from the cache-committed structured card ontology."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# The v1 ontology currently occupies 81 positions.  Keep a small committed
# zero-padded reserve so adding an audited scalar does not silently shift every
# downstream position.
CARD_FEATURE_WIDTH = 96
KINDS = ("pokemon", "energy", "trainer")
STAGES = (
    "basic", "stage_1", "stage_2", "energy", "item", "supporter", "stadium",
    "pokemon_tool", "unknown",
)
ENERGY_TYPES = ("{G}", "{R}", "{W}", "{L}", "{P}", "{F}", "{D}", "{M}", "{C}", "●")
EFFECTS = (
    "damage", "draw", "search", "reveal", "discard", "recover", "attach",
    "evolve", "switch", "heal", "status", "prize", "shuffle", "energy_access",
    "protection", "disruption", "rule",
)
CAPABILITIES = (
    "setup", "evolution", "attacker", "prize_progress", "search", "draw",
    "recovery", "switching", "energy_access", "disruption", "survival",
)


def _observed(field: Any) -> float:
    return float(isinstance(field, dict) and field.get("state") == "observed")


def build_card_feature_table(path: Path | str, *, max_card_id: int = 2048) -> Tensor:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "0014_card_ontology_v1":
        raise ValueError("unsupported card ontology sidecar")
    table = torch.zeros(max_card_id + 1, CARD_FEATURE_WIDTH, dtype=torch.float32)
    for card in payload["cards"]:
        card_id = card["card_id"]
        if not 0 < card_id <= max_card_id:
            continue
        values: list[float] = [1.0]
        values.extend(float(card["card_kind"] == item) for item in KINDS)
        values.extend(float(card["stage"] == item) for item in STAGES)
        hp, retreat = card["hp"], card["retreat"]
        values.extend(
            (
                float(hp.get("value") or 0) / 350.0,
                _observed(hp),
                float(retreat.get("value") or 0) / 5.0,
                _observed(retreat),
                float(bool(card.get("rule"))),
            )
        )
        pokemon_type = card.get("pokemon_type", {}).get("value")
        weakness = card.get("weakness", {}).get("value")
        resistance = card.get("resistance", {}).get("value")
        for observed in (pokemon_type, weakness, resistance):
            values.extend(float(observed == item) for item in ENERGY_TYPES)
        moves = card.get("moves", [])
        damages = [move.get("damage", {}).get("value") or 0 for move in moves]
        costs = [energy for move in moves for energy in move.get("energy_cost", [])]
        values.extend(
            (
                min(len(moves), 4) / 4.0,
                min(len(costs), 10) / 10.0,
                min(max(damages, default=0), 350) / 350.0,
                min(sum(damages), 700) / 700.0,
                min(sum(move.get("name", "").casefold().startswith("[ability]") for move in moves), 2) / 2.0,
            )
        )
        effect_kinds = {
            effect.get("kind")
            for move in moves
            for effect in move.get("effects", [])
        }
        values.extend(float(item in effect_kinds) for item in EFFECTS)
        capabilities = set(card.get("capabilities", []))
        values.extend(float(item in capabilities) for item in CAPABILITIES)
        if len(values) > CARD_FEATURE_WIDTH:
            raise ValueError("card feature contract exceeds fixed width")
        table[card_id, : len(values)] = torch.tensor(values)
    return table


__all__ = ["CARD_FEATURE_WIDTH", "build_card_feature_table"]
