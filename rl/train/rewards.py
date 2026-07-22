from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALAKAZAM = 743
KADABRA = 742
ABRA = 741
BASIC_PSYCHIC = 5


@dataclass(frozen=True)
class PotentialComponents:
    """Visible-state potentials used for auditable reward shaping."""

    prize_race: float = 0.0
    attack_readiness: float = 0.0
    library_safety: float = 0.0

    @property
    def total(self) -> float:
        return self.prize_race + self.attack_readiness + self.library_safety

    def to_dict(self) -> dict[str, float]:
        return {
            "prize_race": self.prize_race,
            "attack_readiness": self.attack_readiness,
            "library_safety": self.library_safety,
            "total": self.total,
        }


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [card for card in player.get(area) or [] if isinstance(card, dict)]


def _card_id(card: dict[str, Any]) -> int | None:
    try:
        value = card.get("id", card.get("cardId"))
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _energy_count(card: dict[str, Any]) -> int:
    values = card.get("energies", card.get("energyCards", [])) or []
    return len(values)


def _players(observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    current = observation.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0) or 0)
    player = players[your_index] if 0 <= your_index < len(players) else {}
    opponent_index = 1 - your_index
    opponent = players[opponent_index] if 0 <= opponent_index < len(players) else {}
    return player, opponent


def observation_potential(observation: dict[str, Any]) -> PotentialComponents:
    """Compute conservative potentials only from observation-visible fields."""
    player, opponent = _players(observation)
    own_prize = int(player.get("prizeCount", len(player.get("prize") or [])) or 0)
    opponent_prize = int(opponent.get("prizeCount", len(opponent.get("prize") or [])) or 0)
    prize_race = (opponent_prize - own_prize) / 6.0

    field = [*_cards(player, "active"), *_cards(player, "bench")]
    readiness = 0.0
    for card in field:
        card_id = _card_id(card)
        if card_id == ALAKAZAM:
            readiness += 0.45
        elif card_id == KADABRA:
            readiness += 0.25
        elif card_id == ABRA:
            readiness += 0.10
        readiness += min(_energy_count(card), 3) * 0.03
    attack_readiness = min(readiness, 1.0)

    deck_count = float(player.get("deckCount", 0) or 0)
    # Normal draw/search above 15 cards is not intrinsically bad. Once the
    # V6 mid-game protection threshold is crossed, make the potential
    # piecewise so shaping reacts to depletion rather than rewarding idling.
    library_safety = (
        0.0 if deck_count >= 15.0 else max(-1.0, (deck_count - 15.0) / 15.0)
    )
    return PotentialComponents(prize_race, attack_readiness, library_safety)


def potential_shaping(
    current: dict[str, Any],
    next_observation: dict[str, Any] | None,
    *,
    gamma: float = 0.99,
) -> dict[str, float]:
    """Return component-wise ``gamma * Phi(next) - Phi(current)`` values."""
    before = observation_potential(current)
    after = observation_potential(next_observation or current)
    values = {
        "prize_race": gamma * after.prize_race - before.prize_race,
        "attack_readiness": gamma * after.attack_readiness - before.attack_readiness,
        "library_safety": gamma * after.library_safety - before.library_safety,
    }
    values["total"] = sum(values.values())
    return values
