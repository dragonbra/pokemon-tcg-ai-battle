"""Approximate paired-panel MDE planning; final inference uses observed flips."""

from __future__ import annotations

import math

Z_95_TWO_SIDED = 1.959963984540054
Z_80_POWER = 0.8416212335729143


def paired_required_games(delta: float, discordant_rate: float) -> int:
    if not 0 < delta < 1 or not 0 < discordant_rate <= 1:
        raise ValueError("delta and discordant rate must be probabilities")
    return math.ceil((Z_95_TWO_SIDED + Z_80_POWER) ** 2 * discordant_rate / delta ** 2)


def paired_mde(games: int, discordant_rate: float) -> float:
    if games < 1 or not 0 < discordant_rate <= 1:
        raise ValueError("invalid MDE inputs")
    return (Z_95_TWO_SIDED + Z_80_POWER) * math.sqrt(discordant_rate / games)


def panel_power_report(games: int = 2048) -> dict[str, object]:
    rates = (0.10, 0.20, 0.30)
    return {
        "games": games,
        "assumed_discordant_rates": list(rates),
        "mde_80_power": {str(rate): paired_mde(games, rate) for rate in rates},
        "required_for_plus_1pp": {str(rate): paired_required_games(.01, rate) for rate in rates},
        "required_for_plus_2pp": {str(rate): paired_required_games(.02, rate) for rate in rates},
        "interpretation": (
            "2048 paired games can indicate roughly 2pp direction when discordance is low; "
            "small deltas are not automatically significant and observed flips control the test."
        ),
    }


__all__ = ["paired_mde", "paired_required_games", "panel_power_report"]
