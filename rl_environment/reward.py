from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RewardProfile:
    """Versionable reward weights; terminal outcome remains the primary signal."""

    name: str = "terminal_v1"
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    prize_potential: float = 0.0
    attack_potential: float = 0.0
    relay_potential: float = 0.0
    library_risk_potential: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class RewardBreakdown:
    """A transparent per-transition reward payload for training dashboards."""

    terminal: float = 0.0
    prize: float = 0.0
    attack: float = 0.0
    relay: float = 0.0
    library_risk: float = 0.0

    @property
    def total(self) -> float:
        return self.terminal + self.prize + self.attack + self.relay + self.library_risk

    def to_dict(self) -> dict[str, float]:
        return {
            "reward/terminal": self.terminal,
            "reward/prize": self.prize,
            "reward/attack": self.attack,
            "reward/relay": self.relay,
            "reward/library_risk": self.library_risk,
            "reward/total": self.total,
        }


def terminal_reward(
    winner: int | None,
    player_index: int,
    profile: RewardProfile = RewardProfile(),
) -> float:
    """Return reward from one player's perspective."""
    if winner is None or winner == 2:
        return profile.draw
    if winner == player_index:
        return profile.win
    return profile.loss


def potential_difference(
    current_potential: float,
    next_potential: float,
    gamma: float,
) -> float:
    """Potential-based shaping term ``gamma * Phi(next) - Phi(current)``."""
    return gamma * next_potential - current_potential
