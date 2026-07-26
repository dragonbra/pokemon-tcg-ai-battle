"""Stable identities shared by source, split, and dataset records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Identity of one canonical episode-player trajectory."""

    date: str
    episode_id: int
    player_index: int
    submission_id_unavailable: bool = True

    def __post_init__(self) -> None:
        if not self.submission_id_unavailable:
            raise ValueError("submission_id_unavailable must be true for this source contract")
        if self.episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        if self.player_index < 0:
            raise ValueError("player_index must be non-negative")

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.date, self.episode_id, self.player_index)


@dataclass(frozen=True, slots=True)
class DecisionIdentity:
    """Identity of one decision within an episode-player trajectory."""

    source: SourceIdentity
    episode_step: int

    def __post_init__(self) -> None:
        if self.episode_step < 0:
            raise ValueError("episode_step must be non-negative")

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (*self.source.key, self.episode_step)
