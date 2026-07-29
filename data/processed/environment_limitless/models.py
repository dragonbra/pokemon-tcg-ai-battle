from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterAudit:
    active_filters: dict[str, str]


@dataclass(frozen=True)
class Tournament:
    tournament_id: int
    date: str
    country: str
    name: str
    format: str
    players: int


@dataclass(frozen=True)
class DeckShare:
    rank: int
    deck_id: int
    variant_id: int | None
    name: str
    points: int
    share: float
    url: str


@dataclass(frozen=True)
class Match:
    tournament_id: str
    round_number: int
    table: int | None
    player1: int
    player2: int
    deck1: str
    deck1_name: str
    deck2: str
    deck2_name: str
    winner: int


@dataclass(frozen=True)
class PairingAudit:
    rows: int
    accepted: int
    byes: int
    unresolved: int
    incomplete: int
    no_result: int


@dataclass
class MatchupCell:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    decisive: int = 0

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.draws + self.decisive

    @property
    def effective_rate(self) -> float:
        if self.decisive:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.n if self.n else 0.0
