from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


class JsonRecord:
    def to_json(self) -> dict[str, object]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class SourceRecord(JsonRecord):
    source_id: str
    competition: str
    ref: str
    version: int | None
    title: str
    author: str
    url: str
    metadata: dict[str, object]
    source_hash: str | None
    status: str
    error: str | None
    collected_at: str


@dataclass(frozen=True)
class DeckRecord(JsonRecord):
    deck_id: str
    source_id: str
    display_name: str
    archetype: str
    primary_pokemon_ids: tuple[int, ...]
    package_path: str
    package_hash: str | None
    role: str
    status: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class RunRecord(JsonRecord):
    run_id: str
    phase: str
    config: dict[str, object]
    started_at: str
    finished_at: str | None
    status: str


@dataclass(frozen=True)
class GameRecord(JsonRecord):
    game_id: str
    run_id: str
    player_a: str
    player_b: str
    player_a_first: bool
    winner: str | None
    result: str
    status: str
    error_kind: str | None
    error: str | None
    steps: int
    payload: dict[str, object]


@dataclass(frozen=True)
class RatingEvent(JsonRecord):
    event_id: str
    run_id: str
    game_id: str
    engine: str
    revision: int
    player_a: str
    player_b: str
    before: dict[str, object]
    outcome: str
    expected_a: float | None
    delta_a: float
    delta_b: float
    after: dict[str, object]
    created_at: str
