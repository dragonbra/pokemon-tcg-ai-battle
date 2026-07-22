from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import DeckRecord, GameRecord, JsonRecord, RatingEvent, RunRecord, SourceRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    competition TEXT NOT NULL,
    ref TEXT NOT NULL,
    version INTEGER,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    url TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_hash TEXT,
    status TEXT NOT NULL,
    error TEXT,
    collected_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decks (
    deck_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    archetype TEXT NOT NULL,
    primary_pokemon_json TEXT NOT NULL,
    package_path TEXT NOT NULL,
    package_hash TEXT,
    role TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES sources(source_id)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    phase TEXT NOT NULL,
    config_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    player_a TEXT NOT NULL,
    player_b TEXT NOT NULL,
    player_a_first INTEGER NOT NULL,
    winner TEXT,
    result TEXT NOT NULL,
    status TEXT NOT NULL,
    error_kind TEXT,
    error TEXT,
    steps INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS rating_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    revision INTEGER NOT NULL,
    player_a TEXT NOT NULL,
    player_b TEXT NOT NULL,
    before_json TEXT NOT NULL,
    outcome TEXT NOT NULL,
    expected_a REAL,
    delta_a REAL NOT NULL,
    delta_b REAL NOT NULL,
    after_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(game_id, engine),
    FOREIGN KEY(run_id) REFERENCES runs(run_id),
    FOREIGN KEY(game_id) REFERENCES games(game_id)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    ratings_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence),
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load(value: str) -> Any:
    return json.loads(value)


class ArenaStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.db_path = self.root / "db" / "arena.sqlite3"

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.root.joinpath("catalog").mkdir(parents=True, exist_ok=True)
        self.root.joinpath("sources").mkdir(parents=True, exist_ok=True)
        self.root.joinpath("packages").mkdir(parents=True, exist_ok=True)
        self.root.joinpath("runs").mkdir(parents=True, exist_ok=True)
        self.root.joinpath("reports").mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def upsert_source(self, record: SourceRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                  competition=excluded.competition, ref=excluded.ref, version=excluded.version,
                  title=excluded.title, author=excluded.author, url=excluded.url,
                  metadata_json=excluded.metadata_json, source_hash=excluded.source_hash,
                  status=excluded.status, error=excluded.error, collected_at=excluded.collected_at""",
                (
                    record.source_id,
                    record.competition,
                    record.ref,
                    record.version,
                    record.title,
                    record.author,
                    record.url,
                    _dump(record.metadata),
                    record.source_hash,
                    record.status,
                    record.error,
                    record.collected_at,
                ),
            )

    def upsert_deck(self, record: DeckRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO decks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(deck_id) DO UPDATE SET
                  source_id=excluded.source_id, display_name=excluded.display_name,
                  archetype=excluded.archetype, primary_pokemon_json=excluded.primary_pokemon_json,
                  package_path=excluded.package_path, package_hash=excluded.package_hash,
                  role=excluded.role, status=excluded.status, metadata_json=excluded.metadata_json""",
                (
                    record.deck_id,
                    record.source_id,
                    record.display_name,
                    record.archetype,
                    _dump(record.primary_pokemon_ids),
                    record.package_path,
                    record.package_hash,
                    record.role,
                    record.status,
                    _dump(record.metadata),
                ),
            )

    def create_run(self, record: RunRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET phase=excluded.phase,
                config_json=excluded.config_json, started_at=excluded.started_at,
                finished_at=excluded.finished_at, status=excluded.status""",
                (
                    record.run_id,
                    record.phase,
                    _dump(record.config),
                    record.started_at,
                    record.finished_at,
                    record.status,
                ),
            )

    def load_run(self, run_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["config"] = _load(str(result.pop("config_json")))
        return result

    def record_game(self, record: GameRecord) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.game_id,
                    record.run_id,
                    record.player_a,
                    record.player_b,
                    int(record.player_a_first),
                    record.winner,
                    record.result,
                    record.status,
                    record.error_kind,
                    record.error,
                    record.steps,
                    _dump(record.payload),
                ),
            )
            if cursor.rowcount == 1:
                self._append_jsonl(record.run_id, "games.jsonl", record)

    def record_rating_event(self, record: RatingEvent) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO rating_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.event_id,
                    record.run_id,
                    record.game_id,
                    record.engine,
                    record.revision,
                    record.player_a,
                    record.player_b,
                    _dump(record.before),
                    record.outcome,
                    record.expected_a,
                    record.delta_a,
                    record.delta_b,
                    _dump(record.after),
                    record.created_at,
                ),
            )
            if cursor.rowcount == 1:
                self._append_jsonl(record.run_id, "rating_events.jsonl", record)

    def record_checkpoint(
        self,
        run_id: str,
        sequence: int,
        ratings: dict[str, object],
        created_at: str,
    ) -> None:
        checkpoint_id = f"{run_id}:{sequence:06d}"
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?)""",
                (checkpoint_id, run_id, sequence, _dump(ratings), created_at),
            )

    def load_latest_checkpoint(self, run_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT sequence, ratings_json, created_at FROM checkpoints WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "sequence": int(row[0]),
            "ratings": _load(str(row[1])),
            "created_at": str(row[2]),
        }

    def completed_game_ids(self, run_id: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT game_id FROM games WHERE run_id = ?", (run_id,)
            ).fetchall()
        return {str(row[0]) for row in rows}

    def load_games(self, run_id: str | None = None) -> tuple[dict[str, object], ...]:
        query = "SELECT * FROM games"
        values: tuple[object, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            values = (run_id,)
        query += " ORDER BY game_id"
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return tuple(self._game_row(row) for row in rows)

    def load_latest_ratings(self) -> tuple[dict[str, object], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM rating_events ORDER BY created_at DESC, event_id DESC"
            ).fetchall()
        seen: set[tuple[str, str, str]] = set()
        latest: list[dict[str, object]] = []
        for row in rows:
            players = (str(row[5]), str(row[6]))
            key = (str(row[3]), players[0], players[1])
            if key in seen:
                continue
            latest.append(self._rating_row(row))
            seen.add(key)
        return tuple(latest)

    def load_sources(self) -> tuple[dict[str, object], ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM sources ORDER BY source_id").fetchall()
        return tuple(
            {
                **dict(row),
                "metadata": _load(str(row["metadata_json"])),
            }
            for row in rows
        )

    def load_decks(self) -> tuple[dict[str, object], ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM decks ORDER BY deck_id").fetchall()
        return tuple(
            {
                **dict(row),
                "primary_pokemon_ids": tuple(_load(str(row["primary_pokemon_json"]))),
                "metadata": _load(str(row["metadata_json"])),
            }
            for row in rows
        )

    def load_rating_events(self, run_id: str | None = None) -> tuple[dict[str, object], ...]:
        query = "SELECT * FROM rating_events"
        values: tuple[object, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            values = (run_id,)
        query += " ORDER BY created_at, event_id"
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return tuple(self._rating_row(row) for row in rows)

    def _append_jsonl(self, run_id: str, filename: str, record: JsonRecord) -> None:
        path = self.root / "runs" / run_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_dump(record.to_json()) + "\n")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _game_row(row: sqlite3.Row) -> dict[str, object]:
        result = dict(row)
        result["player_a_first"] = bool(result["player_a_first"])
        result["payload"] = _load(str(result.pop("payload_json")))
        return result

    @staticmethod
    def _rating_row(row: sqlite3.Row) -> dict[str, object]:
        result = dict(row)
        result["before"] = _load(str(result.pop("before_json")))
        result["after"] = _load(str(result.pop("after_json")))
        return result
