from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arena.models import DeckRecord, GameRecord, RatingEvent, RunRecord, SourceRecord
from arena.storage import ArenaStore


class ArenaStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_round_trips_source_deck_game_and_rating_event(self) -> None:
        store = ArenaStore(self.root)
        store.initialize()
        store.upsert_source(
            SourceRecord(
                "source-1",
                "pokemon-tcg-ai-battle",
                "u/k",
                1,
                "Title",
                "u",
                "https://www.kaggle.com/code/u/k",
                {},
                "hash",
                "exact_submission",
                None,
                "2026-07-22T00:00:00Z",
            )
        )
        store.upsert_deck(
            DeckRecord(
                "deck-1",
                "source-1",
                "Lucario - u",
                "Lucario",
                (1, 2),
                "arena/packages/deck-1",
                "pkg",
                "public",
                "exact_submission",
                {},
            )
        )
        store.create_run(
            RunRecord(
                "run-1",
                "smoke",
                {"seed": 7},
                "2026-07-22T00:00:00Z",
                None,
                "running",
            )
        )
        store.record_game(
            GameRecord(
                "game-1",
                "run-1",
                "deck-1",
                "internal",
                True,
                "deck-1",
                "win",
                "finished",
                None,
                None,
                4,
                {"candidate_first": True},
            )
        )
        store.record_rating_event(
            RatingEvent(
                "event-1",
                "run-1",
                "game-1",
                "elo_compat",
                1,
                "deck-1",
                "internal",
                {"mu": 600},
                "win",
                0.5,
                16,
                -16,
                {"mu": 616},
                "2026-07-22T00:00:01Z",
            )
        )

        self.assertEqual(store.completed_game_ids("run-1"), {"game-1"})
        self.assertEqual(store.load_games("run-1")[0]["result"], "win")
        self.assertEqual(store.load_latest_ratings()[0]["event_id"], "event-1")
        json.loads((self.root / "runs" / "run-1" / "games.jsonl").read_text())

    def test_completed_game_ids_are_unique_and_json_is_safe(self) -> None:
        store = ArenaStore(self.root)
        store.initialize()
        record = GameRecord(
            "game-1",
            "run-1",
            "a",
            "b",
            True,
            None,
            "error",
            "error",
            "worker",
            "boom",
            0,
            {"unicode": "胡地"},
        )
        store.record_game(record)
        store.record_game(record)

        self.assertEqual(store.completed_game_ids("run-1"), {"game-1"})
        record_json = (self.root / "runs" / "run-1" / "games.jsonl").read_text(encoding="utf-8")
        self.assertIn("胡地", record_json)

    def test_store_closes_sqlite_connection_after_operation(self) -> None:
        class FakeConnection:
            row_factory = None

            def __init__(self) -> None:
                self.closed = False

            def executescript(self, _: str) -> None:
                pass

            def __enter__(self) -> "FakeConnection":
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def close(self) -> None:
                self.closed = True

        connection = FakeConnection()
        with patch("arena.storage.sqlite3.connect", return_value=connection):
            ArenaStore(self.root).initialize()

        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
