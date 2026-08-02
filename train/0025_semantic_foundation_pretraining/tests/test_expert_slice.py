from __future__ import annotations

import importlib
import unittest


SLICE = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.data.expert_slice"
)
EXPECTED_DECK = "a" * 64


def _row(episode_id: int, team_name: str, *, deck: str = EXPECTED_DECK) -> dict:
    return {
        "episode_id": episode_id,
        "episode_date": "2026-07-28",
        "team_name": team_name,
        "source_id": {"James Cox": 73, "James Cox & Henry Chao": 74}.get(team_name, 99),
        "player_index": episode_id % 2,
        "first_player": episode_id % 2,
        "seat": "first",
        "terminal_outcome": "win",
        "payload_sha256": f"{episode_id:064x}",
        "deck_sha256": deck,
        "deck_counts": [[1, 59], [2, 1]],
        "split_rank_sha256": f"{episode_id + 10:064x}",
        "split": "validation" if episode_id == 2 else "train",
        "locator": {
            "kind": "zip_member",
            "archive": "/episodes/2026-07-28.zip",
            "member": f"episode-{episode_id}-replay.json",
        },
    }


def _catalog() -> dict:
    return {
        "schema_version": "test_catalog_v1",
        "catalog_sha256": "b" * 64,
        "contracts": {"selection": "winner"},
        "archives": [{"path": "/episodes/2026-07-28.zip", "date": "2026-07-28"}],
        "exclusions": [],
        "sources": [
            {"source_id": 73, "team_name": "James Cox"},
            {"source_id": 74, "team_name": "James Cox & Henry Chao"},
            {"source_id": 99, "team_name": "Someone Else"},
        ],
        "source_vocabulary_sha256": "c" * 64,
        "split": {"algorithm": "fixed", "seed": 7, "counts": {"train": 3}},
        "totals": {},
        "episodes": [
            _row(1, "James Cox"),
            _row(2, "James Cox & Henry Chao"),
            _row(3, "Someone Else"),
            _row(4, "James Cox", deck="d" * 64),
        ],
    }


class ExpertSliceTests(unittest.TestCase):
    def test_exact_sources_and_deck_are_preserved(self) -> None:
        sliced, audit = SLICE.slice_catalog(
            _catalog(),
            team_names=("James Cox", "James Cox & Henry Chao"),
            deck_sha256=EXPECTED_DECK,
            start_date="2026-07-20",
            end_date="2026-08-01",
        )
        self.assertEqual([row["episode_id"] for row in sliced["episodes"]], [1, 2])
        self.assertEqual(
            [row["team_name"] for row in sliced["sources"]],
            ["James Cox", "James Cox & Henry Chao"],
        )
        self.assertEqual(audit["totals"]["episodes"], 2)
        self.assertEqual(audit["totals"]["deck_count"], 1)
        self.assertEqual(audit["split_counts"], {"train": 1, "validation": 1})
        self.assertEqual(audit["source_counts"]["James Cox"], 1)
        self.assertEqual(audit["source_counts"]["James Cox & Henry Chao"], 1)
        self.assertEqual(sliced["parent_catalog_sha256"], "b" * 64)

    def test_duplicate_episode_fails_closed(self) -> None:
        catalog = _catalog()
        catalog["episodes"].append(dict(catalog["episodes"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate Episode"):
            SLICE.slice_catalog(
                catalog,
                team_names=("James Cox", "James Cox & Henry Chao"),
                deck_sha256=EXPECTED_DECK,
                start_date="2026-07-20",
                end_date="2026-08-01",
            )

    def test_incomplete_registered_deck_fails_closed(self) -> None:
        catalog = _catalog()
        catalog["episodes"][0]["deck_counts"] = [[1, 59]]
        with self.assertRaisesRegex(ValueError, "exactly 60"):
            SLICE.slice_catalog(
                catalog,
                team_names=("James Cox", "James Cox & Henry Chao"),
                deck_sha256=EXPECTED_DECK,
                start_date="2026-07-20",
                end_date="2026-08-01",
            )


if __name__ == "__main__":
    unittest.main()
