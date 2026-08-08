from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ..data.archetypes import ArchetypeContract
from ..data import catalog as catalog_module
from ..data.catalog import _assign_splits, inspect_episode


REPO = Path(__file__).resolve().parents[3]


class ArchetypeContractTests(unittest.TestCase):
    def test_contract_has_fourteen_meta_classes_plus_other(self) -> None:
        contract = ArchetypeContract.load()
        self.assertEqual(len(contract.classes), 15)
        self.assertEqual(contract.other_class_id, 14)
        self.assertEqual(contract.classify([121] * 60), 0)
        self.assertEqual(contract.classify([648] * 60), 2)
        self.assertEqual(contract.classify([999] * 60), 14)

    def test_frozen_0806_pool_maps_to_fourteen_classes_plus_other(self) -> None:
        contract = ArchetypeContract.load()
        root = REPO / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks"
        labels = set()
        count = 0
        for deck_path in root.glob("*/deck.csv"):
            cards = [int(line.strip()) for line in deck_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(cards), 60)
            labels.add(contract.classify(cards))
            count += 1
        self.assertEqual(count, 55)
        self.assertEqual(labels, set(range(15)))

    def test_alternate_win_final_diff_uses_only_loser_remaining_prizes(self) -> None:
        players = [{"prize": [None] * 3}, {"prize": [None] * 5}]
        payload = {
            "info": {"EpisodeId": 123, "TeamNames": ["A", "B"]},
            "rewards": [1, -1],
            "statuses": ["DONE", "DONE"],
            "steps": [
                [{"visualize": [{"action": [[121] * 60, [648] * 60], "current": {"firstPlayer": 0}}]}, {}],
                [
                    {"observation": {"current": {"firstPlayer": 0, "yourIndex": 0, "players": players}}},
                    {"observation": {"current": {"firstPlayer": 0, "yourIndex": 1, "players": players}}},
                ],
            ],
        }
        row = inspect_episode(
            json.dumps(payload).encode(), date="2026-08-06", archive=Path("source.zip"),
            member="123.json", contract=ArchetypeContract.load(),
        )
        self.assertEqual([item["final_diff"] for item in row["trajectories"]], [5, -5])
        self.assertEqual([item["value_target"] for item in row["trajectories"]], [1, 0])

    def test_split_keeps_whole_episode_and_exact_validation_count(self) -> None:
        rows = [
            {
                "episode_id": index,
                "episode_date": "2026-08-06" if index % 2 else "2026-08-05",
                "payload_sha256": f"sha-{index}",
                "trajectories": [
                    {"opponent_archetype_target": index % 3},
                    {"opponent_archetype_target": (index + 1) % 3},
                ],
            }
            for index in range(100)
        ]
        _assign_splits(rows, 17)
        self.assertEqual(sum(row["split"] == "validation" for row in rows), 17)
        self.assertTrue(all(row["split"] in {"train", "validation"} for row in rows))

    def test_split_guarantees_validation_coverage_for_rare_classes(self) -> None:
        rows = []
        for index in range(100):
            rare = 14 if index == 99 else index % 3
            rows.append({
                "episode_id": index,
                "episode_date": "2026-08-06",
                "payload_sha256": f"sha-{index}",
                "trajectories": [
                    {"opponent_archetype_target": rare},
                    {"opponent_archetype_target": (index + 1) % 3},
                ],
            })
        _assign_splits(rows, 10)
        validation_classes = {
            item["opponent_archetype_target"]
            for row in rows if row["split"] == "validation"
            for item in row["trajectories"]
        }
        self.assertEqual(sum(row["split"] == "validation" for row in rows), 10)
        self.assertIn(14, validation_classes)

    def test_parallel_catalog_persists_contract_schema(self) -> None:
        class FakePool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def map(self, function, tasks):
                del function
                date, archive, _contract = list(tasks)[0]
                rows = [
                    {
                        "episode_id": episode_id,
                        "episode_date": date,
                        "payload_sha256": f"payload-{episode_id}",
                        "trajectories": [
                            {
                                "player_index": 0, "opponent_archetype_target": 0,
                                "value_target": 1, "terminal_outcome": "win",
                                "final_diff": 3, "final_diff_target": 9,
                            },
                            {
                                "player_index": 1, "opponent_archetype_target": 1,
                                "value_target": 0, "terminal_outcome": "loss",
                                "final_diff": -3, "final_diff_target": 3,
                            },
                        ],
                    }
                    for episode_id in (1, 2)
                ]
                return [(date, rows, {}, {"date": date, "path": archive, "bytes": 0, "sha256": "x", "qualified": 2})]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pokemon-tcg-ai-battle-episodes-2026-08-06.zip").touch()
            with patch.object(catalog_module, "ProcessPoolExecutor", FakePool):
                result = catalog_module.build_catalog(
                    root,
                    root / "catalog.json",
                    date_quotas={"2026-08-06": 2},
                    validation_count=1,
                )
                verified = catalog_module.verify_catalog(root / "catalog.json")
        self.assertEqual(result["archetype_contract"]["schema_version"], "0036_opponent_archetypes_v1")
        self.assertEqual(verified["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
