from __future__ import annotations

import json
import unittest
from pathlib import Path

from ..data.archetypes import ArchetypeContract
from ..data.dragapult_catalog import _assign_splits, _select_weighted_dates, inspect_registration_header


REPO = Path(__file__).resolve().parents[3]
EXACT_007 = REPO / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks/dragapult_ex_07bedfffbfad/deck.csv"


def replay(decks: list[list[int]], winner: int = 0, episode_id: int = 123) -> bytes:
    payload = {
        "info": {"EpisodeId": episode_id, "TeamNames": ["A", "B"]},
        "rewards": [1 if actor == winner else -1 for actor in (0, 1)],
        "statuses": ["DONE", "DONE"],
        "steps": [[{
            "visualize": [{"action": decks, "current": {"firstPlayer": -1}}]
        }, {}]],
    }
    return json.dumps(payload, separators=(", ", ": ")).encode()


class DragapultCatalogTests(unittest.TestCase):
    @staticmethod
    def selection_rows(counts: dict[str, int]) -> list[dict[str, object]]:
        rows = []
        episode_id = 1
        for date, count in counts.items():
            for _ in range(count):
                rows.append({
                    "episode_date": date,
                    "episode_id": episode_id,
                    "payload_sha256": f"{episode_id:064x}",
                })
                episode_id += 1
        return rows

    def test_only_dragapult_actor_is_focal(self) -> None:
        row = inspect_registration_header(
            replay([[2] * 60, [121] * 3 + [5] * 57], winner=1),
            date="2026-08-06", archive=Path("daily.zip"), member="123.json",
            contract=ArchetypeContract.load(), exact_007_path=EXACT_007,
        )
        self.assertEqual([item["player_index"] for item in row["trajectories"]], [1])
        self.assertEqual(row["trajectories"][0]["value_target"], 1)

    def test_dragapult_mirror_keeps_both_focal_actors(self) -> None:
        row = inspect_registration_header(
            replay([[121] * 3 + [5] * 57, [121] * 4 + [7] * 56]),
            date="2026-08-06", archive=Path("daily.zip"), member="123.json",
            contract=ArchetypeContract.load(), exact_007_path=EXACT_007,
        )
        self.assertEqual([item["player_index"] for item in row["trajectories"]], [0, 1])

    def test_exact_007_requires_complete_multiset_equality(self) -> None:
        exact = [int(line) for line in EXACT_007.read_text().splitlines()]
        near = list(exact)
        near[-1] = 121
        row = inspect_registration_header(
            replay([exact, near]), date="2026-08-06", archive=Path("daily.zip"),
            member="123.json", contract=ArchetypeContract.load(), exact_007_path=EXACT_007,
        )
        self.assertEqual([item["is_exact_007"] for item in row["trajectories"]], [True, False])

    def test_weighted_selection_covers_every_date_and_exact_target(self) -> None:
        rows = self.selection_rows({
            "2026-07-10": 100,
            "2026-07-20": 100,
            "2026-08-06": 100,
        })
        selected, audit = _select_weighted_dates(rows, 120)
        self.assertEqual(len(selected), 120)
        self.assertEqual(len({row["episode_id"] for row in selected}), 120)
        counts = {row["date"]: row["selected_episodes"] for row in audit}
        self.assertGreaterEqual(min(counts.values()), 1)
        self.assertGreater(counts["2026-08-06"], counts["2026-07-10"])

    def test_weighted_selection_redistributes_sparse_date_capacity(self) -> None:
        rows = self.selection_rows({"2026-07-10": 2, "2026-08-01": 100, "2026-08-06": 100})
        selected, audit = _select_weighted_dates(rows, 150)
        counts = {row["date"]: row["selected_episodes"] for row in audit}
        self.assertEqual(len(selected), 150)
        self.assertEqual(counts["2026-07-10"], 2)
        self.assertEqual(sum(counts.values()), 150)

    def test_split_forces_rare_observed_class_into_validation(self) -> None:
        rows = []
        for episode_id in range(100):
            rows.append({
                "episode_id": episode_id,
                "episode_date": "2026-08-06",
                "payload_sha256": f"{episode_id:064x}",
                "trajectories": [{
                    "value_target": episode_id % 2,
                    "is_exact_007": False,
                    "opponent_archetype_target": 8 if episode_id == 99 else 0,
                }],
            })
        _assign_splits(rows, 10)
        validation_classes = {
            item["opponent_archetype_target"]
            for row in rows if row["split"] == "validation"
            for item in row["trajectories"]
        }
        self.assertEqual(validation_classes, {0, 8})
        self.assertEqual(sum(row["split"] == "validation" for row in rows), 10)


if __name__ == "__main__":
    unittest.main()
