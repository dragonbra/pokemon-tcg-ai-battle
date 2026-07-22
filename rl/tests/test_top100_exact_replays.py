from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from rl.ptcg.audit_kaggle_bc_dataset import audit
from rl.ptcg.build_kaggle_bc_dataset import build_aggregate_dataset
from rl.ptcg.download_top100_exact_replays import (
    canonical_deck_sha256,
    episode_player_index,
    select_leaderboard_submission,
)
from rl.ptcg.features import PTCGFeatureConfig


def _observation(player_index: int) -> dict:
    return {
        "current": {
            "yourIndex": player_index,
            "players": [
                {"active": [], "bench": [], "hand": [], "discard": []},
                {"active": [], "bench": [], "hand": [], "discard": []},
            ],
        },
        "select": {
            "type": 1,
            "context": 7,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 1}, {"type": 2}],
        },
    }


class Top100ExactReplayTests(unittest.TestCase):
    def test_deck_hash_uses_multiset_not_source_order(self) -> None:
        left = list(range(60))
        self.assertEqual(canonical_deck_sha256(left), canonical_deck_sha256(reversed(left)))
        changed = list(left)
        changed[-1] = 58
        self.assertNotEqual(canonical_deck_sha256(left), canonical_deck_sha256(changed))

    def test_submission_date_resolves_score_refresh_race(self) -> None:
        leaderboard = SimpleNamespace(
            submission_date="2026-07-22T17:05:25.453Z",
            score="1002.4",
        )
        submissions = [
            SimpleNamespace(
                id=10,
                date_submitted="2026-07-22T17:05:25.453Z",
                public_score="751.6",
            ),
            SimpleNamespace(
                id=11,
                date_submitted="2026-07-22T17:05:10.433Z",
                public_score="983.0",
            ),
        ]
        selected, reason = select_leaderboard_submission(leaderboard, submissions)
        self.assertEqual(selected.id, 10)
        self.assertEqual(reason, "submission_date")

    def test_episode_player_index_uses_submission_id(self) -> None:
        episode = SimpleNamespace(
            id=123,
            agents=[
                SimpleNamespace(index=0, submission_id=10),
                SimpleNamespace(index=1, submission_id=20),
            ],
        )
        self.assertEqual(episode_player_index(episode, 20), 1)

    def test_aggregate_dataset_keeps_two_experts_from_one_episode(self) -> None:
        payload = {
            "info": {
                "EpisodeId": 123,
                "Agents": [{"Name": "left"}, {"Name": "right"}],
            },
            "rewards": [1, -1],
            "steps": [
                [
                    {
                        "action": [],
                        "status": "ACTIVE",
                        "observation": _observation(0),
                        "visualize": [{"action": [[5] * 60, [5] * 60]}],
                    },
                    {"action": [], "status": "INACTIVE", "observation": {}},
                ],
                [
                    {"action": [0], "status": "INACTIVE", "observation": {}},
                    {
                        "action": [],
                        "status": "ACTIVE",
                        "observation": _observation(1),
                    },
                ],
                [
                    {"action": [], "status": "DONE", "observation": {}},
                    {"action": [1], "status": "DONE", "observation": {}},
                ],
            ],
        }
        manifest = {
            "schema_version": "ptcg_top_ladder_exact_deck_replays_v1",
            "episodes": [
                {
                    "episode_id": 123,
                    "file": "episode-123-replay.json",
                    "split": "train",
                    "expert_players": [
                        {"player_index": 0, "submission_id": 10, "team_name": "left"},
                        {"player_index": 1, "submission_id": 20, "team_name": "right"},
                    ],
                }
            ],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode-123-replay.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "dataset.jsonl"
            summary = build_aggregate_dataset(
                root,
                manifest,
                output,
                feature_config=PTCGFeatureConfig(),
                storage_path=root,
                min_free_gib=0,
            )
            records = [json.loads(line) for line in output.read_text().splitlines()]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audit_report = audit(output, manifest_path)
        self.assertEqual(summary["unique_replays"], 1)
        self.assertEqual(summary["expert_trajectories"], 2)
        self.assertEqual(len(records), 2)
        self.assertEqual({row["submission_id"] for row in records}, {10, 20})
        self.assertEqual({row["split"] for row in records}, {"train"})
        self.assertEqual(audit_report["status"], "passed")
        self.assertEqual(audit_report["violations"], {})


if __name__ == "__main__":
    unittest.main()
