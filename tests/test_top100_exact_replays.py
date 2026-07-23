from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from train.kaggle_bc_top20.training import download_expert_replays
from train.kaggle_bc_top20.training.audit_kaggle_bc_dataset import audit
from train.kaggle_bc_top20.training.build_kaggle_bc_dataset import build_aggregate_dataset
from train.kaggle_bc_top20.training.download_expert_replays import (
    _audit_replay,
    _discard_thread_api_client,
    _retry_after_seconds,
    eligible_episode_rows,
    load_exact_source_manifest,
    seed_replays_from_dataset_roots,
)
from train.kaggle_bc_top20.training.download_top100_exact_replays import (
    canonical_deck_sha256,
    episode_player_index,
    select_leaderboard_submission,
)
from train.alakazam_bc_rl.features import PTCGFeatureConfig, feature_config_for_schema


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
    def test_daily_dataset_seed_links_only_frozen_episode_ids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            daily = root / "pokemon-tcg-ai-battle-episodes-2026-07-22"
            output = root / "raw"
            daily.mkdir()
            (daily / "123.json").write_text("{}", encoding="utf-8")
            (daily / "999.json").write_text("{}", encoding="utf-8")
            report = seed_replays_from_dataset_roots(
                output,
                episode_ids=[123, 456],
                dataset_roots=[daily],
            )

            seeded = output / "episode-123-replay.json"
            self.assertTrue(seeded.is_symlink())
            self.assertEqual(seeded.resolve(), (daily / "123.json").resolve())
            self.assertFalse((output / "episode-999-replay.json").exists())
            self.assertEqual(report["seeded_episode_count"], 1)
            self.assertEqual(report["missing_episode_ids"], [456])

    def test_unauthorized_retry_can_discard_a_stale_thread_client(self) -> None:
        download_expert_replays._THREAD_LOCAL.client = object()
        _discard_thread_api_client()
        self.assertFalse(hasattr(download_expert_replays._THREAD_LOCAL, "client"))

    def test_request_slot_rechecks_a_concurrently_extended_deadline(self) -> None:
        download_expert_replays._NEXT_REQUEST_AT = 10.0
        monotonic = mock.Mock(side_effect=[0.0, 10.0, 20.0])
        sleeps: list[float] = []

        def record_sleep(delay: float) -> None:
            sleeps.append(delay)
            if len(sleeps) == 1:
                download_expert_replays._NEXT_REQUEST_AT = 20.0

        with mock.patch.object(download_expert_replays.time, "monotonic", monotonic):
            with mock.patch.object(download_expert_replays.time, "sleep", record_sleep):
                download_expert_replays._wait_for_request_slot(2.0)
        self.assertEqual(sleeps, [10.0, 10.0])
        self.assertEqual(download_expert_replays._NEXT_REQUEST_AT, 22.0)

    def test_retry_after_obeys_numeric_and_http_date_server_deadlines(self) -> None:
        numeric = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "115"}))
        dated = SimpleNamespace(
            response=SimpleNamespace(
                headers={"Retry-After": "Thu, 23 Jul 2026 05:27:45 GMT"}
            )
        )
        now = datetime(2026, 7, 23, 5, 25, 45, tzinfo=timezone.utc)
        self.assertEqual(_retry_after_seconds(numeric), 115.0)
        self.assertEqual(_retry_after_seconds(dated, now=now), 120.0)

    def test_exact_source_loader_and_episode_filter_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            deck = [5] * 60
            (root / "deck.csv").write_text("5\n" * 60, encoding="utf-8")
            from train.kaggle_bc_top20.training.download_expert_replays import canonical_deck_sha256

            source = {
                "schema_version": "ptcg_single_expert_bc_source_v1",
                "single_policy_constraint": True,
                "source_policy": {
                    "team_id": 11,
                    "submission_id": 22,
                    "team_name": "same display name",
                },
                "deck_file": "deck.csv",
                "deck_profile": {"deck_sha256": canonical_deck_sha256(deck)},
            }
            source_path = root / "source_manifest.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            loaded = load_exact_source_manifest(source_path)

        episodes = [
            SimpleNamespace(
                id=1,
                type="PUBLIC",
                state="COMPLETED",
                create_time="2026-07-23T00:00:00Z",
                end_time="2026-07-23T00:01:00Z",
                agents=[
                    SimpleNamespace(index=0, submission_id=99),
                    SimpleNamespace(index=1, submission_id=22),
                ],
            ),
            SimpleNamespace(
                id=2,
                type="PRIVATE",
                state="COMPLETED",
                agents=[SimpleNamespace(index=0, submission_id=22)],
            ),
            SimpleNamespace(
                id=3,
                type="PUBLIC",
                state="RUNNING",
                agents=[SimpleNamespace(index=0, submission_id=22)],
            ),
        ]
        rows = eligible_episode_rows(episodes, 22)
        self.assertEqual(loaded["_deck_sha256"], source["deck_profile"]["deck_sha256"])
        self.assertEqual([(row["episode_id"], row["player_index"]) for row in rows], [(1, 1)])

    def test_replay_audit_uses_exact_player_index_not_display_name(self) -> None:
        payload = {
            "info": {
                "EpisodeId": 123,
                "Agents": [{"Name": "duplicate"}, {"Name": "duplicate"}],
            },
            "rewards": [-1, 1],
            "steps": [
                [
                    {"visualize": [{"action": [[3] * 60, [5] * 60]}]},
                    {},
                ],
                [{}, {}],
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode-123-replay.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            row = _audit_replay(
                path,
                expected_episode_id=123,
                player_index=1,
                submission_id=22,
                team_id=11,
                team_name="duplicate",
                expected_deck=[5] * 60,
            )
        self.assertEqual(row["agent_index"], 1)
        self.assertEqual(row["expert_players"][0]["submission_id"], 22)

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

    def test_submission_date_allows_unique_one_millisecond_api_drift(self) -> None:
        leaderboard = SimpleNamespace(
            submission_date="2026-07-20T06:00:03.736Z",
            score="1015.8",
        )
        submissions = [
            SimpleNamespace(
                id=10,
                date_submitted="2026-07-20T06:00:03.737Z",
                public_score="1015.4",
            ),
            SimpleNamespace(
                id=11,
                date_submitted="2026-07-16T22:59:29.123Z",
                public_score="1015.8",
            ),
        ]
        selected, reason = select_leaderboard_submission(leaderboard, submissions)
        self.assertEqual(selected.id, 10)
        self.assertEqual(reason, "submission_date_nearest_2s")

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

    def test_universal_single_expert_data_manifest_passes_semantic_audit(self) -> None:
        observation = _observation(0)
        # Universal entity features repeat player-level Active conditions on
        # empty slots; those status-only rows do not imply a missing card ID.
        observation["current"]["players"][0]["poisoned"] = True
        payload = {
            "info": {"EpisodeId": 123, "Agents": [{"Name": "left"}, {"Name": "right"}]},
            "rewards": [1, -1],
            "steps": [
                [
                    {
                        "action": [],
                        "status": "ACTIVE",
                        "observation": observation,
                        "visualize": [{"action": [[5] * 60, [3] * 60]}],
                    },
                    {"action": [], "status": "INACTIVE", "observation": {}},
                ],
                [
                    {"action": [0], "status": "DONE", "observation": {}},
                    {"action": [], "status": "DONE", "observation": {}},
                ],
            ],
        }
        manifest = {
            "schema_version": "ptcg_single_expert_exact_replays_v2",
            "source_identity": {"team_id": 11, "submission_id": 22, "deck_sha256": "x"},
            "episodes": [
                {
                    "episode_id": 123,
                    "file": "episode-123-replay.json",
                    "split": "train",
                    "expert_players": [
                        {"player_index": 0, "submission_id": 22, "team_name": "left"}
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
                feature_config=feature_config_for_schema("ptcg_features_universal"),
                storage_path=root,
                min_free_gib=0,
            )
            report = audit(output, Path(summary["data_manifest"]), require_single_expert=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["feature_schemas"], {"ptcg_features_universal": 1})
        self.assertIn("low_decision_density", report["risk_flags"])


if __name__ == "__main__":
    unittest.main()
