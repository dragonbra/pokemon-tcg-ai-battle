from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


CATALOG = import_module("train.0016_alakazam_multideck_bc.data.replay_catalog")
build_replay_catalog = CATALOG.build_replay_catalog
write_replay_catalog = CATALOG.write_replay_catalog


def _deck(*, battle_cage: int = 0, wondrous_patch: int = 0) -> list[int]:
    cards = [1264] * battle_cage + [1146] * wondrous_patch
    return cards + list(range(2000, 2000 + 60 - len(cards)))


def _deck_hash(cards: list[int]) -> str:
    payload = ",".join(str(card_id) for card_id in sorted(cards)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _replay(
    episode_id: int,
    teams: list[str],
    decks: list[list[int]],
    rewards: list[object],
    first_player: int = 0,
    statuses: list[str] | None = None,
) -> dict[str, object]:
    return {
        "info": {"EpisodeId": episode_id, "TeamNames": teams},
        "rewards": rewards,
        "statuses": statuses or ["DONE", "DONE"],
        "steps": [
            [
                {
                    "visualize": [
                        {"action": decks, "current": {"firstPlayer": first_player}}
                    ]
                },
                {"visualize": []},
            ]
        ],
    }


class ReplayCatalogTest(unittest.TestCase):
    def test_accepts_selected_done_winner_when_opponent_times_out(self) -> None:
        payload = _replay(
            99,
            ["timed-out opponent", "bono"],
            [_deck(), _deck()],
            [None, 1],
            first_player=0,
            statuses=["TIMEOUT", "DONE"],
        )
        raw = json.dumps(payload).encode("utf-8")
        record = CATALOG._episode_record(
            payload,
            expected_id=99,
            expected_team="bono",
            player_index=1,
            episode_date="2026-07-10",
            payload_sha256=hashlib.sha256(raw).hexdigest(),
            locator={"kind": "file", "archive": "", "member": "episode-99.json"},
        )
        self.assertEqual(record["outcome"], "win")

    def test_split_is_deterministic_with_validation_per_non_singleton_stratum(self) -> None:
        records = [
            {
                "episode_id": episode_id,
                "team_name": "同一来源",
                "source_id": 1,
                "seat": seat,
            }
            for episode_id, seat in ((1, "first"), (2, "first"), (3, "second"))
        ]
        first = [dict(record) for record in records]
        second = [dict(record) for record in reversed(records)]

        first_audit = CATALOG._assign_splits(first)
        second_audit = CATALOG._assign_splits(second)

        first_splits = {row["episode_id"]: row["split"] for row in first}
        second_splits = {row["episode_id"]: row["split"] for row in second}
        self.assertEqual(first_splits, second_splits)
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(
            sum(first_splits[episode_id] == "validation" for episode_id in (1, 2)),
            1,
        )
        self.assertEqual(first_splits[3], "train")

    def test_deduplicates_and_preserves_exact_unicode_source_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = root / "archives"
            archives.mkdir()
            archive_name = "episodes-2026-07-10.zip"
            first_deck = _deck(battle_cage=2, wondrous_patch=1)
            second_deck = _deck(battle_cage=4)
            first = _replay(101, ["对手", "goonew"], [_deck(), first_deck], [-1, 1])
            second = _replay(102, ["Yúshin Ito", "对手"], [second_deck, _deck()], [1, -1])
            with zipfile.ZipFile(archives / archive_name, "w") as bundle:
                bundle.writestr("nested/101.json", json.dumps(first))
                bundle.writestr("102.json", json.dumps(second))

            audit = {
                "errors": [],
                "episodes": [
                    {
                        "episode_id": 101,
                        "team_name": "goonew",
                        "player_index": 1,
                        "battle_cage_count": 2,
                        "wondrous_patch_count": 1,
                        "archives": [archive_name],
                    },
                    {
                        "episode_id": 102,
                        "team_name": "Yúshin Ito",
                        "player_index": 0,
                        "battle_cage_count": 4,
                        "wondrous_patch_count": 0,
                        "archives": [archive_name],
                    },
                ],
            }
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            goonew = root / "goonew"
            goonew.mkdir()
            replay_path = goonew / "episode-101-replay.json"
            raw = json.dumps(first).encode("utf-8")
            replay_path.write_bytes(raw)
            manifest = {
                "team_name": "goonew",
                "errors": [],
                "episodes": [
                    {
                        "episode_id": 101,
                        "player_index": 1,
                        "outcome": "win",
                        "create_time": "2026-07-10 12:00:00",
                        "file": replay_path.name,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "deck_sha256": _deck_hash(first_deck),
                        "battle_cage_count": 2,
                        "wondrous_patch_count": 1,
                    }
                ],
            }
            manifest_path = goonew / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            catalog = build_replay_catalog(audit_path, archives, manifest_path)

            self.assertEqual(catalog["totals"]["episode_count"], 2)
            self.assertEqual(catalog["totals"]["overlap_count"], 1)
            self.assertEqual(
                [source["team_name"] for source in catalog["sources"]],
                ["Yúshin Ito", "goonew"],
            )
            by_id = {row["episode_id"]: row for row in catalog["episodes"]}
            self.assertEqual(by_id[101]["source_id"], 2)
            self.assertEqual(by_id[101]["episode_date"], "2026-07-10")
            self.assertEqual(by_id[101]["seat"], "second")
            self.assertEqual(by_id[101]["split"], "train")
            self.assertEqual(by_id[101]["deck_counts"][0], [1146, 1])
            self.assertEqual(by_id[101]["locator"]["member"], "nested/101.json")
            self.assertEqual(len(by_id[101]["alternate_locators"]), 1)
            self.assertEqual(by_id[102]["battle_cage_count"], 4)

            output = root / "catalog.json"
            write_replay_catalog(catalog, output)
            decoded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                decoded["source_vocabulary_sha256"],
                catalog["source_vocabulary_sha256"],
            )
            self.assertNotIn("\\u", output.read_text(encoding="utf-8"))

    def test_rejects_exact_team_name_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = root / "archives"
            archives.mkdir()
            replay = _replay(201, ["Yushin Ito", "other"], [_deck(), _deck()], [1, -1])
            archive_name = "episodes-2026-07-10.zip"
            with zipfile.ZipFile(archives / archive_name, "w") as bundle:
                bundle.writestr("201.json", json.dumps(replay))
            audit_path = root / "audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "errors": [],
                        "episodes": [
                            {
                                "episode_id": 201,
                                "team_name": "yushin ito",
                                "player_index": 0,
                                "archives": [archive_name],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            goonew = root / "goonew"
            goonew.mkdir()
            manifest_path = goonew / "manifest.json"
            manifest_path.write_text(
                json.dumps({"team_name": "goonew", "errors": [], "episodes": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exact team name mismatch"):
                build_replay_catalog(audit_path, archives, manifest_path)


if __name__ == "__main__":
    unittest.main()
