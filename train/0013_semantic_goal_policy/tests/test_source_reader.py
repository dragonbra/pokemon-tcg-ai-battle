from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

records_module = importlib.import_module("train.0013_semantic_goal_policy.data.records")
source_module = importlib.import_module("train.0013_semantic_goal_policy.data.source")
DecisionIdentity = records_module.DecisionIdentity
SourceIdentity = records_module.SourceIdentity
EpisodePatch = source_module.EpisodePatch
EpisodeSource = source_module.EpisodeSource
ReplayableEpisodeRecords = source_module.ReplayableEpisodeRecords
SourceAudit = source_module.SourceAudit
iter_canonical_episodes = source_module.iter_canonical_episodes
normalize_team_identity = source_module.normalize_team_identity


def _payload(
    episode_id: int,
    teams: list[object],
    rewards: list[object],
    *,
    marker: str = "archive",
    steps: list[object] | None = None,
    statuses: list[object] | None = None,
) -> bytes:
    if steps is None:
        steps = [
            [{"action": [], "status": "ACTIVE"}, {"action": [], "status": "ACTIVE"}],
            [{"action": [1], "status": "DONE"}, {"action": [2], "status": "DONE"}],
        ]
    return json.dumps(
        {
            "info": {"EpisodeId": episode_id, "TeamNames": teams},
            "rewards": rewards,
            "statuses": statuses if statuses is not None else ["DONE"] * len(teams),
            "steps": steps,
            "marker": marker,
        }
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _zip(self, name: str, members: list[tuple[str, bytes]]) -> Path:
        if name == "wrong-name.zip":
            path = self.root / name
        else:
            date = "2026-07-23" if "2026-07-23" in name else "2026-07-24"
            path = self.root / name / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as bundle:
            for member_name, payload in members:
                bundle.writestr(member_name, payload)
        return path

    def _source(
        self,
        archive: Path,
        *,
        date: str = "2026-07-24",
        patches: tuple[EpisodePatch, ...] = (),
        archive_hash: str | None = None,
    ) -> EpisodeSource:
        return EpisodeSource(date, archive, archive_hash or _sha256(archive), patches)

    def _patch(self, path: Path, episode_id: int, *, date: str = "2026-07-24") -> EpisodePatch:
        if path.parent == self.root:
            canonical = self.root / "patches" / date / path.name
            canonical.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, canonical)
            path = canonical
        return EpisodePatch(date, episode_id, path, _sha256(path))

    def _collect(self, sources: list[EpisodeSource]) -> list[object]:
        episodes: list[object] = []
        iter_canonical_episodes(sources, episodes.extend)
        return episodes

    def test_normalization_is_casefold_plus_whitespace_collapse_exact(self) -> None:
        self.assertEqual(normalize_team_identity("  YUSHIN\t Ito\n"), "yushin ito")
        self.assertNotEqual(normalize_team_identity("Yushin Ito 2"), "yushin ito")

    def test_filters_to_unique_winner_and_preserves_complete_trajectory(self) -> None:
        steps = [
            [{"action": [8], "status": "ACTIVE"}, {"action": [1], "status": "ACTIVE"}],
            [{"action": [7, 3], "status": "ACTIVE"}, {"action": [2], "status": "ACTIVE"}],
            [{"action": [6], "status": "DONE"}, {"action": [4], "status": "DONE"}],
        ]
        archive = self._zip(
            "day.zip",
            [
                ("101.json", _payload(101, ["other", "  YUSHIN\tITO "], [-1, 1], steps=steps)),
                ("102.json", _payload(102, ["Yushin Ito", "other"], [-1, 1])),
                ("103.json", _payload(103, ["Yushin Ito 2", "other"], [1, -1])),
            ],
        )
        episodes = self._collect([self._source(archive)])
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(episode.source, SourceIdentity("2026-07-24", 101, 1, True))
        self.assertEqual(tuple(row["action"] for row in episode.player_trajectory), ((1,), (2,), (4,)))
        with self.assertRaises(TypeError):
            episode.payload["steps"][0][1]["action"] = (99,)  # type: ignore[index]

    def test_draw_is_not_a_winner(self) -> None:
        archive = self._zip("day.zip", [("104.json", _payload(104, ["Yushin Ito", "other"], [0, 0]))])
        self.assertEqual(self._collect([self._source(archive)]), [])

    def test_patch_replaces_by_declared_canonical_id_before_filtering(self) -> None:
        archive = self._zip("day.zip", [("201.json", _payload(201, ["other", "Yushin Ito"], [-1, 1]))])
        patch_path = self.root / "episode-201-replay.json"
        patch_path.write_bytes(_payload(201, ["Yushin Ito", "other"], [1, -1], marker="patch"))
        episodes = self._collect([self._source(archive, patches=(self._patch(patch_path, 201),))])
        self.assertEqual(episodes[0].payload["marker"], "patch")
        self.assertEqual(episodes[0].source.player_index, 0)

    def test_gap_patch_is_emitted_when_archive_member_is_absent(self) -> None:
        archive = self._zip("day.zip", [("202.json", _payload(202, ["other", "team"], [-1, 1]))])
        patch_path = self.root / "episode-203-replay.json"
        patch_path.write_bytes(_payload(203, ["Yushin Ito", "other"], [1, -1]))
        episodes = self._collect([self._source(archive, patches=(self._patch(patch_path, 203),))])
        self.assertEqual([episode.source.episode_id for episode in episodes], [203])

    def test_duplicate_archive_patch_and_cross_date_ids_fail(self) -> None:
        duplicate = self._zip("duplicate.zip", [("301.json", _payload(301, ["other", "team"], [-1, 1])), ("nested/301.json", _payload(301, ["other", "team"], [-1, 1]))])
        with self.assertRaisesRegex(ValueError, "duplicate canonical episode ID 301"):
            iter_canonical_episodes([self._source(duplicate)], lambda _: None)

        archive = self._zip("patches.zip", [("302.json", _payload(302, ["other", "team"], [-1, 1]))])
        patch_path = self.root / "episode-302-replay.json"
        patch_path.write_bytes(_payload(302, ["Yushin Ito", "other"], [1, -1]))
        declared = self._patch(patch_path, 302)
        with self.assertRaisesRegex(ValueError, "duplicate patch declaration"):
            iter_canonical_episodes([self._source(archive, patches=(declared, declared))], lambda _: None)

        other = self._zip("2026-07-23/other.zip", [("302.json", _payload(302, ["other", "team"], [-1, 1]))])
        with self.assertRaisesRegex(ValueError, "duplicate canonical episode ID 302 across dates"):
            iter_canonical_episodes([self._source(archive), self._source(other, date="2026-07-23")], lambda _: None)

    def test_member_and_patch_declarations_bind_to_payload(self) -> None:
        mismatch = self._zip("mismatch.zip", [("401.json", _payload(402, ["other", "team"], [-1, 1]))])
        with self.assertRaisesRegex(ValueError, "archive member episode ID mismatch"):
            iter_canonical_episodes([self._source(mismatch)], lambda _: None)

        archive = self._zip("empty.zip", [("403.json", _payload(403, ["other", "team"], [-1, 1]))])
        patch_path = self.root / "episode-404-replay.json"
        patch_path.write_bytes(_payload(405, ["Yushin Ito", "other"], [1, -1]))
        canonical_patch = self._patch(patch_path, 404)
        bad_id = EpisodePatch(
            "2026-07-24",
            404,
            canonical_patch.path,
            canonical_patch.sha256,
        )
        with self.assertRaisesRegex(ValueError, "patch episode ID mismatch"):
            iter_canonical_episodes([self._source(archive, patches=(bad_id,))], lambda _: None)
        wrong_date = EpisodePatch(
            "2026-07-23",
            405,
            self.root / "patches" / "2026-07-23" / "episode-405-replay.json",
            _sha256(patch_path),
        )
        with self.assertRaisesRegex(ValueError, "patch date mismatch"):
            iter_canonical_episodes([self._source(archive, patches=(wrong_date,))], lambda _: None)

    def test_malformed_completion_and_dimensions_fail_closed(self) -> None:
        cases = {
            "empty steps": _payload(501, ["Yushin Ito", "other"], [1, -1], steps=[]),
            "scalar step": _payload(501, ["Yushin Ito", "other"], [1, -1], steps=[1]),
            "null member": _payload(501, ["Yushin Ito", "other"], [1, -1], steps=[[None, {}]]),
            "short step": _payload(501, ["Yushin Ito", "other"], [1, -1], steps=[[{}]]),
            "statuses dimension": _payload(501, ["Yushin Ito", "other"], [1, -1], statuses=["DONE"]),
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                archive = self._zip(f"{label}.zip", [("501.json", payload)])
                with self.assertRaises(ValueError):
                    self._collect([self._source(archive)])

    def test_hash_invalid_date_ambiguous_and_malformed_fail(self) -> None:
        archive = self._zip("day.zip", [("601.json", _payload(601, ["Yushin Ito", "other"], [1, -1]))])
        with self.assertRaisesRegex(ValueError, "archive SHA-256 mismatch"):
            iter_canonical_episodes([self._source(archive, archive_hash="0" * 64)], lambda _: None)
        with self.assertRaisesRegex(ValueError, "invalid source date"):
            iter_canonical_episodes([self._source(archive, date="2026-02-30")], lambda _: None)
        ambiguous = self._zip("ambiguous.zip", [("602.json", _payload(602, ["Yushin Ito", " yushin ito "], [1, -1]))])
        with self.assertRaisesRegex(ValueError, "ambiguous expert identity"):
            iter_canonical_episodes([self._source(ambiguous)], lambda _: None)
        malformed = self._zip("bad.zip", [("603.json", b"{bad")])
        with self.assertRaisesRegex(ValueError, "malformed episode JSON"):
            iter_canonical_episodes([self._source(malformed)], lambda _: None)

    def test_source_preflight_is_transactional_and_member_reads_are_streamed(self) -> None:
        archive = self._zip("stream.zip", [("701.json", _payload(701, ["Yushin Ito", "other"], [1, -1])), ("702.json", _payload(702, ["other", "team"], [-1, 1]))])
        original_read = zipfile.ZipExtFile.read
        calls: list[int] = []

        def bounded_read(handle: zipfile.ZipExtFile, size: int = -1) -> bytes:
            calls.append(size)
            self.assertNotEqual(size, -1, "member must not use unbounded read")
            return original_read(handle, size)

        with patch.object(zipfile.ZipExtFile, "read", bounded_read):
            episodes = self._collect([self._source(archive)])
        self.assertEqual(len(episodes), 1)
        self.assertTrue(calls)

        duplicate_patch = self.root / "episode-701-replay.json"
        duplicate_patch.write_bytes(_payload(701, ["Yushin Ito", "other"], [1, -1]))
        declared = self._patch(duplicate_patch, 701)
        with self.assertRaisesRegex(ValueError, "duplicate patch declaration"):
            iter_canonical_episodes(
                [self._source(archive, patches=(declared, declared))],
                lambda _: None,
            )

    def test_late_malformed_episode_commits_no_outputs(self) -> None:
        archive = self._zip(
            "pokemon-tcg-ai-battle-episodes-2026-07-24.zip",
            [
                ("901.json", _payload(901, ["Yushin Ito", "other"], [1, -1])),
                ("902.json", b"{bad"),
            ],
        )
        committed: list[object] = []
        with self.assertRaisesRegex(ValueError, "malformed episode JSON"):
            iter_canonical_episodes([self._source(archive)], committed.extend)
        self.assertEqual(committed, [])

    def test_ineligible_incomplete_episode_is_skipped_and_audited(self) -> None:
        incomplete = _payload(
            910,
            ["Yushin Ito", "other"],
            [1, None],
            statuses=["DONE", "TIMEOUT"],
            steps=[[{"status": "DONE"}, {"status": "TIMEOUT"}]],
        )
        archive = self._zip(
            "pokemon-tcg-ai-battle-episodes-2026-07-24.zip",
            [("910.json", incomplete)],
        )
        committed: list[object] = []
        audit = iter_canonical_episodes([self._source(archive)], committed.extend)
        self.assertEqual(committed, [])
        self.assertEqual(audit.seen, 1)
        self.assertEqual(audit.eligible, 0)
        self.assertEqual(audit.ineligible_noncomplete, 1)
        self.assertEqual(audit.spool_bytes, 0)

    def test_archive_and_patch_paths_bind_date_and_id_patterns(self) -> None:
        wrong_archive = self._zip(
            "wrong-name.zip",
            [("920.json", _payload(920, ["other", "team"], [-1, 1]))],
        )
        with self.assertRaisesRegex(ValueError, "archive path filename/date mismatch"):
            iter_canonical_episodes([self._source(wrong_archive)], lambda _: None)

        archive = self._zip(
            "pokemon-tcg-ai-battle-episodes-2026-07-24.zip",
            [("921.json", _payload(921, ["other", "team"], [-1, 1]))],
        )
        patch_dir = self.root / "wrong-date"
        patch_dir.mkdir()
        patch_path = patch_dir / "episode-922-replay.json"
        patch_path.write_bytes(_payload(922, ["Yushin Ito", "other"], [1, -1]))
        with self.assertRaisesRegex(ValueError, "patch path date mismatch"):
            iter_canonical_episodes(
                [self._source(archive, patches=(self._patch(patch_path, 922),))],
                lambda _: None,
            )

    def test_sink_receives_lazy_replayable_records_valid_only_during_call(self) -> None:
        archive = self._zip(
            "lazy.zip",
            [(f"{930 + i}.json", _payload(930 + i, ["Yushin Ito", "other"], [1, -1])) for i in range(3)],
        )
        captured: list[object] = []
        retained_iterators: list[object] = []

        def sink(records: object) -> None:
            self.assertIsInstance(records, ReplayableEpisodeRecords)
            self.assertNotIsInstance(records, tuple)
            first = [episode.source.episode_id for episode in records]  # type: ignore[union-attr]
            second = [episode.source.episode_id for episode in records]  # type: ignore[union-attr]
            self.assertEqual(first, second)
            retained = iter(records)  # type: ignore[arg-type]
            next(retained)
            retained_iterators.append(retained)
            captured.append(records)

        iter_canonical_episodes([self._source(archive)], sink)
        with self.assertRaisesRegex(RuntimeError, "no longer valid"):
            list(captured[0])  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "no longer valid"):
            next(retained_iterators[0])  # type: ignore[arg-type]

    def test_episode_size_cap_covers_declared_historical_maximum(self) -> None:
        self.assertEqual(source_module.MAX_EPISODE_BYTES, 320 * 1024 * 1024)
        self.assertGreater(source_module.MAX_EPISODE_BYTES, 269_741_493)

    def test_spool_cap_and_free_space_fail_before_sink(self) -> None:
        archive = self._zip(
            "limits.zip",
            [("940.json", _payload(940, ["Yushin Ito", "other"], [1, -1]))],
        )
        calls: list[object] = []
        with patch.object(source_module, "MAX_SPOOL_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "spool byte cap"):
                iter_canonical_episodes([self._source(archive)], calls.append)
        self.assertEqual(calls, [])

        usage = shutil.disk_usage(self.root)
        fake_usage = type(usage)(usage.total, usage.used, 0)
        with patch.object(source_module.shutil, "disk_usage", return_value=fake_usage):
            with self.assertRaisesRegex(OSError, "insufficient temporary free space"):
                iter_canonical_episodes([self._source(archive)], calls.append)
        self.assertEqual(calls, [])

    def test_tempfile_cleanup_on_success_and_sink_exception(self) -> None:
        archive = self._zip(
            "cleanup.zip",
            [("950.json", _payload(950, ["Yushin Ito", "other"], [1, -1]))],
        )
        opened: list[object] = []
        original = tempfile.TemporaryFile

        def tracked_tempfile(*args: object, **kwargs: object) -> object:
            handle = original(*args, **kwargs)
            opened.append(handle)
            return handle

        with patch.object(source_module.tempfile, "TemporaryFile", tracked_tempfile):
            iter_canonical_episodes([self._source(archive)], lambda records: list(records))
        self.assertTrue(opened[-1].closed)  # type: ignore[union-attr]

        def failing_sink(records: object) -> None:
            next(iter(records))  # type: ignore[arg-type]
            raise RuntimeError("sink failed")

        with patch.object(source_module.tempfile, "TemporaryFile", tracked_tempfile):
            with self.assertRaisesRegex(RuntimeError, "sink failed"):
                iter_canonical_episodes([self._source(archive)], failing_sink)
        self.assertTrue(opened[-1].closed)  # type: ignore[union-attr]

    def test_parallel_reader_preserves_canonical_order_and_audit(self) -> None:
        archives=[]
        for date,start in (("2026-07-23",970),("2026-07-24",980)):
            archive=self._zip(f"{date}/parallel.zip",[(f"{start+i}.json",_payload(start+i,["Yushin Ito","other"],[1,-1])) for i in range(3)])
            archives.append(self._source(archive,date=date))
        serial=[];parallel=[]
        serial_audit=iter_canonical_episodes(archives,lambda values:serial.extend((e.source.date,e.source.episode_id) for e in values))
        parallel_audit=source_module.iter_canonical_episodes_parallel(archives,lambda values:parallel.extend((e.source.date,e.source.episode_id) for e in values),workers=2)
        self.assertEqual(parallel,serial)
        self.assertEqual(parallel_audit,serial_audit)

    def test_identity_keys_include_date_and_submission_is_unavailable(self) -> None:
        source = SourceIdentity("2026-07-24", 800, 0, True)
        decision = DecisionIdentity(source, 7)
        self.assertEqual(source.key, ("2026-07-24", 800, 0))
        self.assertEqual(decision.key, ("2026-07-24", 800, 0, 7))
        with self.assertRaises(ValueError):
            SourceIdentity("2026-07-24", 800, 0, False)


if __name__ == "__main__":
    unittest.main()
