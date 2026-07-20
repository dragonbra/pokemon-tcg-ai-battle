import json
from html import unescape
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


from scripts.replay_visualizer import (
    ReplayFormatError,
    attach_trace_metadata,
    extract_visualize_frames,
    load_replay,
)


class ReplayAdapterTests(unittest.TestCase):
    def test_extracts_kaggle_visualize_from_first_step(self):
        frames = [{"obs": "initial"}, {"obs": "after-action"}]
        record = {"steps": [[{"visualize": frames}]]}

        self.assertEqual(extract_visualize_frames(record), frames)

    def test_extracts_local_top_level_visualize_before_legacy_alias(self):
        frames = [{"state": 1}]
        record = {"visualize": frames, "visualize_frames": [{"state": 2}]}

        self.assertEqual(extract_visualize_frames(record), frames)

    def test_extracts_local_visualize_frames_alias(self):
        frames = [{"state": 1}]

        self.assertEqual(extract_visualize_frames({"visualize_frames": frames}), frames)

    def test_load_replay_records_source_and_original_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "kaggle.json"
            path.write_text(
                '{"steps": [[{"visualize": [{"state": 1}]}]]}',
                encoding="utf-8",
            )

            replay = load_replay(path)

        self.assertEqual(replay.path, path)
        self.assertEqual(replay.source, "kaggle-steps")
        self.assertEqual(replay.frames, [{"state": 1}])

    def test_rejects_replay_without_visualize_frames(self):
        for record in ({}, {"trace": [{"step": 1}]}, {"steps": []}):
            with self.subTest(record=record):
                with self.assertRaisesRegex(ReplayFormatError, "visualize"):
                    extract_visualize_frames(record)

    def test_rejects_non_object_frame(self):
        with self.assertRaisesRegex(ReplayFormatError, "帧"):
            extract_visualize_frames({"visualize": ["not-an-object"]})

    def test_load_replay_reports_invalid_json_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.json"
            path.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ReplayFormatError, str(path)):
                load_replay(path)

    def test_attach_trace_metadata_does_not_overwrite_existing_fields(self):
        frames = [{"obs": "existing", "action": [[9], [9]]}, {}]
        trace = [{"observation": {"turn": 1}, "action": [2]}]

        attach_trace_metadata(frames, trace)

        self.assertEqual(frames[0]["obs"], "existing")
        self.assertEqual(frames[0]["action"], [[9], [9]])
        self.assertEqual(frames[1]["obs"], {"turn": 1})
        self.assertEqual(frames[1]["action"], [[2], [2]])

    def test_attach_trace_metadata_places_action_on_observed_player(self):
        frames = [{"state": "initial"}, {}]
        trace = [
            {
                "observation": {"current": {"yourIndex": 1}},
                "action": [2],
            }
        ]

        attach_trace_metadata(frames, trace)

        self.assertEqual(frames[1]["action"], [[], [2]])

    def test_launcher_posts_json_to_viewer(self):
        from scripts.replay_visualizer import create_viewer_launcher

        frames = [{"state": "x", "text": "a'b"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            launcher = create_viewer_launcher(
                frames,
                Path(temp_dir) / "launcher.html",
                "https://viewer.test/replay",
            )
            html = launcher.read_text(encoding="utf-8")

        self.assertIn('action="https://viewer.test/replay"', html)
        self.assertIn('method="POST"', html)
        self.assertIn('name="json"', html)
        self.assertIn(json.dumps(frames, ensure_ascii=False), unescape(html))

    def test_cli_no_open_prints_launcher_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            replay = Path(temp_dir) / "replay.json"
            replay.write_text('{"visualize": [{"state": 1}]}', encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "scripts/visualize_replay.py", str(replay), "--no-open"],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn("launcher", result.stdout)
        self.assertIn("ptcgvis.heroz.jp/Visualizer/Replay/0", result.stdout)

    def test_cli_reports_old_trace_with_nonzero_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            replay = Path(temp_dir) / "trace.json"
            replay.write_text('{"trace": [{"step": 1}]}', encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "scripts/visualize_replay.py", str(replay), "--no-open"],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("--visualize-output", result.stderr)

    def test_build_visual_replay_keeps_trace_and_attaches_frames(self):
        from scripts.run_local_battle import _build_visual_replay

        result = {"agent0": "a", "agent1": "b", "trace": [{"action": [0]}]}
        frames = [{"state": "initial"}, {"state": "after"}]

        replay = _build_visual_replay(result, frames)

        self.assertEqual(replay["replay_format"], "ptcg-local-v1")
        self.assertEqual(replay["trace"], result["trace"])
        self.assertEqual(replay["visualize"][1]["action"], [[0], [0]])
