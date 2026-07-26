from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.metrics import GameMetric
from evaluation.metrics.registry import CORE_METRIC_IDS
from evaluation.packages.loader import SubmissionPackage
from evaluation.reporting import ReportData
from evaluation.runner.batch import (
    BatchConfig,
    ReportData as BatchReportData,
    _case_candidate,
    _finalize_formal_report,
    _metric_refs,
    _write_evaluation_backlink_atomic,
    _metric_registry,
    _summary,
    run_batch,
)
from evaluation.runner.models import GameResult
from evaluation.runtime.loader import compute_cg_manifest
from evaluation.traces.store import TraceStore


class TraceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.temp_root = self.root / "temporary"
        self.report_root = self.root / "report"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_rejects_report_root_inside_temp_root(self) -> None:
        with self.assertRaises(ValueError):
            TraceStore(self.temp_root, self.temp_root / "report")

    def test_summary_counts_finished_invalid_action_forfeit_as_completed(self) -> None:
        summary = _summary(
            (
                {
                    "opponent": "opponent",
                    "winner": 0,
                    "status": "finished",
                    "finished": True,
                    "error_kind": "opponent_error",
                },
            )
        )

        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["completed_games"], 1)
        self.assertEqual(summary["completion_rate"], 1.0)

    def test_cleanup_preserves_existing_shared_temp_root_and_unrelated_file(self) -> None:
        self.temp_root.mkdir()
        unrelated = self.temp_root / "caller-owned.txt"
        unrelated.write_text("keep", encoding="utf-8")

        store = TraceStore(self.temp_root, self.report_root)
        owned_root = store.temp_root
        self.assertNotEqual(owned_root, self.temp_root)
        store.temp_path("game-1").write_text("{}", encoding="utf-8")

        store.cleanup()

        self.assertTrue(self.temp_root.is_dir())
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")
        self.assertFalse(owned_root.exists())

    def test_write_game_record_persists_complete_trace_payload(self) -> None:
        store = TraceStore(self.temp_root, self.report_root)
        trace_path = store.temp_path("game-1")
        result = GameResult(
            game_id="game-1",
            opponent="fixture",
            candidate_first=True,
            candidate_physical_index=0,
            finished=True,
            winner=0,
            status="finished",
            error_kind=None,
            error=None,
            steps=1,
            trace_path=trace_path,
        )
        trace = {
            "run_id": "run-1",
            "trace": [{"observation": {"secret": "complete"}, "action": [0]}],
            "result": {"winner": 0},
            "metric_refs": {"fixture": {"status": "available", "value": 1}},
        }

        store.write_game_record(result, trace)

        self.assertEqual(json.loads(trace_path.read_text(encoding="utf-8")), trace)


class BatchRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_package(
        self,
        name: str,
        deck_card: int,
        *,
        crash_on_import: bool = False,
        hang_on_import: bool = False,
    ) -> SubmissionPackage:
        package_root = self.root / name
        cg_root = package_root / "cg"
        cg_root.mkdir(parents=True)
        (cg_root / "__init__.py").write_text("", encoding="utf-8")
        (cg_root / "api.py").write_text("API_VALUE = 1\n", encoding="utf-8")
        (cg_root / "libcg.so").write_bytes(b"fixture runtime")
        (cg_root / "game.py").write_text(self.game_source(), encoding="utf-8")
        deck = [deck_card] * 60
        (package_root / "deck.csv").write_text(
            "\n".join(str(card) for card in deck) + "\n",
            encoding="utf-8",
        )
        if crash_on_import:
            source = "import os\nos._exit(23)\n"
        elif hang_on_import:
            source = "import time\ntime.sleep(60)\n"
        else:
            source = self.agent_source(deck)
        (package_root / "main.py").write_text(source, encoding="utf-8")
        return SubmissionPackage(
            name=name,
            root=package_root,
            deck=deck,
            entrypoint=package_root / "main.py",
            package_hash=f"{name}-package-hash",
            deck_hash=f"{name}-deck-hash",
            cg_manifest=compute_cg_manifest(cg_root),
        )

    @staticmethod
    def agent_source(deck: list[int]) -> str:
        return (
            "def agent(observation):\n"
            "    if observation.get('select') is None:\n"
            f"        return {deck!r}\n"
            "    return [0]\n"
        )

    @staticmethod
    def game_source() -> str:
        return (
            "STEP = 0\n"
            "DECK0 = []\n\n"
            "def observation(result=-1):\n"
            "    return {\n"
            "        'current': {'turn': STEP, 'yourIndex': STEP % 2, 'result': result},\n"
            "        'select': None if result != -1 else {'type': 'choose', 'option': [0]},\n"
            "    }\n\n"
            "def battle_start(deck0, deck1):\n"
            "    global STEP, DECK0\n"
            "    STEP = 0\n"
            "    DECK0 = list(deck0)\n"
            "    return observation(), {}\n\n"
            "def battle_select(action):\n"
            "    global STEP\n"
            "    STEP += 1\n"
            "    if STEP == 3:\n"
            "        return observation(0 if DECK0[0] == 7 else 1)\n"
            "    return observation()\n\n"
            "def battle_finish():\n"
            "    return None\n"
        )

    def make_config(
        self,
        candidate: SubmissionPackage,
        opponents: tuple[SubmissionPackage, ...],
        *,
        games: int,
        worker_timeout_seconds: float = 30.0,
    ) -> BatchConfig:
        return BatchConfig(
            candidate=candidate,
            opponents=opponents,
            games_per_opponent=games,
            output_root=self.root / "reports",
            visualize=False,
            max_steps=10,
            control=None,
            plugin_ids=(),
            keep_temp=False,
            worker_timeout_seconds=worker_timeout_seconds,
        )

    def test_default_batch_metric_registry_includes_all_core_plugins(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)

        config = self.make_config(candidate, (opponent,), games=1)
        registry = _metric_registry(config.metric_module_paths)

        self.assertEqual(
            tuple(plugin.metric_id for plugin in registry.plugins),
            CORE_METRIC_IDS,
        )

    def test_metric_refs_include_lightweight_payload_and_denominators(self) -> None:
        metric = GameMetric(
            "fixture",
            "available",
            2,
            3,
            2 / 3,
            (),
            (),
            {"direction": "first", "nested": [1, 2]},
        )

        self.assertEqual(
            _metric_refs({"fixture": metric}),
            {
                "fixture": {
                    "status": "available",
                    "numerator": 2,
                    "denominator": 3,
                    "value": 2 / 3,
                    "payload": {"direction": "first", "nested": [1, 2]},
                }
            },
        )

    def test_batch_metric_registry_appends_dynamic_plugin(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        module_path = self.root / "extra_metric.py"
        module_path.write_text(
            """
from evaluation.metrics.base import AggregateMetric, GameMetric

class ExtraPlugin:
    metric_id = "batch_extra"

    def analyze_game(self, trace, context):
        return GameMetric(self.metric_id, "available", 0, 0, None, (), ())

    def aggregate(self, results):
        return AggregateMetric(self.metric_id, 0, 0, None, {})
""",
            encoding="utf-8",
        )

        config = replace(
            self.make_config(candidate, (opponent,), games=1),
            metric_module_paths=(str(module_path),),
        )
        registry = _metric_registry(config.metric_module_paths)

        self.assertEqual(
            tuple(plugin.metric_id for plugin in registry.plugins),
            (*CORE_METRIC_IDS, "batch_extra"),
        )

    def test_batch_rejects_ambiguous_or_core_overriding_metric_configuration(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        override_path = self.root / "override_metric.py"
        override_path.write_text(
            """
class OverridePlugin:
    metric_id = "outcome"

    def analyze_game(self, trace, context):
        return None

    def aggregate(self, results):
        return None
""",
            encoding="utf-8",
        )

        ambiguous = replace(
            self.make_config(candidate, (opponent,), games=1),
            plugin_ids=("outcome",),
            metric_module_paths=(str(override_path),),
        )
        with patch("evaluation.runner.batch._run_worker") as run_worker:
            with self.assertRaisesRegex(ValueError, "plugin_ids.*metric_module_paths"):
                run_batch(ambiguous)
        run_worker.assert_not_called()

        override = replace(
            self.make_config(candidate, (opponent,), games=1),
            metric_module_paths=(str(override_path),),
        )
        with patch("evaluation.runner.batch._run_worker") as run_worker:
            with self.assertRaisesRegex(ValueError, "core metric"):
                run_batch(override)
        run_worker.assert_not_called()

    def test_case_candidate_uses_metric_diagnostics_for_errors_and_deck_out(self) -> None:
        base_result = GameResult(
            game_id="fixture-001",
            opponent="fixture",
            candidate_first=True,
            candidate_physical_index=0,
            finished=True,
            winner=1,
            status="finished",
            error_kind=None,
            error=None,
            steps=3,
            trace_path=self.root / "fixture-001.json",
        )
        deck_out = GameMetric(
            metric_id="library_pressure",
            status="success",
            numerator=0,
            denominator=1,
            value=0,
            evidence=({"step": 3},),
            diagnostics=({"deck_out": True},),
        )
        library_case = _case_candidate(base_result, {"library_pressure": deck_out})

        visualization = GameMetric(
            metric_id="correctness",
            status="error",
            numerator=1,
            denominator=1,
            value="visualization_error",
            evidence=({"step": 3},),
            diagnostics=(),
        )
        visualization_case = _case_candidate(
            replace(base_result, winner=None),
            {"correctness": visualization},
        )

        self.assertEqual(library_case.failure_class, "library_deck_out")
        self.assertEqual(library_case.metric_ids, ("library_pressure",))
        self.assertEqual(visualization_case.failure_class, "visualization_error")
        self.assertFalse(visualization_case.is_loss)

    def test_batch_alternates_side_and_embeds_compact_records_in_report(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)

        result = run_batch(self.make_config(candidate, (opponent,), games=5))

        report_root = self.root / "reports" / result.run_id
        records = result.game_records
        manifest = result.manifest

        self.assertEqual(
            [record["swap"] for record in records],
            [False, True, False, True, False],
        )
        self.assertEqual([record["status"] for record in records], ["finished"] * 5)
        self.assertEqual([record["winner"] for record in records], [0] * 5)
        self.assertEqual(len(records), 5)
        self.assertTrue(all("observation" not in record for record in records))
        self.assertTrue(all(record["metric_refs"] for record in records))
        self.assertEqual(manifest["candidate"]["package_hash"], "candidate-package-hash")
        self.assertEqual(manifest["opponents"][0]["deck_hash"], "opponent-deck-hash")
        self.assertEqual(manifest["games"], 5)
        self.assertEqual(manifest["swap_policy"], "alternate_candidate_first")
        self.assertEqual(manifest["plugins"], list(CORE_METRIC_IDS))
        self.assertEqual(
            manifest["engine_runtime"]["cg_tree_hash"],
            candidate.cg_manifest["tree_hash"],
        )
        self.assertTrue(manifest["started_at"])
        self.assertTrue(manifest["finished_at"])
        self.assertEqual(manifest["trace_policy"]["retain_limit"], 0)
        self.assertEqual(manifest["artifact_policy"]["retained_files"], ["report.html"])
        self.assertGreater(manifest["wall_time_seconds"], 0)
        self.assertGreater(result.report_data.summary["performance"]["games_per_second"], 0)
        self.assertEqual(set(result.metric_results), set(CORE_METRIC_IDS))
        self.assertEqual(result.report_data.metrics, result.metric_results)
        self.assertEqual({path.name for path in report_root.iterdir()}, {"report.html"})

    def test_batch_rejects_arbitrary_explicit_report_before_workers(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        config = replace(
            self.make_config(candidate, (opponent,), games=1),
            report_path=self.root / "reports" / "V1_flat_contract.html",
            update_project_index=True,
        )

        with patch("evaluation.runner.batch._run_worker") as run_worker:
            with self.assertRaisesRegex(ValueError, "formal evaluation report path"):
                run_batch(config)
        run_worker.assert_not_called()

    def _formal_runtime_paths(self, version_name: str = "V1_initial_contract") -> SimpleNamespace:
        archive = self.root / "experiments" / "0013_fixture"
        run_root = self.root / "rl_runs" / "0013_fixture" / "versions" / version_name
        artifact = run_root / "artifact"
        return SimpleNamespace(
            project_id="0013_fixture",
            project_archive=archive,
            run_root=run_root,
            artifact=artifact,
            status=artifact / "status.json",
            evaluation=archive / "evaluation" / f"{version_name}.html",
            version_name=version_name,
        )

    def _initialize_formal_runtime(self, paths: SimpleNamespace) -> None:
        paths.project_archive.mkdir(parents=True, exist_ok=True)
        (paths.project_archive / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "ptcg_experiment_project_v1",
                    "project_id": "0013_fixture",
                    "objective": "fixture",
                    "deck": "fixture",
                    "expert_source": "fixture",
                    "dataset_contract": "fixture",
                    "engine_revision": "fixture",
                    "opponent_pool_snapshot": "fixture",
                    "status": "initialized",
                    "created_at": "2026-07-26T00:00:00+00:00",
                    "paths": {},
                }
            ),
            encoding="utf-8",
        )
        paths.artifact.mkdir(parents=True)
        paths.status.write_text(
            json.dumps({"state": "allocated", "version": paths.version_name}),
            encoding="utf-8",
        )

    def test_formal_preflight_rejects_uninitialized_runtime_before_workers(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        paths = self._formal_runtime_paths()
        config = replace(
            self.make_config(candidate, (opponent,), games=1),
            report_path=paths.evaluation,
            update_project_index=True,
        )

        with (
            patch("evaluation.runner.batch._formal_version_paths", return_value=paths),
            patch("evaluation.runner.batch._run_worker") as run_worker,
        ):
            with self.assertRaisesRegex(ValueError, "not initialized"):
                run_batch(config)
        run_worker.assert_not_called()

    def test_formal_preflight_rejects_invalid_manifest_or_status_before_workers(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        cases = (
            ("manifest_identity", {"project_id": "9999_wrong"}, None),
            ("manifest_fields", {"objective": ""}, None),
            ("status_state", None, {"state": "invented"}),
            ("status_empty", None, {"state": ""}),
        )
        for tag, manifest_update, status_update in cases:
            with self.subTest(tag=tag):
                paths = self._formal_runtime_paths(f"V1_{tag}")
                self._initialize_formal_runtime(paths)
                if manifest_update is not None:
                    manifest_path = paths.project_archive / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest.update(manifest_update)
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                if status_update is not None:
                    status = json.loads(paths.status.read_text(encoding="utf-8"))
                    status.update(status_update)
                    paths.status.write_text(json.dumps(status), encoding="utf-8")
                config = replace(
                    self.make_config(candidate, (opponent,), games=1),
                    report_path=paths.evaluation,
                )
                with (
                    patch("evaluation.runner.batch._formal_version_paths", return_value=paths),
                    patch("evaluation.runner.batch._run_worker") as run_worker,
                ):
                    with self.assertRaisesRegex(ValueError, "manifest|status"):
                        run_batch(config)
                run_worker.assert_not_called()

    def test_formal_preflight_rejects_report_and_backlink_collisions_before_workers(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        for collision in ("report", "backlink"):
            with self.subTest(collision=collision):
                paths = self._formal_runtime_paths(f"V1_{collision}")
                self._initialize_formal_runtime(paths)
                if collision == "report":
                    paths.evaluation.parent.mkdir(parents=True, exist_ok=True)
                    paths.evaluation.write_text("existing", encoding="utf-8")
                else:
                    (paths.artifact / "evaluation.json").write_text("{}", encoding="utf-8")
                config = replace(
                    self.make_config(candidate, (opponent,), games=1),
                    report_path=paths.evaluation,
                    update_project_index=True,
                )
                with (
                    patch("evaluation.runner.batch._formal_version_paths", return_value=paths),
                    patch("evaluation.runner.batch._run_worker") as run_worker,
                ):
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        run_batch(config)
                run_worker.assert_not_called()

    def test_formal_preflight_rejects_invalid_version_before_workers(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        report_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "0013_fixture"
            / "evaluation"
            / "invalid.html"
        )
        config = replace(
            self.make_config(candidate, (opponent,), games=1),
            report_path=report_path,
            update_project_index=True,
        )

        with patch("evaluation.runner.batch._run_worker") as run_worker:
            with self.assertRaisesRegex(ValueError, "invalid version"):
                run_batch(config)
        run_worker.assert_not_called()

    @staticmethod
    def _minimal_report_data() -> ReportData:
        return ReportData(
            manifest={"run_id": "run-fixture", "candidate": {"name": "candidate"}},
            summary={"total_games": 0},
            games=(),
            metrics={},
            cases=(),
        )

    def test_formal_finalization_orders_report_index_then_backlink(self) -> None:
        report_path = self.root / "evaluation" / "V1_order.html"
        paths = self._formal_runtime_paths("V1_order")
        events: list[str] = []

        with (
            patch(
                "evaluation.runner.batch.write_report_file_atomic",
                side_effect=lambda *_: events.append("report"),
            ),
            patch(
                "evaluation.runner.batch.write_evaluation_index",
                side_effect=lambda *_: events.append("index"),
            ),
            patch("evaluation.runner.batch._formal_version_paths", return_value=paths),
            patch(
                "evaluation.runner.batch._write_evaluation_backlink_atomic",
                side_effect=lambda *_args, **_kwargs: events.append("backlink"),
            ),
            patch.object(Path, "read_bytes", return_value=b"report"),
        ):
            _finalize_formal_report(
                self._minimal_report_data(),
                report_path,
                run_id="run-order",
                update_project_index=True,
            )

        self.assertEqual(events, ["report", "index", "backlink"])

    def test_formal_batch_config_cannot_disable_index_regeneration(self) -> None:
        report_path = self.root / "evaluation" / "V1_index_required.html"
        paths = self._formal_runtime_paths("V1_index_required")
        events: list[str] = []
        with (
            patch("evaluation.runner.batch.write_report_file_atomic"),
            patch(
                "evaluation.runner.batch.write_evaluation_index",
                side_effect=lambda *_: events.append("index"),
            ),
            patch("evaluation.runner.batch._formal_version_paths", return_value=paths),
            patch("evaluation.runner.batch._write_evaluation_backlink_atomic"),
            patch.object(Path, "read_bytes", return_value=b"report"),
        ):
            _finalize_formal_report(
                self._minimal_report_data(),
                report_path,
                run_id="run-index-required",
                update_project_index=False,
            )
        self.assertEqual(events, ["index"])

    def test_backlink_race_never_overwrites_concurrent_publisher(self) -> None:
        paths = self._formal_runtime_paths("V1_backlink_race")
        paths.artifact.mkdir(parents=True)
        paths.evaluation.parent.mkdir(parents=True)
        paths.evaluation.write_bytes(b"report")
        real_link = __import__("os").link

        def concurrent_link(source: Path, target: Path) -> None:
            target.write_text("concurrent winner", encoding="utf-8")
            real_link(source, target)

        with (
            patch("evaluation.runner.batch.os.link", side_effect=concurrent_link),
            patch.object(Path, "relative_to", return_value=Path("report.html")),
        ):
            with self.assertRaises(FileExistsError):
                _write_evaluation_backlink_atomic(
                    paths,
                    run_id="run-race",
                    report_sha256="hash",
                )

        backlink = paths.artifact / "evaluation.json"
        self.assertEqual(backlink.read_text(encoding="utf-8"), "concurrent winner")
        self.assertFalse(any(path.name.endswith(".tmp") for path in paths.artifact.iterdir()))

    def test_index_failure_retains_report_and_does_not_create_backlink(self) -> None:
        report_path = self.root / "evaluation" / "V1_index_failure.html"
        paths = self._formal_runtime_paths("V1_index_failure")

        with (
            patch("evaluation.runner.batch._formal_version_paths", return_value=paths),
            patch(
                "evaluation.runner.batch.write_evaluation_index",
                side_effect=OSError("index failed"),
            ),
            patch("evaluation.runner.batch._write_evaluation_backlink_atomic") as record_evaluation,
        ):
            with self.assertRaisesRegex(OSError, "index failed"):
                _finalize_formal_report(
                    self._minimal_report_data(),
                    report_path,
                    run_id="run-index",
                    update_project_index=True,
                )

        self.assertTrue(report_path.is_file())
        record_evaluation.assert_not_called()
        self.assertFalse((paths.artifact / "evaluation.json").exists())

    def test_backlink_failure_retains_report_and_index_without_false_backlink(self) -> None:
        report_path = self.root / "evaluation" / "V1_backlink_failure.html"
        paths = self._formal_runtime_paths("V1_backlink_failure")

        with (
            patch("evaluation.runner.batch._formal_version_paths", return_value=paths),
            patch(
                "evaluation.runner.batch._write_evaluation_backlink_atomic",
                side_effect=OSError("backlink failed"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "backlink failed"):
                _finalize_formal_report(
                    self._minimal_report_data(),
                    report_path,
                    run_id="run-backlink",
                    update_project_index=True,
                )

        self.assertTrue(report_path.is_file())
        self.assertTrue((report_path.parent / "index.html").is_file())
        self.assertFalse((paths.artifact / "evaluation.json").exists())

    def test_parallel_workers_bound_concurrency_and_preserve_record_order(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponents = (
            self.make_package("opponent-a", 8),
            self.make_package("opponent-b", 8),
        )
        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_worker(request, trace_path, _temp_root, _timeout_seconds, _cpu_threads):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            result = GameResult(
                game_id=request.game_id,
                opponent=request.opponent.name,
                candidate_first=request.candidate_first,
                candidate_physical_index=0 if request.candidate_first else 1,
                finished=True,
                winner=0,
                status="finished",
                error_kind=None,
                error=None,
                steps=3,
                trace_path=trace_path,
            )
            payload = {
                "game_id": result.game_id,
                "opponent": result.opponent,
                "candidate_first": result.candidate_first,
                "candidate_physical_index": result.candidate_physical_index,
                "finished": result.finished,
                "winner": result.winner,
                "status": result.status,
                "error_kind": result.error_kind,
                "error": result.error,
                "steps": result.steps,
            }
            trace_path.write_text(json.dumps({"trace": [], "result": payload}), encoding="utf-8")
            return result

        config = replace(
            self.make_config(candidate, opponents, games=2),
            workers=2,
            worker_cpu_threads=1,
        )
        with patch("evaluation.runner.batch._run_worker", side_effect=fake_worker):
            result = run_batch(config)

        self.assertEqual(peak, 2)
        self.assertEqual(
            [record["game_id"] for record in result.game_records],
            ["opponent-a-001", "opponent-a-002", "opponent-b-001", "opponent-b-002"],
        )
        self.assertEqual(result.manifest["workers"], 2)
        self.assertEqual(result.manifest["worker_cpu_threads"], 1)

    def test_auto_iteration_profile_writes_payloads_and_presentations(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        config = replace(
            self.make_config(candidate, (opponent,), games=1),
            metric_profile_id="auto_iteration_v8_setup_relay",
        )

        result = run_batch(config)
        report_root = self.root / "reports" / result.run_id
        manifest = result.manifest
        games = result.game_records
        metrics = result.metric_results

        self.assertEqual(manifest["metric_profile"]["id"], "auto_iteration_v8_setup_relay")
        self.assertEqual(manifest["metric_profile"]["revision"], 7)
        self.assertEqual(result.report_data.metric_profile["revision"], 7)
        self.assertIn("setup_relay", result.report_data.presentations)
        self.assertIn("attack_quality", result.report_data.presentations)
        self.assertIn("payload", metrics["setup_relay"])
        self.assertIn("payload", games[0]["metric_refs"]["setup_relay"])
        self.assertIn("Attack quality", (report_root / "report.html").read_text(encoding="utf-8"))

    def test_batch_writes_one_canonical_report_from_fake_worker_results(self) -> None:
        candidate = self.make_package("candidate", 7)
        control = self.make_package("control", 9)
        opponents = (
            self.make_package("opponent-a", 8),
            self.make_package("opponent-b", 8),
        )
        outcomes = {
            "opponent-a-001": (True, 0, "finished", None),
            "opponent-a-002": (True, 1, "finished", None),
            "opponent-b-001": (False, None, "unfinished", "step_limit"),
            "opponent-b-002": (False, None, "worker_crash", "worker_crash"),
        }

        def fake_worker(request, trace_path, _temp_root, _timeout_seconds, _cpu_threads):
            finished, winner, status, error_kind = outcomes[request.game_id]
            result = GameResult(
                game_id=request.game_id,
                opponent=request.opponent.name,
                candidate_first=request.candidate_first,
                candidate_physical_index=0 if request.candidate_first else 1,
                finished=finished,
                winner=winner,
                status=status,
                error_kind=error_kind,
                error=f"fixture {error_kind}" if error_kind else None,
                steps=2 if finished else 0,
                trace_path=trace_path,
            )
            payload = {
                "game_id": result.game_id,
                "opponent": result.opponent,
                "candidate_first": result.candidate_first,
                "candidate_physical_index": result.candidate_physical_index,
                "finished": result.finished,
                "winner": result.winner,
                "status": result.status,
                "error_kind": result.error_kind,
                "error": result.error,
                "steps": result.steps,
                "trace_path": str(trace_path),
            }
            trace_path.write_text(
                json.dumps({"trace": [], "result": payload}),
                encoding="utf-8",
            )
            return result

        config = replace(
            self.make_config(candidate, opponents, games=2),
            control=control,
        )
        with patch("evaluation.runner.batch._run_worker", side_effect=fake_worker):
            result = run_batch(config)

        report_root = self.root / "reports" / result.run_id
        self.assertEqual({path.name for path in report_root.iterdir()}, {"report.html"})
        self.assertIs(BatchReportData, ReportData)
        self.assertIsInstance(result.report_data, ReportData)
        self.assertEqual(set(result.report_data.metrics), set(CORE_METRIC_IDS))
        self.assertEqual(
            [record["candidate_first"] for record in result.report_data.games],
            [True, False, True, False],
        )

        manifest = result.manifest
        summary = result.report_data.summary
        metrics = result.metric_results
        games = result.game_records
        cases = result.case_records
        self.assertEqual(manifest, result.report_data.manifest)
        self.assertEqual(summary, result.report_data.summary)
        self.assertEqual(games, result.report_data.games)
        self.assertEqual(metrics, result.report_data.metrics)
        self.assertEqual(cases, result.report_data.cases)
        self.assertEqual(result.game_records, result.report_data.games)
        self.assertEqual(result.metric_results, result.report_data.metrics)
        self.assertEqual(result.case_records, result.report_data.cases)
        self.assertTrue(all("trace" not in game and "observation" not in game for game in games))

        self.assertEqual(
            {key: summary[key] for key in (
                "total_games",
                "wins",
                "losses",
                "draws",
                "errors",
                "unfinished",
                "completed_games",
                "win_rate",
                "completion_rate",
            )},
            {
                "total_games": 4,
                "wins": 1,
                "losses": 1,
                "draws": 0,
                "errors": 1,
                "unfinished": 1,
                "completed_games": 2,
                "win_rate": 0.25,
                "completion_rate": 0.5,
            },
        )
        self.assertEqual(
            summary["by_opponent"]["opponent-b"],
            {
                "games": 2,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "errors": 1,
                "unfinished": 1,
                "win_rate": 0.0,
            },
        )
        self.assertEqual(
            metrics["correctness"]["diagnostics"]["failure_classes"],
            {"unfinished": 1, "worker_crash": 1},
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["game_id"], "opponent-b-002")
        self.assertIsNone(cases[0]["trace_path"])

        html = (report_root / "report.html").read_text(encoding="utf-8")
        for expected in (*CORE_METRIC_IDS, *[package.name for package in opponents]):
            self.assertIn(expected, html)
        for expected in ("worker_crash", "opponent-b-002", "candidate", "control", "unavailable"):
            self.assertIn(expected, html)
        for forbidden in ("promotion", "reject", "revert"):
            self.assertNotIn(forbidden, html.lower())

    def test_worker_crash_is_recorded_and_later_games_continue(self) -> None:
        candidate = self.make_package("candidate", 7)
        crashed = self.make_package("crashed", 8, crash_on_import=True)
        healthy = self.make_package("healthy", 8)

        result = run_batch(self.make_config(candidate, (crashed, healthy), games=1))

        self.assertEqual(
            [record["status"] for record in result.game_records],
            ["worker_crash", "finished"],
        )
        self.assertEqual(result.game_records[0]["error_kind"], "worker_crash")
        self.assertEqual(result.game_records[1]["winner"], 0)

    def test_worker_timeout_is_recorded_and_later_games_continue(self) -> None:
        candidate = self.make_package("candidate", 7)
        hanging = self.make_package("hanging", 8, hang_on_import=True)
        healthy = self.make_package("healthy", 8)

        result = run_batch(
            self.make_config(
                candidate,
                (hanging, healthy),
                games=1,
                worker_timeout_seconds=0.5,
            )
        )

        self.assertEqual(
            [record["status"] for record in result.game_records],
            ["worker_crash", "finished"],
        )
        self.assertEqual(result.game_records[0]["error_kind"], "worker_crash")
        self.assertEqual(result.game_records[1]["winner"], 0)
        self.assertEqual(
            {path.name for path in (self.root / "reports" / result.run_id).iterdir()},
            {"report.html"},
        )

    def test_invalid_worker_trace_becomes_worker_crash_and_later_games_continue(self) -> None:
        candidate = self.make_package("candidate", 7)
        missing = self.make_package("missing", 8)
        malformed = self.make_package("malformed", 8)
        non_object = self.make_package("non-object", 8)
        malformed_object = self.make_package("malformed-object", 8)
        empty_result = self.make_package("empty-result", 8)
        missing_result_field = self.make_package("missing-result-field", 8)
        invalid_encoding = self.make_package("invalid-encoding", 8)
        conflicting = self.make_package("conflicting", 8)
        healthy = self.make_package("healthy", 8)

        def fake_worker_run(command: list[str], **_kwargs: object) -> object:
            request_path = Path(command[-2])
            result_path = Path(command[-1])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            game_id = request_payload["game_id"]
            trace_path = Path(request_payload["trace_path"])
            result_payload = {
                "game_id": game_id,
                "opponent": request_payload["opponent"]["name"],
                "candidate_first": request_payload["candidate_first"],
                "candidate_physical_index": 0,
                "finished": True,
                "winner": 0,
                "status": "finished",
                "error_kind": None,
                "error": None,
                "steps": 3,
                "trace_path": str(trace_path),
            }
            result_path.write_text(json.dumps(result_payload), encoding="utf-8")
            if game_id == "malformed-001":
                trace_path.write_text("{broken", encoding="utf-8")
            elif game_id == "non-object-001":
                trace_path.write_text("[]", encoding="utf-8")
            elif game_id == "malformed-object-001":
                trace_path.write_text(json.dumps({"trace": []}), encoding="utf-8")
            elif game_id == "empty-result-001":
                trace_path.write_text(
                    json.dumps({"trace": [], "result": {}}),
                    encoding="utf-8",
                )
            elif game_id == "missing-result-field-001":
                missing_steps = dict(result_payload)
                del missing_steps["steps"]
                trace_path.write_text(
                    json.dumps({"trace": [], "result": missing_steps}),
                    encoding="utf-8",
                )
            elif game_id == "invalid-encoding-001":
                trace_path.write_bytes(b"\xff")
            elif game_id == "conflicting-001":
                conflicting_result = dict(result_payload)
                conflicting_result["winner"] = 1
                trace_path.write_text(
                    json.dumps({"trace": [], "result": conflicting_result}),
                    encoding="utf-8",
                )
            elif game_id == "healthy-001":
                trace_path.write_text(
                    json.dumps({"trace": [], "result": result_payload}),
                    encoding="utf-8",
                )
            return __import__("subprocess").CompletedProcess(command, 0, "", "")

        with patch("evaluation.runner.batch.subprocess.run", side_effect=fake_worker_run):
            result = run_batch(
                self.make_config(
                    candidate,
                    (
                        missing,
                        malformed,
                        non_object,
                        malformed_object,
                        empty_result,
                        missing_result_field,
                        invalid_encoding,
                        conflicting,
                        healthy,
                    ),
                    games=1,
                )
            )

        self.assertEqual(
            [record["status"] for record in result.game_records],
            [
                "worker_crash",
                "worker_crash",
                "worker_crash",
                "worker_crash",
                "worker_crash",
                "worker_crash",
                "worker_crash",
                "worker_crash",
                "finished",
            ],
        )
        self.assertEqual(
            [record["error_kind"] for record in result.game_records[:8]],
            ["worker_crash"] * 8,
        )
        self.assertEqual(
            {path.name for path in (self.root / "reports" / result.run_id).iterdir()},
            {"report.html"},
        )
        self.assertEqual(result.game_records[-1]["winner"], 0)

    def test_batch_cleanup_removes_empty_run_specific_temp_parent(self) -> None:
        candidate = self.make_package("candidate", 7)
        opponent = self.make_package("opponent", 8)
        temp_base = self.root / "system-temp"

        with patch("evaluation.runner.batch.tempfile.gettempdir", return_value=str(temp_base)):
            result = run_batch(self.make_config(candidate, (opponent,), games=1))

        self.assertFalse((temp_base / "evaluation" / result.run_id).exists())


if __name__ == "__main__":
    unittest.main()
