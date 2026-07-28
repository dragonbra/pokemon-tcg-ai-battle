from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings
from unittest.mock import patch

from rl_environment.logging import TrainingLogger
from rl_environment.wandb_logging import (
    WandbSettings,
    WandbSink,
    create_wandb_sink_from_environment,
)


class FakeRun:
    def __init__(self, *, fail_log: bool = False, fail_finish: bool = False) -> None:
        self.fail_log = fail_log
        self.fail_finish = fail_finish
        self.defined_metrics: list[tuple[str, dict[str, object]]] = []
        self.logged: list[dict[str, object]] = []
        self.summary: dict[str, object] = {}
        self.finish_codes: list[int] = []

    def define_metric(self, name: str, **kwargs: object) -> None:
        self.defined_metrics.append((name, dict(kwargs)))

    def log(self, values: dict[str, object]) -> None:
        if self.fail_log:
            raise RuntimeError("injected upload failure")
        self.logged.append(dict(values))

    def finish(self, exit_code: int = 0) -> None:
        if self.fail_finish:
            raise RuntimeError("injected finish failure")
        self.finish_codes.append(exit_code)


class FakeWandb:
    def __init__(self, run: FakeRun, *, fail_init: bool = False) -> None:
        self.run = run
        self.fail_init = fail_init
        self.init_calls: list[dict[str, object]] = []
        self.settings_calls: list[dict[str, object]] = []

    def Settings(self, **kwargs: object) -> dict[str, object]:
        self.settings_calls.append(dict(kwargs))
        return dict(kwargs)

    def init(self, **kwargs: object) -> FakeRun:
        self.init_calls.append(dict(kwargs))
        if self.fail_init:
            raise RuntimeError("injected init failure")
        return self.run


class InspectingSink:
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path
        self.records: list[dict[str, object]] = []
        self.closed = False

    def log(self, record: dict[str, object]) -> None:
        on_disk = self.jsonl_path.read_text(encoding="utf-8").splitlines()
        if len(on_disk) != 1:
            raise AssertionError("canonical JSONL was not flushed before the mirror")
        self.records.append(dict(record))

    def close(self, exit_code: int = 0) -> None:
        self.closed = True


class RecordingWriter:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def add_scalar(self, *_: object) -> None:
        self.events.append("tensorboard")

    def flush(self) -> None:
        self.events.append("tensorboard_flush")

    def close(self) -> None:
        self.events.append("tensorboard_close")


class RecordingSink(InspectingSink):
    def __init__(self, jsonl_path: Path, events: list[str]) -> None:
        super().__init__(jsonl_path)
        self.events = events

    def log(self, record: dict[str, object]) -> None:
        super().log(record)
        self.events.append("wandb")


class WandbLoggingTests(unittest.TestCase):
    def _settings(
        self,
        directory: Path,
        *,
        job_type: str = "bc_train",
        mode: str = "offline",
    ) -> WandbSettings:
        return WandbSettings(
            mode=mode,  # type: ignore[arg-type]
            project="pokemon-tcg-policy-learning",
            entity=None,
            run_id="0001-test-v1-1234567890",
            name="V1_test",
            group="0001-test",
            job_type=job_type,
            tags=("smoke", "offline"),
            config={"seed": 7},
            directory=directory,
        )

    def test_sink_initializes_stable_offline_run_and_filters_history(self) -> None:
        with TemporaryDirectory() as directory:
            fake_run = FakeRun()
            fake_wandb = FakeWandb(fake_run)
            sink = WandbSink(self._settings(Path(directory)), sdk=fake_wandb)
            sink.log(
                {
                    "step": 3,
                    "timestamp": 123.0,
                    "train/loss": 0.25,
                    "train/enabled": True,
                    "nested": {"not": "uploaded"},
                    "note": "not uploaded as history",
                }
            )
            sink.close()

        init = fake_wandb.init_calls[0]
        self.assertEqual(init["id"], "0001-test-v1-1234567890")
        self.assertNotIn("resume", init)
        self.assertEqual(init["mode"], "offline")
        self.assertEqual(init["group"], "0001-test")
        self.assertEqual(init["job_type"], "bc_train")
        self.assertEqual(init["tags"], ["smoke", "offline"])
        self.assertEqual(fake_wandb.settings_calls[0]["console"], "wrap")
        self.assertEqual(
            init["settings"],
            {"console": "wrap", "disable_code": True},
        )
        self.assertEqual(
            fake_run.logged,
            [{"trainer/epoch": 3, "bc/train/loss": 0.25}],
        )
        self.assertIn(("trainer/epoch", {}), fake_run.defined_metrics)
        self.assertIn(
            ("progress/*", {"step_metric": "progress/iteration"}),
            fake_run.defined_metrics,
        )
        self.assertIn(
            (
                "bc/validation/exact_action",
                {"step_metric": "trainer/epoch"},
            ),
            fake_run.defined_metrics,
        )
        self.assertIn(
            ("rollout/rolling_500/*", {"step_metric": "env/episodes"}),
            fake_run.defined_metrics,
        )
        self.assertEqual(fake_run.finish_codes, [0])

    def test_online_run_uses_explicit_resume_identity_and_notes(self) -> None:
        with TemporaryDirectory() as directory:
            fake_run = FakeRun()
            fake_wandb = FakeWandb(fake_run)
            with patch.dict(os.environ, {"WANDB_NOTES": "中文实验说明"}):
                WandbSink(
                    self._settings(Path(directory), mode="online"),
                    sdk=fake_wandb,
                )

        self.assertEqual(fake_wandb.init_calls[0]["resume"], "allow")
        self.assertEqual(fake_wandb.init_calls[0]["notes"], "中文实验说明")

    def test_jsonl_is_flushed_before_explicit_mirror(self) -> None:
        with TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "metrics.jsonl"
            sink = InspectingSink(jsonl_path)
            with TrainingLogger(jsonl_path, wandb_sink=sink) as logger:
                record = logger.log(1, {"train/loss": 0.5})

            rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]

        self.assertEqual(rows, [record])
        self.assertEqual(sink.records, [record])
        self.assertTrue(sink.closed)

    def test_wandb_failure_does_not_lose_canonical_record(self) -> None:
        with TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "metrics.jsonl"
            fake_run = FakeRun(fail_log=True)
            sink = WandbSink(self._settings(Path(directory)), sdk=FakeWandb(fake_run))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with TrainingLogger(jsonl_path, wandb_sink=sink) as logger:
                    logger.log(1, {"train/loss": 0.5})

            rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["train/loss"], 0.5)
        self.assertTrue(any("W&B logging failed" in str(item.message) for item in caught))

    def test_ppo_history_uses_update_axis_and_stage_namespace(self) -> None:
        with TemporaryDirectory() as directory:
            fake_run = FakeRun()
            sink = WandbSink(
                self._settings(Path(directory), job_type="ppo_train"),
                sdk=FakeWandb(fake_run),
            )
            sink.log({"step": 4, "train/ppo_policy_loss": -0.125})

        self.assertEqual(
            fake_run.logged,
            [{"trainer/update": 4, "ppo/policy_loss": -0.125}],
        )

    def test_environment_settings_load_config_and_redact_secret_fields(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = output / "metrics.jsonl"
            (output / "config.json").write_text(
                json.dumps({"seed": 7, "access_token": "do-not-upload"}),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "WANDB_MODE": "offline",
                    "WANDB_PROJECT": "test-project",
                    "WANDB_TAGS": "smoke, synthetic",
                },
                clear=True,
            ):
                settings = WandbSettings.from_environment(
                    metrics,
                    {"train/bc_loss": 0.5},
                )

        self.assertEqual(settings.mode, "offline")
        self.assertEqual(settings.project, "test-project")
        self.assertEqual(settings.job_type, "bc_train")
        self.assertEqual(settings.tags, ("smoke", "synthetic"))
        self.assertEqual(settings.config["seed"], 7)
        self.assertEqual(settings.config["access_token"], "<redacted>")

    def test_online_environment_does_not_auto_track_noncanonical_test_path(self) -> None:
        with TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            with patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=True):
                sink = create_wandb_sink_from_environment(
                    metrics,
                    {"train/bc_loss": 0.5},
                )

        self.assertIsNone(sink)

    def test_canonical_nested_version_uses_sibling_wandb_staging(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = (
                root
                / "rl_runs"
                / "0013_semantic_goal_policy"
                / "versions"
                / "V1_initial_contract"
                / "artifact"
                / "training_metrics.jsonl"
            )
            with patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=True), patch(
                "rl_environment.wandb_logging.REPOSITORY_ROOT", root
            ), patch("rl_environment.wandb_logging.WandbSink", return_value="sink") as factory:
                settings = WandbSettings.from_environment(metrics, {"train/bc_loss": 0.5})
                allowed = create_wandb_sink_from_environment(metrics, {"train/bc_loss": 0.5})

        self.assertEqual(settings.group, "0013_semantic_goal_policy")
        self.assertEqual(
            settings.name,
            "0013 · semantic_goal_policy · V1_initial_contract",
        )
        self.assertEqual(
            settings.directory,
            root / "rl_runs" / "0013_semantic_goal_policy" / "versions" / "V1_initial_contract" / "wandb",
        )
        self.assertIs(allowed, factory.return_value)

    def test_automatic_inference_routes_rollout_and_eval_namespaces(self) -> None:
        cases = (
            ({"rollout/return": 0.5}, {"env/decisions": 2, "rollout/return": 0.5}),
            ({"eval/win_rate": 0.5}, {"env/episodes": 2, "eval/win_rate": 0.5}),
            (
                {"representation/entropy": 0.5},
                {"trainer/epoch": 2, "representation/entropy": 0.5},
            ),
        )
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"WANDB_MODE": "offline"}, clear=True
        ):
            for metrics, expected in cases:
                with self.subTest(metrics=metrics):
                    fake_run = FakeRun()
                    settings = WandbSettings.from_environment(
                        Path(directory) / "metrics.jsonl", metrics
                    )
                    sink = WandbSink(settings, sdk=FakeWandb(fake_run))
                    sink.log({"step": 2, **metrics})
                    self.assertEqual(fake_run.logged, [expected])

    def test_successful_close_persists_terminal_literal_status_schema(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            jsonl_path = artifact / "training_metrics.jsonl"
            sink = WandbSink(self._settings(Path(directory)), sdk=FakeWandb(FakeRun()))
            logger = TrainingLogger(jsonl_path, wandb_sink=sink)
            logger.close()
            status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))
            wandb = status["wandb"]

        self.assertEqual(wandb["state"], "synced")
        self.assertEqual(wandb["reason"], None)
        self.assertEqual(wandb["project"], "pokemon-tcg-policy-learning")
        self.assertIn("entity", wandb)
        self.assertEqual(wandb["run_id"], "0001-test-v1-1234567890")
        self.assertIn("url", wandb)

    def test_automatic_no_row_logger_resolves_sink_on_close(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = (
                root
                / "rl_runs"
                / "0013_semantic_goal_policy"
                / "versions"
                / "V1_initial_contract"
                / "artifact"
            )
            jsonl_path = artifact / "training_metrics.jsonl"
            fake_sink = WandbSink(self._settings(root), sdk=FakeWandb(FakeRun()))
            with patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=True), patch(
                "rl_environment.wandb_logging.REPOSITORY_ROOT", root
            ), patch(
                "rl_environment.wandb_logging.create_wandb_sink_from_environment",
                return_value=fake_sink,
            ) as factory:
                logger = TrainingLogger(jsonl_path)
                logger.close()
            status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))

        factory.assert_called_once_with(jsonl_path, {})
        self.assertEqual(status["wandb"]["state"], "synced")

    def test_sink_routes_canonical_namespaces_to_their_explicit_axes(self) -> None:
        cases = (
            (
                "bc_train",
                {
                    "progress/iteration": 17,
                    "progress/fraction": 0.5,
                    "progress/iterations_per_second": 8.0,
                    "trainer/epoch": 2,
                },
                {
                    "progress/iteration": 17,
                    "progress/fraction": 0.5,
                    "progress/iterations_per_second": 8.0,
                    "trainer/epoch": 2,
                },
            ),
            ("bc_train", {"train/loss": 0.5}, {"trainer/epoch": 2, "bc/train/loss": 0.5}),
            (
                "value_calibration",
                {"train/value_mae": 0.5},
                {"trainer/epoch": 2, "value/train/mae": 0.5},
            ),
            (
                "ppo_train",
                {"train/ppo_policy_loss": 0.5},
                {"trainer/update": 2, "ppo/policy_loss": 0.5},
            ),
            (
                "rollout_train",
                {"rollout/return": 0.5, "env/decisions": 7, "env/episodes": 3},
                {"env/decisions": 7, "rollout/return": 0.5, "env/episodes": 3},
            ),
            ("eval", {"eval/win_rate": 0.5}, {"env/episodes": 2, "eval/win_rate": 0.5}),
            (
                "representation",
                {"representation/entropy": 0.5},
                {"trainer/epoch": 2, "representation/entropy": 0.5},
            ),
            (
                "counterfactual",
                {"counterfactual/regret": 0.5},
                {"trainer/epoch": 2, "counterfactual/regret": 0.5},
            ),
            (
                "invariance",
                {"invariance/error": 0.5},
                {"trainer/epoch": 2, "invariance/error": 0.5},
            ),
        )
        with TemporaryDirectory() as directory:
            for job_type, metrics, expected in cases:
                with self.subTest(job_type=job_type):
                    fake_run = FakeRun()
                    sink = WandbSink(
                        self._settings(Path(directory), job_type=job_type),
                        sdk=FakeWandb(fake_run),
                    )
                    sink.log({"step": 2, **metrics})
                    self.assertEqual(fake_run.logged, [expected])

    def test_tensorboard_flushes_before_wandb_mirror(self) -> None:
        with TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "metrics.jsonl"
            events: list[str] = []
            logger = TrainingLogger(jsonl_path, wandb_sink=RecordingSink(jsonl_path, events))
            logger._writer = RecordingWriter(events)
            logger.log(1, {"train/loss": 0.5})
            logger.close()

        self.assertEqual(events[:2], ["tensorboard", "tensorboard_flush"])
        self.assertEqual(events[2], "wandb")

    def test_finish_failure_is_persisted_after_close_without_rows(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            jsonl_path = artifact / "training_metrics.jsonl"
            sink = WandbSink(
                self._settings(Path(directory)),
                sdk=FakeWandb(FakeRun(fail_finish=True)),
            )
            logger = TrainingLogger(jsonl_path, wandb_sink=sink)
            logger.close()
            status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(status["wandb"]["sync_state"], "failed")
        self.assertIn("injected finish failure", status["wandb"]["failure"])

    def test_init_failure_is_persisted_after_close_without_rows(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            jsonl_path = artifact / "training_metrics.jsonl"
            sink = WandbSink(
                self._settings(Path(directory)),
                sdk=FakeWandb(FakeRun(), fail_init=True),
            )
            logger = TrainingLogger(jsonl_path, wandb_sink=sink)
            logger.close()
            status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(status["wandb"]["sync_state"], "failed")
        self.assertIn("injected init failure", status["wandb"]["failure"])

    def test_malformed_canonical_path_is_not_auto_tracked(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "rl_runs" / "bad-project" / "versions" / "bad-version" / "artifact"
            metrics = malformed / "training_metrics.jsonl"
            with patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=True), patch(
                "rl_environment.wandb_logging.REPOSITORY_ROOT", root
            ):
                sink = create_wandb_sink_from_environment(metrics, {"train/bc_loss": 0.5})

        self.assertIsNone(sink)

    def test_remote_failure_persists_sync_status_after_local_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            jsonl_path = artifact / "training_metrics.jsonl"
            fake_run = FakeRun(fail_log=True)
            sink = WandbSink(self._settings(Path(directory)), sdk=FakeWandb(fake_run))
            with TrainingLogger(jsonl_path, wandb_sink=sink) as logger:
                logger.log(1, {"train/loss": 0.5})

            status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))
            rows = jsonl_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rows), 1)
        self.assertEqual(status["wandb"]["project"], "pokemon-tcg-policy-learning")
        self.assertEqual(status["wandb"]["run_id"], "0001-test-v1-1234567890")
        self.assertEqual(status["wandb"]["sync_state"], "failed")
        self.assertIn("injected upload failure", status["wandb"]["failure"])

    def test_logger_reports_exception_exit_code_to_wandb(self) -> None:
        with TemporaryDirectory() as directory:
            fake_run = FakeRun()
            sink = WandbSink(self._settings(Path(directory)), sdk=FakeWandb(fake_run))
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with TrainingLogger(Path(directory) / "metrics.jsonl", wandb_sink=sink):
                    raise RuntimeError("training failed")

        self.assertEqual(fake_run.finish_codes, [1])


if __name__ == "__main__":
    unittest.main()
