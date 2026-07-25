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
    def __init__(self, *, fail_log: bool = False) -> None:
        self.fail_log = fail_log
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
        self.finish_codes.append(exit_code)


class FakeWandb:
    def __init__(self, run: FakeRun) -> None:
        self.run = run
        self.init_calls: list[dict[str, object]] = []

    def init(self, **kwargs: object) -> FakeRun:
        self.init_calls.append(dict(kwargs))
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
        self.assertEqual(
            fake_run.logged,
            [{"trainer/epoch": 3, "bc/train/loss": 0.25}],
        )
        self.assertIn(("trainer/epoch", {}), fake_run.defined_metrics)
        self.assertEqual(fake_run.finish_codes, [0])

    def test_online_run_uses_explicit_resume_identity(self) -> None:
        with TemporaryDirectory() as directory:
            fake_run = FakeRun()
            fake_wandb = FakeWandb(fake_run)
            WandbSink(
                self._settings(Path(directory), mode="online"),
                sdk=fake_wandb,
            )

        self.assertEqual(fake_wandb.init_calls[0]["resume"], "allow")

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


if __name__ == "__main__":
    unittest.main()
