"""Optional W&B mirror for canonical local training metrics."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal
import warnings

from rl_environment.runs import PROJECT_ID, REPOSITORY_ROOT, VERSIONED_ATTEMPT, WANDB_PROJECT, wandb_run_id


WandbMode = Literal["disabled", "offline", "online"]


@dataclass(frozen=True)
class WandbSettings:
    """Non-secret settings for one W&B lifecycle run."""

    mode: WandbMode
    project: str
    entity: str | None
    run_id: str
    name: str
    group: str
    job_type: str
    tags: tuple[str, ...]
    config: dict[str, Any]
    directory: Path

    @classmethod
    def from_environment(
        cls,
        jsonl_path: str | Path,
        metrics: dict[str, Any],
    ) -> "WandbSettings":
        """Build safe settings without reading or persisting the API key."""
        path = Path(jsonl_path).resolve()
        raw_mode = os.environ.get("WANDB_MODE", "disabled").strip().lower()
        if raw_mode not in {"disabled", "offline", "online"}:
            raise ValueError(
                "WANDB_MODE must be disabled, offline, or online; "
                f"received {raw_mode!r}"
            )
        mode: WandbMode = raw_mode  # type: ignore[assignment]
        group, name, default_directory = _run_location(path)
        group = os.environ.get("WANDB_RUN_GROUP", group)
        name = os.environ.get("WANDB_NAME", name)
        run_id = os.environ.get("WANDB_RUN_ID", wandb_run_id(group, name))
        job_type = os.environ.get("WANDB_JOB_TYPE", _infer_job_type(metrics))
        tags = tuple(
            tag.strip()
            for tag in os.environ.get("WANDB_TAGS", "").split(",")
            if tag.strip()
        )
        directory = Path(os.environ.get("WANDB_DIR", default_directory)).resolve()
        return cls(
            mode=mode,
            project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY") or None,
            run_id=run_id,
            name=name,
            group=group,
            job_type=job_type,
            tags=tags,
            config=_load_config(path),
            directory=directory,
        )


class WandbSink:
    """Mirror finite scalar history to W&B without owning canonical state."""

    def __init__(self, settings: WandbSettings, *, sdk: Any | None = None) -> None:
        self.settings = settings
        self._failed = False
        self._failure: str | None = None
        self._closed = False
        self._run: Any | None = None
        if settings.mode == "disabled":
            return
        try:
            wandb = sdk if sdk is not None else importlib.import_module("wandb")
            settings.directory.mkdir(parents=True, exist_ok=True)
            init_options: dict[str, Any] = {
                "project": settings.project,
                "entity": settings.entity,
                "id": settings.run_id,
                "name": settings.name,
                "group": settings.group,
                "job_type": settings.job_type,
                "tags": list(settings.tags),
                "config": settings.config,
                "dir": str(settings.directory),
                "mode": settings.mode,
                "save_code": False,
            }
            if settings.mode == "online":
                init_options["resume"] = "allow"
            settings_factory = getattr(wandb, "Settings", None)
            if settings_factory is not None:
                init_options["settings"] = settings_factory(
                    console="off",
                    disable_code=True,
                )
            self._run = wandb.init(**init_options)
            self._define_axes()
        except Exception as error:  # pragma: no cover - exact SDK failures vary
            self._failed = True
            self._failure = str(error)
            warnings.warn(f"W&B initialization failed; continuing locally: {error}")

    @property
    def active(self) -> bool:
        return self._run is not None and not self._failed

    def _define_axes(self) -> None:
        if self._run is None:
            return
        self._run.define_metric("trainer/epoch")
        self._run.define_metric("trainer/update")
        self._run.define_metric("env/decisions")
        self._run.define_metric("env/episodes")
        self._run.define_metric("bc/*", step_metric="trainer/epoch")
        self._run.define_metric("value/*", step_metric="trainer/epoch")
        self._run.define_metric("ppo/*", step_metric="trainer/update")
        self._run.define_metric("rollout/*", step_metric="env/decisions")
        self._run.define_metric("eval/*", step_metric="env/episodes")
        self._run.define_metric("representation/*", step_metric="trainer/epoch")
        self._run.define_metric("counterfactual/*", step_metric="trainer/epoch")
        self._run.define_metric("invariance/*", step_metric="trainer/epoch")
        self._run.define_metric("system/*", step_metric="trainer/epoch")

    def status(self) -> dict[str, Any]:
        state = "failed" if self._failed else "synced" if self._closed else "active"
        if self._run is None and not self._failed:
            state = "disabled"
        url = getattr(self._run, "url", None) if self._run is not None else None
        return {
            "project": self.settings.project,
            "entity": self.settings.entity,
            "run_id": self.settings.run_id,
            "url": url,
            "state": state,
            "reason": self._failure,
            "sync_state": state,
            "failure": self._failure,
        }

    def log(self, record: dict[str, Any]) -> None:
        if not self.active:
            return
        axis = _metric_axis(self.settings.job_type, record)
        payload: dict[str, int | float] = {axis: int(record.get(axis, record["step"]))}
        for key, value in record.items():
            if key in {"step", "timestamp", axis} or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                payload[_wandb_metric_name(self.settings.job_type, key)] = value
        try:
            self._run.log(payload)
        except Exception as error:
            self._failed = True
            self._failure = str(error)
            warnings.warn(f"W&B logging failed; continuing locally: {error}")

    def set_summary(self, values: dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            for key, value in values.items():
                if isinstance(value, (str, bool, int)) or value is None:
                    self._run.summary[key] = value
                elif isinstance(value, float) and math.isfinite(value):
                    self._run.summary[key] = value
        except Exception as error:
            self._failed = True
            self._failure = str(error)
            warnings.warn(f"W&B summary update failed; continuing locally: {error}")

    def close(self, exit_code: int = 0) -> None:
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
            self._closed = True
        except Exception as error:  # pragma: no cover - exact SDK failures vary
            self._failed = True
            self._failure = str(error)
            warnings.warn(f"W&B finish failed; local metrics are complete: {error}")


def create_wandb_sink_from_environment(
    jsonl_path: str | Path,
    metrics: dict[str, Any],
) -> WandbSink | None:
    """Create a sink only after the first canonical record has been written."""
    path = Path(jsonl_path).resolve()
    if not _auto_tracking_allowed(path):
        return None
    try:
        settings = WandbSettings.from_environment(path, metrics)
    except ValueError as error:
        warnings.warn(f"W&B is disabled because configuration is invalid: {error}")
        return None
    if settings.mode == "disabled":
        return None
    return WandbSink(settings)


def _auto_tracking_allowed(jsonl_path: Path) -> bool:
    if os.environ.get("WANDB_ALLOW_NONCANONICAL", "0") == "1":
        return True
    return _canonical_version_location(jsonl_path) is not None


def _run_location(jsonl_path: Path) -> tuple[str, str, str]:
    canonical = _canonical_version_location(jsonl_path)
    if canonical is not None:
        project, version, run_root = canonical
        return project, version, str(run_root / "wandb")
    group = "local-smoke"
    name = jsonl_path.parent.name or "training"
    return group, name, str(jsonl_path.parent / "wandb")


def _canonical_version_location(jsonl_path: Path) -> tuple[str, str, Path] | None:
    path = jsonl_path.resolve()
    try:
        relative = path.relative_to(REPOSITORY_ROOT.resolve() / "rl_runs")
    except ValueError:
        return None
    if len(relative.parts) != 5:
        return None
    project, versions, version, artifact, filename = relative.parts
    if versions != "versions" or artifact != "artifact" or filename != "training_metrics.jsonl":
        return None
    if PROJECT_ID.fullmatch(project) is None or VERSIONED_ATTEMPT.fullmatch(version) is None:
        return None
    return project, version, path.parents[1]


def _load_config(jsonl_path: Path) -> dict[str, Any]:
    candidates = (
        jsonl_path.with_name("training_config.json"),
        jsonl_path.with_name("config.json"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            warnings.warn(f"W&B config mirror skipped unreadable {path}: {error}")
            return {}
        return _redact_config(value) if isinstance(value, dict) else {}
    return {}


def _redact_config(value: Any, *, key: str = "") -> Any:
    sensitive = ("api_key", "password", "secret", "token", "credential")
    if any(part in key.lower() for part in sensitive):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(item_key): _redact_config(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_config(item, key=key) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _infer_job_type(metrics: dict[str, Any]) -> str:
    keys = tuple(metrics)
    if any(key.startswith("rollout/") for key in keys):
        return "rollout_train"
    if any(key.startswith("eval/") for key in keys):
        return "eval"
    if any(key.startswith("representation/") for key in keys):
        return "representation"
    if any(key.startswith("counterfactual/") for key in keys):
        return "counterfactual"
    if any(key.startswith("invariance/") for key in keys):
        return "invariance"
    if any("ppo" in key for key in keys):
        return "ppo_train"
    if any("value_mae" in key or "value_rmse" in key for key in keys):
        return "value_calibration"
    return "bc_train"


def _metric_axis(job_type: str, record: dict[str, Any]) -> str:
    if job_type == "ppo_train":
        return "trainer/update"
    if job_type == "rollout_train":
        return "env/decisions"
    if job_type == "eval":
        return "env/episodes"
    return "trainer/epoch"


def _wandb_metric_name(job_type: str, key: str) -> str:
    if key.startswith((
        "bc/",
        "value/",
        "ppo/",
        "rollout/",
        "eval/",
        "representation/",
        "counterfactual/",
        "invariance/",
        "system/",
        "env/",
        "trainer/",
    )):
        return key
    if job_type == "bc_train":
        if key in {"train/loss", "train/bc_loss"}:
            return "bc/train/loss"
        if key == "validation/bc_loss":
            return "bc/validation/loss"
        return f"bc/{key}"
    if job_type == "value_calibration":
        if key.startswith("train/value_"):
            return f"value/train/{key.removeprefix('train/value_')}"
        if key.startswith("validation/value_"):
            return f"value/validation/{key.removeprefix('validation/value_')}"
        return f"value/{key}"
    if job_type == "ppo_train":
        if key.startswith("train/ppo_"):
            return f"ppo/{key.removeprefix('train/ppo_')}"
        return f"ppo/{key}"
    return key
