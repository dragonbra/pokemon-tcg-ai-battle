from __future__ import annotations

import json
import math
import time
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol
import warnings


class MetricSink(Protocol):
    def log(self, record: dict[str, Any]) -> None: ...

    def close(self, exit_code: int = 0) -> None: ...


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


class TrainingLogger:
    """Append canonical JSONL metrics and optionally mirror scalars to TensorBoard."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tensorboard_dir: str | Path | None = None,
        *,
        wandb_sink: MetricSink | None = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.jsonl_path.open("a", encoding="utf-8")
        self._writer = None
        self._wandb_sink = wandb_sink
        self._wandb_resolved = wandb_sink is not None
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(str(tensorboard_dir))
            except ImportError:
                self._writer = None

    def _resolve_wandb_sink(self, metrics: dict[str, Any]) -> None:
        if self._wandb_resolved:
            return
        from rl_environment.wandb_logging import create_wandb_sink_from_environment

        self._wandb_sink = create_wandb_sink_from_environment(self.jsonl_path, metrics)
        self._wandb_resolved = True

    def initialize_wandb(self, metrics: dict[str, Any] | None = None) -> None:
        """Eagerly create a formal run so system telemetry is visible immediately."""
        self._resolve_wandb_sink(metrics or {})
        self._persist_wandb_status()

    def log(
        self,
        step: int,
        metrics: dict[str, Any],
        *,
        tensorboard_metrics: dict[str, Any] | None = None,
        mirror_wandb: bool = True,
    ) -> dict[str, Any]:
        record = {
            "timestamp": time.time(),
            "step": int(step),
            **{key: _json_value(value) for key, value in metrics.items()},
        }
        self._file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        self._file.flush()
        if self._writer is not None:
            for key, value in (tensorboard_metrics or metrics).items():
                if (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                ):
                    self._writer.add_scalar(key, value, step)
            self._writer.flush()
        if mirror_wandb:
            self._resolve_wandb_sink(metrics)
        if mirror_wandb and self._wandb_sink is not None:
            try:
                self._wandb_sink.log(record)
            except Exception as error:
                warnings.warn(f"W&B mirror failed; continuing locally: {error}")
            self._persist_wandb_status()
        return record

    def _persist_wandb_status(self) -> None:
        status = getattr(self._wandb_sink, "status", None)
        if not callable(status):
            return
        try:
            value = status()
            status_path = self.jsonl_path.with_name("status.json")
            existing = (
                json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            )
            if not isinstance(existing, dict):
                existing = {}
            existing["wandb"] = _json_value(value)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=status_path.parent,
                    prefix=f".{status_path.name}.",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    json.dump(existing, temporary, ensure_ascii=True, indent=2, sort_keys=True)
                    temporary.write("\n")
                    temporary.flush()
                    os.fsync(temporary.fileno())
                temporary_path.replace(status_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        except Exception as error:
            warnings.warn(f"W&B status persistence failed; continuing locally: {error}")

    def close(self, exit_code: int = 0) -> None:
        self._resolve_wandb_sink({})
        if self._wandb_sink is not None:
            try:
                self._wandb_sink.close(exit_code=exit_code)
            except Exception as error:
                warnings.warn(f"W&B mirror close failed; local metrics are complete: {error}")
            self._persist_wandb_status()
        if self._writer is not None:
            self._writer.close()
        self._file.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.close(exit_code=1 if exc_type is not None else 0)
