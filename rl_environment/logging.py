from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


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

    def __init__(self, jsonl_path: str | Path, tensorboard_dir: str | Path | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.jsonl_path.open("a", encoding="utf-8")
        self._writer = None
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(str(tensorboard_dir))
            except ImportError:
                self._writer = None

    def log(self, step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        record = {
            "timestamp": time.time(),
            "step": int(step),
            **{key: _json_value(value) for key, value in metrics.items()},
        }
        self._file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        self._file.flush()
        if self._writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self._writer.add_scalar(key, value, step)
            self._writer.flush()
        return record

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._file.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
