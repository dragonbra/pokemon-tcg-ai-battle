from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from evaluation.runner.models import GameResult


class TraceStore:
    """管理单次 batch 的临时完整 trace 和内存中的精简对局记录。"""

    def __init__(self, temp_root: Path, report_root: Path):
        caller_temp_root = temp_root.resolve()
        self.report_root = report_root.resolve()
        if caller_temp_root == self.report_root:
            raise ValueError("temp_root and report_root must be different")
        try:
            self.report_root.relative_to(caller_temp_root)
        except ValueError:
            pass
        else:
            raise ValueError("report_root must not be inside temp_root")
        self.temp_root = caller_temp_root / f".trace-store-{uuid.uuid4().hex}"
        self._ownership_marker = self.temp_root / ".trace-store-owned"
        self._ownership_token = uuid.uuid4().hex
        self._records: list[dict[str, object]] = []
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._ownership_marker.write_text(self._ownership_token, encoding="ascii")
        self.report_root.mkdir(parents=True, exist_ok=True)

    @property
    def game_records(self) -> tuple[dict[str, object], ...]:
        return tuple(self._records)

    def temp_path(self, game_id: str) -> Path:
        self._validate_game_id(game_id)
        path = self.temp_root / f"{game_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_game_record(self, result: GameResult, trace: dict) -> None:
        self._validate_game_id(result.game_id)
        if not isinstance(trace, dict):
            raise ValueError("trace must be a mapping")
        metric_refs = trace.get("metric_refs", {})
        if not isinstance(metric_refs, dict):
            raise ValueError("metric_refs must be a mapping")
        trace_path = result.trace_path.resolve()
        self._ensure_temp_trace(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        record: dict[str, object] = {
            "game_id": result.game_id,
            "opponent": result.opponent,
            "swap": not result.candidate_first,
            "candidate_first": result.candidate_first,
            "status": result.status,
            "winner": result.winner,
            "steps": result.steps,
            "error_kind": result.error_kind,
            "trace_path": None,
            "metric_refs": metric_refs,
        }
        self._records.append(record)

    def cleanup(self) -> None:
        """只删除本 store 创建的临时根目录，重复调用安全。"""
        if not self.temp_root.exists():
            return
        if not self._ownership_marker.is_file():
            raise RuntimeError(f"refusing to remove unowned temp root: {self.temp_root}")
        token = self._ownership_marker.read_text(encoding="ascii")
        if token != self._ownership_token:
            raise RuntimeError(f"refusing to remove unowned temp root: {self.temp_root}")
        shutil.rmtree(self.temp_root)

    def _ensure_temp_trace(self, path: Path) -> None:
        try:
            path.relative_to(self.temp_root)
        except ValueError as exc:
            raise ValueError("trace path must be inside temp_root") from exc

    @staticmethod
    def _validate_game_id(game_id: str) -> None:
        if not game_id or Path(game_id).name != game_id or game_id in {".", ".."}:
            raise ValueError("game_id must be a file name")
