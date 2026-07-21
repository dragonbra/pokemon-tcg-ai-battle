from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from evaluation.runner.models import GameResult


class TraceStore:
    """管理单次 batch 的完整临时 trace 与长期精简 games.jsonl。"""

    def __init__(self, temp_root: Path, report_root: Path, retain_limit: int = 3):
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
        if retain_limit < 0:
            raise ValueError("retain_limit must not be negative")
        self.retain_limit = retain_limit
        self.temp_root = caller_temp_root / f".trace-store-{uuid.uuid4().hex}"
        self._ownership_marker = self.temp_root / ".trace-store-owned"
        self._ownership_token = uuid.uuid4().hex
        self._records: list[dict[str, object]] = []
        self._results: dict[str, GameResult] = {}
        self._retained_paths: dict[str, Path] = {}
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
            "trace_path": self._serialized_retained_path(result.game_id),
            "metric_refs": metric_refs,
        }
        self._records.append(record)
        self._results[result.game_id] = result
        self._write_records()

    def retain(self, selected_game_ids: set[str]) -> dict[str, Path]:
        if len(selected_game_ids) > self.retain_limit:
            raise ValueError(f"cannot retain more than {self.retain_limit} traces")
        for game_id in selected_game_ids:
            self._validate_game_id(game_id)
            if game_id not in self._results:
                raise KeyError(f"unknown game ID: {game_id}")

        sources: dict[str, Path] = {}
        for game_id in sorted(selected_game_ids):
            source = self._results[game_id].trace_path.resolve()
            self._ensure_temp_trace(source)
            if not source.is_file():
                raise FileNotFoundError(f"selected trace is missing for {game_id}: {source}")
            sources[game_id] = source

        traces_root = self.report_root / "traces"
        traces_root.mkdir(parents=True, exist_ok=True)
        existing_paths = set(traces_root.iterdir())
        owned_paths = set(self._retained_paths.values())
        unowned_paths = existing_paths - owned_paths
        if unowned_paths:
            raise ValueError("refusing to replace unowned report traces")
        for existing_trace in owned_paths:
            existing_trace.unlink()
        self._retained_paths.clear()
        for game_id in sorted(selected_game_ids):
            source = sources[game_id]
            destination = traces_root / f"{game_id}.json"
            shutil.copy2(source, destination)
            self._retained_paths[game_id] = destination

        for record in self._records:
            game_id = str(record["game_id"])
            record["trace_path"] = self._serialized_retained_path(game_id)
        self._write_records()
        return dict(self._retained_paths)

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

    def _write_records(self) -> None:
        games_path = self.report_root / "games.jsonl"
        with games_path.open("w", encoding="utf-8") as games_file:
            for record in self._records:
                games_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                games_file.write("\n")

    def _serialized_retained_path(self, game_id: str) -> str | None:
        path = self._retained_paths.get(game_id)
        return str(path) if path is not None else None

    def _ensure_temp_trace(self, path: Path) -> None:
        try:
            path.relative_to(self.temp_root)
        except ValueError as exc:
            raise ValueError("trace path must be inside temp_root") from exc

    @staticmethod
    def _validate_game_id(game_id: str) -> None:
        if not game_id or Path(game_id).name != game_id or game_id in {".", ".."}:
            raise ValueError("game_id must be a file name")
