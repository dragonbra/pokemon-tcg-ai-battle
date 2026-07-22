from __future__ import annotations

from pathlib import Path

from .batch import _read_or_create_trace, _run_worker
from .models import GameRequest, GameResult


def run_game(
    request: GameRequest,
    *,
    temp_root: Path,
    timeout_seconds: float = 30.0,
) -> tuple[GameResult, dict[str, object]]:
    """运行一场隔离对局并返回轻量 trace，不生成 batch 报告。"""
    temp_root.mkdir(parents=True, exist_ok=True)
    trace_path = temp_root / f"{request.game_id}.json"
    result = _run_worker(request, trace_path, temp_root, timeout_seconds)
    return _read_or_create_trace(request, result, trace_path)
