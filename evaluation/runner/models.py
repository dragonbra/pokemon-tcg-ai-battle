from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.packages.loader import SubmissionPackage


@dataclass(frozen=True)
class GameRequest:
    run_id: str
    game_id: str
    candidate: SubmissionPackage
    opponent: SubmissionPackage
    candidate_first: bool
    max_steps: int
    visualize: bool
    engine_turn_draw_limit: int = 0
    seed: int = 0
    arbitrary_legal_actions: bool = False


@dataclass(frozen=True)
class GameResult:
    game_id: str
    opponent: str
    candidate_first: bool
    candidate_physical_index: int
    finished: bool
    winner: int | None
    status: str
    error_kind: str | None
    error: str | None
    steps: int
    trace_path: Path
    performance: dict[str, Any] | None = None
