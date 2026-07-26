from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from evaluation.packages.loader import PackageValidationError, SubmissionPackage, _clear_cg_modules
from evaluation.runner.models import GameRequest, GameResult
from evaluation.runtime.loader import assert_cg_compatible


def _state_summary(observation: dict[str, Any] | None) -> dict[str, Any]:
    current = (observation or {}).get("current") or {}
    select = (observation or {}).get("select") or {}
    return {
        "turn": current.get("turn"),
        "yourIndex": current.get("yourIndex"),
        "result": current.get("result"),
        "selectType": select.get("type"),
        "optionCount": len(select.get("option") or []),
    }


@contextmanager
def _temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _path_matches_any_root(entry: str, roots: tuple[Path, ...]) -> bool:
    try:
        resolved_entry = Path(entry).resolve()
    except OSError:
        return False
    return any(resolved_entry == root.resolve() for root in roots)


def _local_module_names(package_root: Path) -> set[str]:
    names = {
        path.stem
        for path in package_root.glob("*.py")
        if path.stem.isidentifier() and path.stem != "cg"
    }
    names.update(
        path.name
        for path in package_root.iterdir()
        if path.is_dir()
        and path.name.isidentifier()
        and path.name != "cg"
    )
    return names


def _clear_modules_with_top_level_names(names: set[str]) -> None:
    for module_name in tuple(sys.modules):
        if module_name.partition(".")[0] in names:
            sys.modules.pop(module_name, None)


@contextmanager
def _temporary_agent_import_context(
    package: SubmissionPackage,
    excluded_roots: tuple[Path, ...],
) -> Iterator[None]:
    """让裸本地导入只解析到当前 agent，同时不影响对局所需的 cg。"""
    package_root = package.root.resolve()
    local_names = _local_module_names(package_root)
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.partition(".")[0] in local_names
    }
    previous_path = list(sys.path)

    _clear_modules_with_top_level_names(local_names)
    sys.path[:] = [
        str(package_root),
        *[
            entry
            for entry in previous_path
            if not _path_matches_any_root(entry, (package_root, *excluded_roots))
        ],
    ]
    try:
        with _temporary_cwd(package_root):
            yield
    finally:
        _clear_modules_with_top_level_names(local_names)
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path


def _load_agent_module(
    package: SubmissionPackage,
    module_suffix: str,
    excluded_roots: tuple[Path, ...],
):
    module_name = "_evaluation_worker_" + hashlib.sha256(
        f"{package.entrypoint.resolve()}::{module_suffix}".encode("utf-8")
    ).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, package.entrypoint)
    if spec is None or spec.loader is None:
        raise PackageValidationError(f"{package.name} main.py could not be loaded")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with _temporary_agent_import_context(package, excluded_roots):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _isolated_agent(
    package: SubmissionPackage,
    agent: Any,
    excluded_roots: tuple[Path, ...],
):
    def call(observation: dict[str, Any]) -> Any:
        with _temporary_agent_import_context(package, excluded_roots):
            return agent(observation)

    return call


def _load_game_api_with_runtime(runtime_root: Path) -> object:
    """加载 cg.game，并让 cg 包在整个 worker 对局期间保持可导入。"""
    cg_root = (runtime_root / "cg").resolve()
    package_path = cg_root / "__init__.py"
    package_spec = importlib.util.spec_from_file_location(
        "cg",
        package_path,
        submodule_search_locations=[str(cg_root)],
    )
    if package_spec is None or package_spec.loader is None:
        raise PackageValidationError("cg package could not be loaded")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["cg"] = package
    package_spec.loader.exec_module(package)

    game_path = cg_root / "game.py"
    game_spec = importlib.util.spec_from_file_location("cg.game", game_path)
    if game_spec is None or game_spec.loader is None:
        raise PackageValidationError("cg/game.py could not be loaded")
    game_module = importlib.util.module_from_spec(game_spec)
    sys.modules["cg.game"] = game_module
    game_spec.loader.exec_module(game_module)
    return game_module


def _load_agents(request: GameRequest) -> tuple[Any, Any]:
    _clear_cg_modules()
    game_module = _load_game_api_with_runtime(request.candidate.root)
    candidate_exclusions = (request.opponent.root,)
    opponent_exclusions = (request.candidate.root,)
    candidate_module = _load_agent_module(
        request.candidate,
        "candidate",
        candidate_exclusions,
    )
    opponent_module = _load_agent_module(
        request.opponent,
        "opponent",
        opponent_exclusions,
    )
    return (
        game_module,
        _isolated_agent(request.candidate, candidate_module.agent, candidate_exclusions),
        _isolated_agent(request.opponent, opponent_module.agent, opponent_exclusions),
    )


def _normalize_winner(physical_winner: int | None, candidate_physical_index: int) -> int | None:
    if physical_winner is None or physical_winner < 0:
        return None
    return 0 if physical_winner == candidate_physical_index else 1


def _error_result(
    request: GameRequest,
    trace_path: Path,
    *,
    candidate_physical_index: int,
    steps: int,
    status: str,
    error_kind: str,
    error: str,
) -> GameResult:
    return GameResult(
        game_id=request.game_id,
        opponent=request.opponent.name,
        candidate_first=request.candidate_first,
        candidate_physical_index=candidate_physical_index,
        finished=False,
        winner=None,
        status=status,
        error_kind=error_kind,
        error=error,
        steps=steps,
        trace_path=trace_path,
    )


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _error_kind_for_phase(phase: str) -> str:
    return {
        "validation": "worker_error",
        "compatibility": "cg_mismatch",
        "load": "load_error",
        "start": "start_error",
        "game": "game_error",
    }.get(phase, "worker_error")


def _serialized_result(result: GameResult) -> dict[str, Any]:
    serialized = asdict(result)
    serialized["trace_path"] = str(result.trace_path)
    return serialized


def run_game(request: GameRequest, trace_path: Path) -> GameResult:
    candidate_physical_index = 0 if request.candidate_first else 1
    physical_packages = (
        (request.candidate, request.opponent)
        if request.candidate_first
        else (request.opponent, request.candidate)
    )

    trace: list[dict[str, Any]] = []
    selection_count = 0
    payload: dict[str, Any] = {
        "run_id": request.run_id,
        "game_id": request.game_id,
        "candidate": request.candidate.name,
        "opponent": request.opponent.name,
        "candidate_first": request.candidate_first,
        "trace": trace,
    }
    observation: dict[str, Any] | None = None
    visualization_error: str | None = None
    game_module = None
    start_attempted = False
    result: GameResult | None = None
    phase = "validation"
    parent_cwd = Path.cwd()
    parent_sys_path = list(sys.path)
    parent_cg_modules = {
        name: module for name, module in sys.modules.items() if name == "cg" or name.startswith("cg.")
    }

    try:
        if request.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        phase = "compatibility"
        assert_cg_compatible(request.candidate, request.opponent)
        phase = "load"
        game_module, candidate_agent, opponent_agent = _load_agents(request)
        phase = "start"
        start_attempted = True
        observation, _start_data = game_module.battle_start(
            physical_packages[0].deck,
            physical_packages[1].deck,
        )

        if observation is None:
            result = _error_result(
                request,
                trace_path,
                candidate_physical_index=candidate_physical_index,
                steps=0,
                status="start_error",
                error_kind="start_error",
                error="battle_start returned no observation",
            )
        else:
            phase = "game"
            for step in range(request.max_steps):
                summary = _state_summary(observation)
                current = observation.get("current") or {}
                physical_winner = current.get("result")
                if isinstance(physical_winner, int) and physical_winner >= 0:
                    trace.append({"step": step, "state": summary, "observation": observation})
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=_normalize_winner(physical_winner, candidate_physical_index),
                        status="finished",
                        error_kind=None,
                        error=None,
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break

                current_player = int(current.get("yourIndex", 0))
                selected_agent = candidate_agent if current_player == candidate_physical_index else opponent_agent
                try:
                    action = selected_agent(observation)
                except BaseException as exc:
                    error_side = (
                        "candidate_error"
                        if current_player == candidate_physical_index
                        else "opponent_error"
                    )
                    trace.append(
                        {
                            "step": step,
                            "state": summary,
                            "observation": observation,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    result = _error_result(
                        request,
                        trace_path,
                        candidate_physical_index=candidate_physical_index,
                        steps=selection_count,
                        status=error_side,
                        error_kind=error_side,
                        error=_exception_text(exc),
                    )
                    break

                selection_count += 1
                trace.append(
                    {
                        "step": step,
                        "state": summary,
                        "observation": observation,
                        "action": action,
                    }
                )
                try:
                    observation = game_module.battle_select(action)
                except IndexError as exc:
                    error_side = (
                        "candidate_error"
                        if current_player == candidate_physical_index
                        else "opponent_error"
                    )
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=1 if error_side == "candidate_error" else 0,
                        status="finished",
                        error_kind=error_side,
                        error=_exception_text(exc),
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break
                physical_winner = (observation.get("current") or {}).get("result")
                if isinstance(physical_winner, int) and physical_winner >= 0:
                    trace.append(
                        {
                            "step": step + 1,
                            "state": _state_summary(observation),
                            "observation": observation,
                        }
                    )
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=_normalize_winner(physical_winner, candidate_physical_index),
                        status="finished",
                        error_kind=None,
                        error=None,
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break
            else:
                result = _error_result(
                    request,
                    trace_path,
                    candidate_physical_index=candidate_physical_index,
                    steps=selection_count,
                    status="unfinished",
                    error_kind="step_limit",
                    error=f"step limit reached ({request.max_steps})",
                )
    except BaseException as exc:
        result = _error_result(
            request,
            trace_path,
            candidate_physical_index=candidate_physical_index,
            steps=selection_count,
            status=_error_kind_for_phase(phase),
            error_kind=_error_kind_for_phase(phase),
            error=str(exc) if phase == "validation" else _exception_text(exc),
        )
    finally:
        try:
            if start_attempted and game_module is not None and request.visualize:
                try:
                    payload["visualize"] = json.loads(game_module.visualize_data())
                except BaseException as exc:
                    visualization_error = _exception_text(exc)
            if start_attempted and game_module is not None:
                try:
                    game_module.battle_finish()
                except BaseException as exc:
                    if result is None or result.finished:
                        result = _error_result(
                            request,
                            trace_path,
                            candidate_physical_index=candidate_physical_index,
                            steps=selection_count,
                            status="finish_error",
                            error_kind="finish_error",
                            error=_exception_text(exc),
                        )
            if visualization_error is not None:
                payload["visualization_error"] = visualization_error
            if result is None:
                result = _error_result(
                    request,
                    trace_path,
                    candidate_physical_index=candidate_physical_index,
                    steps=selection_count,
                    status="worker_error",
                    error_kind="worker_error",
                    error="worker did not produce a result",
                )
            payload["result"] = _serialized_result(result)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            sys.path[:] = parent_sys_path
            _clear_cg_modules()
            sys.modules.update(parent_cg_modules)
            os.chdir(parent_cwd)
    return result


def _package_from_payload(payload: dict[str, Any]) -> SubmissionPackage:
    return SubmissionPackage(
        name=str(payload["name"]),
        root=Path(payload["root"]),
        deck=[int(card_id) for card_id in payload["deck"]],
        entrypoint=Path(payload["entrypoint"]),
        package_hash=str(payload["package_hash"]),
        deck_hash=str(payload["deck_hash"]),
        cg_manifest=dict(payload["cg_manifest"]),
    )


def _request_from_payload(payload: dict[str, Any]) -> tuple[GameRequest, Path]:
    request = GameRequest(
        run_id=str(payload["run_id"]),
        game_id=str(payload["game_id"]),
        candidate=_package_from_payload(payload["candidate"]),
        opponent=_package_from_payload(payload["opponent"]),
        candidate_first=bool(payload["candidate_first"]),
        max_steps=int(payload["max_steps"]),
        visualize=bool(payload["visualize"]),
    )
    return request, Path(payload["trace_path"])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("usage: python -m evaluation.runner.worker <request.json> <result.json>")

    request_path = Path(args[0])
    result_path = Path(args[1])
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    request, trace_path = _request_from_payload(request_payload)
    result = run_game(request, trace_path)
    result_payload = asdict(result)
    result_payload["trace_path"] = str(result.trace_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
