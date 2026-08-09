from __future__ import annotations

import hashlib
import ctypes
import importlib
import importlib.util
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Iterator

from evaluation.packages.loader import PackageValidationError, SubmissionPackage, _clear_cg_modules
from evaluation.runner.models import GameRequest, GameResult
from evaluation.runner.progress_guard import progress_guard_forfeit_reason
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


def _remote_policy_agent(
    socket_path: str,
    *,
    deck: list[int],
    role: str,
):
    connection = Client(socket_path, family="AF_UNIX")
    exact_deck = tuple(int(card_id) for card_id in deck)

    def call(observation: dict[str, Any]) -> Any:
        connection.send({"observation": observation, "deck": exact_deck})
        response = connection.recv()
        if not isinstance(response, dict) or not response.get("ok"):
            detail = response.get("error") if isinstance(response, dict) else response
            raise RuntimeError(f"shared {role} inference failed: {detail}")
        return response.get("action")

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


def _package_requires_pytorch(package: SubmissionPackage) -> bool:
    manifest = package.package_manifest or {}
    framework = manifest.get("runtime_framework")
    if framework is not None:
        if framework != "pytorch":
            return False
        load_order = manifest.get("native_runtime_load_order")
        if load_order != "torch_before_cg":
            raise PackageValidationError(
                f"{package.name} declares PyTorch but not "
                "native_runtime_load_order=torch_before_cg"
            )
        return True
    # Historical model packages predate the explicit runtime manifest contract.
    return (package.root / "strategy/model.bin").is_file()


def _preload_local_agent_runtime_dependencies(request: GameRequest) -> None:
    """Load PyTorch before cg/libcg for every in-process PyTorch policy."""
    if request.arbitrary_legal_actions:
        return
    local_packages = []
    if not os.environ.get("EVALUATION_CANDIDATE_INFERENCE_SOCKET"):
        local_packages.append(request.candidate)
    if not os.environ.get("EVALUATION_OPPONENT_INFERENCE_SOCKET"):
        local_packages.append(request.opponent)
    if any(_package_requires_pytorch(package) for package in local_packages):
        importlib.import_module("torch")


def _load_agents(request: GameRequest) -> tuple[Any, Any]:
    _preload_local_agent_runtime_dependencies(request)
    _clear_cg_modules()
    game_module = _load_game_api_with_runtime(request.candidate.root)
    if request.arbitrary_legal_actions:
        return (
            game_module,
            _arbitrary_legal_agent(request.candidate.deck),
            _arbitrary_legal_agent(request.opponent.deck),
        )
    candidate_exclusions = (request.opponent.root,)
    opponent_exclusions = (request.candidate.root,)
    candidate_inference_socket = os.environ.get(
        "EVALUATION_CANDIDATE_INFERENCE_SOCKET"
    )
    opponent_inference_socket = os.environ.get(
        "EVALUATION_OPPONENT_INFERENCE_SOCKET"
    )
    candidate_agent = None
    if candidate_inference_socket:
        candidate_agent = _remote_policy_agent(
            candidate_inference_socket,
            deck=request.candidate.deck,
            role="candidate",
        )
    else:
        candidate_module = _load_agent_module(
            request.candidate,
            "candidate",
            candidate_exclusions,
        )
        candidate_agent = _isolated_agent(
            request.candidate,
            candidate_module.agent,
            candidate_exclusions,
        )
    if opponent_inference_socket:
        opponent_agent = _remote_policy_agent(
            opponent_inference_socket,
            deck=request.opponent.deck,
            role="opponent",
        )
    else:
        opponent_module = _load_agent_module(
            request.opponent,
            "opponent",
            opponent_exclusions,
        )
        opponent_agent = _isolated_agent(
            request.opponent,
            opponent_module.agent,
            opponent_exclusions,
        )
    return (
        game_module,
        candidate_agent,
        opponent_agent,
    )


def _arbitrary_legal_agent(deck: list[int]):
    exact_deck = tuple(int(card_id) for card_id in deck)

    def call(observation: dict[str, Any]) -> list[int]:
        select = observation.get("select")
        if select is None:
            return list(exact_deck)
        if not isinstance(select, dict) or not isinstance(select.get("option"), list):
            raise ValueError("engine observation has no legal option list")
        minimum = select.get("minCount", 0)
        maximum = select.get("maxCount", len(select["option"]))
        if (
            type(minimum) is not int
            or type(maximum) is not int
            or not 0 <= minimum <= maximum <= len(select["option"])
        ):
            raise ValueError("engine observation has invalid selection bounds")
        if select.get("type") == 0 and minimum <= 1 <= maximum:
            end = next(
                (
                    index
                    for index, option in enumerate(select["option"])
                    if isinstance(option, dict) and option.get("type") == 14
                ),
                None,
            )
            if end is not None:
                return [end]
        return list(range(minimum))

    return call


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
    worker_started_ns = time.perf_counter_ns()
    engine_start_ns = 0
    engine_select_ns = 0
    agent_ns = 0
    agent_calls = 0
    engine_select_calls = 0
    random.seed(request.policy_seed or request.seed)
    candidate_physical_index = 0
    physical_packages = (request.candidate, request.opponent)

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
    previous_engine_library = os.environ.get("PTCG_CG_LIBRARY")

    try:
        if request.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        phase = "compatibility"
        assert_cg_compatible(request.candidate, request.opponent)
        if request.engine_library is not None:
            os.environ["PTCG_CG_LIBRARY"] = str(request.engine_library.resolve())
        phase = "load"
        game_module, candidate_agent, opponent_agent = _load_agents(request)
        if request.engine_library is not None:
            engine_library = getattr(game_module, "lib", None)
            configure = getattr(engine_library, "ConfigureSeeds", None)
            if configure is None:
                raise RuntimeError("seeded engine runtime has no ConfigureSeeds symbol")
            configure.restype = ctypes.c_int
            configure.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong]
            configured = int(configure(request.seed, request.search_seed))
            if configured != 0:
                raise RuntimeError(f"seeded engine configuration failed: {configured}")
        phase = "start"
        start_attempted = True
        engine_started_ns = time.perf_counter_ns()
        try:
            observation, _start_data = game_module.battle_start(
                physical_packages[0].deck,
                physical_packages[1].deck,
            )
        finally:
            engine_start_ns += time.perf_counter_ns() - engine_started_ns

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

                engine_turn = current.get("turn")
                if (
                    request.engine_turn_draw_limit > 0
                    and type(engine_turn) is int
                    and engine_turn >= request.engine_turn_draw_limit
                ):
                    trace.append({"step": step, "state": summary, "observation": observation})
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=None,
                        status="finished",
                        error_kind=None,
                        error=(
                            "Arena turn-limit draw at engine turn "
                            f"{request.engine_turn_draw_limit}"
                        ),
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break

                current_player = int(current.get("yourIndex", 0))
                select = observation.get("select") or {}
                forced_first_player = select.get("context") == 41
                selected_agent = candidate_agent if current_player == candidate_physical_index else opponent_agent
                try:
                    if forced_first_player:
                        action = [0] if request.candidate_first else [1]
                    else:
                        agent_started_ns = time.perf_counter_ns()
                        try:
                            action = selected_agent(observation)
                        finally:
                            agent_ns += time.perf_counter_ns() - agent_started_ns
                            agent_calls += 1
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

                forfeit_reason = progress_guard_forfeit_reason(action)
                if forfeit_reason is not None:
                    trace.append(
                        {
                            "step": step,
                            "state": summary,
                            "observation": observation,
                            "adjudication": {
                                "kind": "progress_guard_forfeit",
                                "reason": forfeit_reason,
                                "losing_physical_player": current_player,
                            },
                        }
                    )
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=(
                            1
                            if current_player == candidate_physical_index
                            else 0
                        ),
                        status="finished",
                        error_kind=None,
                        error=(
                            "Progress-guard forfeit: "
                            f"{forfeit_reason}"
                        ),
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break

                selection_count += 1
                trace_entry = {
                    "step": step,
                    "state": summary,
                    "observation": observation,
                    "action": action,
                }
                if forced_first_player:
                    trace_entry["forced_by_harness"] = "first_player"
                trace.append(trace_entry)
                try:
                    engine_selected_ns = time.perf_counter_ns()
                    try:
                        observation = game_module.battle_select(action)
                    finally:
                        engine_select_ns += time.perf_counter_ns() - engine_selected_ns
                        engine_select_calls += 1
                    if forced_first_player:
                        actual_first = (observation.get("current") or {}).get("firstPlayer")
                        expected_first = 0 if request.candidate_first else 1
                        if actual_first != expected_first:
                            raise RuntimeError(
                                "official engine did not honor forced first player: "
                                f"expected {expected_first}, got {actual_first}"
                            )
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
            result = replace(
                result,
                performance={
                    "worker_wall_seconds": (time.perf_counter_ns() - worker_started_ns) / 1e9,
                    "engine_start_seconds": engine_start_ns / 1e9,
                    "engine_select_seconds": engine_select_ns / 1e9,
                    "agent_seconds": agent_ns / 1e9,
                    "agent_calls": agent_calls,
                    "engine_select_calls": engine_select_calls,
                },
            )
            payload["result"] = _serialized_result(result)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            sys.path[:] = parent_sys_path
            _clear_cg_modules()
            sys.modules.update(parent_cg_modules)
            os.chdir(parent_cwd)
            if previous_engine_library is None:
                os.environ.pop("PTCG_CG_LIBRARY", None)
            else:
                os.environ["PTCG_CG_LIBRARY"] = previous_engine_library
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
        display_name=(
            str(payload["display_name"])
            if payload.get("display_name") is not None
            else None
        ),
        representative_cards=tuple(payload.get("representative_cards") or ()),
        package_manifest=(
            dict(payload["package_manifest"])
            if payload.get("package_manifest") is not None
            else None
        ),
    )


def _request_from_payload(payload: dict[str, Any]) -> tuple[GameRequest, Path]:
    request = GameRequest(
        run_id=str(payload["run_id"]),
        game_id=str(payload["game_id"]),
        candidate=_package_from_payload(payload["candidate"]),
        opponent=_package_from_payload(payload["opponent"]),
        candidate_first=bool(payload["candidate_first"]),
        max_steps=int(payload["max_steps"]),
        engine_turn_draw_limit=int(payload.get("engine_turn_draw_limit", 0)),
        visualize=bool(payload["visualize"]),
        seed=int(payload.get("seed", 0)),
        policy_seed=int(payload.get("policy_seed", payload.get("seed", 0))),
        search_seed=int(payload.get("search_seed", payload.get("seed", 0))),
        engine_library=(
            Path(payload["engine_library"])
            if payload.get("engine_library") is not None
            else None
        ),
        arbitrary_legal_actions=bool(payload.get("arbitrary_legal_actions", False)),
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
