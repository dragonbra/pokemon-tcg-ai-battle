"""Equal-contract Policy-0806 and official-engine throughput profiling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import replace
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from evaluation.frozen_0806_full_evaluation import (
    DEFAULT_SEED,
    POLICY_0806_TARGET,
    _batch_config,
)
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.runner.batch import _policy_inference_server, run_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / ".tmp/evaluation/evaluation_inference_profile"


class _ProcessTreeRssMonitor:
    def __init__(self, root_pid: int, interval_seconds: float = 0.05) -> None:
        self._root_pid = root_pid
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_bytes = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self._sample()
        return self.peak_bytes

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        processes: dict[int, tuple[int, int]] = {}
        for status_path in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = {}
                for line in status_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith(("Pid:", "PPid:", "VmRSS:")):
                        key, value = line.split(":", 1)
                        fields[key] = value.strip().split()[0]
                pid = int(fields["Pid"])
                processes[pid] = (int(fields["PPid"]), int(fields.get("VmRSS", 0)) * 1024)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue
        descendants = {self._root_pid}
        changed = True
        while changed:
            changed = False
            for pid, (parent, _rss) in processes.items():
                if parent in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        total = sum(processes.get(pid, (0, 0))[1] for pid in descendants)
        self.peak_bytes = max(self.peak_bytes, total)


def aggregate_worker_performance(
    records: Iterable[Mapping[str, Any]], *, wall_seconds: float, workers: int
) -> dict[str, float | int]:
    rows = [record.get("performance") for record in records]
    rows = [row for row in rows if isinstance(row, Mapping)]
    float_fields = (
        "worker_wall_seconds",
        "engine_start_seconds",
        "engine_select_seconds",
        "agent_seconds",
    )
    int_fields = ("agent_calls", "engine_select_calls")
    result: dict[str, float | int] = {"games": len(rows)}
    for field in float_fields:
        result[field] = sum(float(row.get(field, 0.0)) for row in rows)
    for field in int_fields:
        result[field] = sum(int(row.get(field, 0)) for row in rows)
    selections = int(result["engine_select_calls"])
    result["engine_seconds_per_selection"] = (
        float(result["engine_select_seconds"]) / selections if selections else 0.0
    )
    capacity = wall_seconds * workers
    result["worker_capacity_utilization"] = (
        float(result["worker_wall_seconds"]) / capacity if capacity > 0 else 0.0
    )
    return result


def _control(socket_path: Path, command: str) -> dict[str, Any]:
    connection = Client(str(socket_path), family="AF_UNIX")
    try:
        connection.send({"command": command})
        response = connection.recv()
    finally:
        connection.close()
    if not isinstance(response, dict) or not response.get("ok"):
        raise RuntimeError(f"inference server control failed: {response!r}")
    return response


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _gpu_metadata() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "nvidia_smi": completed.stdout.strip(),
        "error": completed.stderr.strip() if completed.returncode else "",
    }


def _workload(catalog: Any, games: int) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    if games < 2 or games % 2:
        raise ValueError("games must be an even integer of at least two")
    if not catalog.opponents:
        raise ValueError("profile catalog has no opponents")
    pair_count = games // 2
    opponent_count = min(pair_count, len(catalog.opponents))
    counts = [0] * opponent_count
    for pair_index in range(pair_count):
        counts[pair_index % opponent_count] += 2
    return catalog.opponents[:opponent_count], tuple(counts)


def _config(
    catalog: Any,
    *,
    output_root: Path,
    games: int,
    workers: int,
    batch_size: int,
    batch_wait_ms: float,
    inference_dtype: str,
    candidate_socket: Path | None,
    arbitrary_legal_actions: bool,
    forced_action_shortcut: bool,
    engine_pool_size: int,
) -> Any:
    opponents, counts = _workload(catalog, games)
    base = _batch_config(
        catalog,
        catalog.candidates[0],
        POLICY_0806_TARGET,
        output_root=output_root,
        workers=workers,
        batch_size=batch_size,
        batch_wait_ms=batch_wait_ms,
        inference_dtype=inference_dtype,
        candidate_socket=candidate_socket or Path("/unused"),
        opponent_socket=candidate_socket or Path("/unused"),
        opponents=opponents,
        counts=counts,
    )
    if arbitrary_legal_actions:
        return replace(
            base,
            candidate_inference_device=None,
            opponent_inference_root=None,
            opponent_inference_device=None,
            candidate_inference_socket=None,
            opponent_inference_socket=None,
            share_policy_inference_server=False,
            arbitrary_legal_actions=True,
            inference_profile=False,
            inference_forced_action_shortcut=False,
            engine_pool_size=engine_pool_size,
        )
    return replace(
        base,
        inference_profile=True,
        inference_forced_action_shortcut=forced_action_shortcut,
        engine_pool_size=engine_pool_size,
    )


def run_profile(
    *,
    mode: str,
    games: int,
    workers: int,
    batch_size: int,
    batch_wait_ms: float,
    inference_dtype: str,
    output: Path,
    engine_pool_size: int = 1,
) -> dict[str, Any]:
    if mode not in {"policy", "forced", "engine"}:
        raise ValueError("mode must be policy, forced, or engine")
    catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
    forced = mode == "forced"
    arbitrary = mode == "engine"
    with ExitStack() as stack:
        socket_path = None
        if not arbitrary:
            socket_path = stack.enter_context(
                _policy_inference_server(
                    root=catalog.candidate_policy.root,
                    device="cuda:0",
                    batch_size=batch_size,
                    batch_wait_ms=batch_wait_ms,
                    label="profile",
                    ability_repeat_limit=8,
                    inference_dtype=inference_dtype,
                    profile=True,
                    forced_action_shortcut=forced,
                )
            )
            assert socket_path is not None
            warmup = _config(
                catalog,
                output_root=DEFAULT_OUTPUT_ROOT / "warmup",
                games=2,
                workers=min(2, workers),
                batch_size=batch_size,
                batch_wait_ms=batch_wait_ms,
                inference_dtype=inference_dtype,
                candidate_socket=socket_path,
                arbitrary_legal_actions=False,
                forced_action_shortcut=forced,
                engine_pool_size=engine_pool_size,
            )
            warmup_result = run_batch(warmup)
            if warmup_result.report_data.summary["errors"] or warmup_result.report_data.summary["unfinished"]:
                raise RuntimeError("Policy-0806 warm-up did not complete cleanly")
            _control(socket_path, "reset_profile")

        config = _config(
            catalog,
            output_root=DEFAULT_OUTPUT_ROOT / mode,
            games=games,
            workers=workers,
            batch_size=batch_size,
            batch_wait_ms=batch_wait_ms,
            inference_dtype=inference_dtype,
            candidate_socket=socket_path,
            arbitrary_legal_actions=arbitrary,
            forced_action_shortcut=forced,
            engine_pool_size=engine_pool_size,
        )
        rss_monitor = _ProcessTreeRssMonitor(os.getpid())
        rss_monitor.start()
        started = time.perf_counter()
        try:
            result = run_batch(config)
            wall_seconds = time.perf_counter() - started
        finally:
            peak_process_tree_rss_bytes = rss_monitor.stop()
        summary = result.report_data.summary
        if summary["errors"] or summary["unfinished"] or summary["completed_games"] != games:
            raise RuntimeError("profile workload did not complete cleanly")
        inference = (
            _control(socket_path, "profile")["profile"]
            if socket_path is not None
            else None
        )

    payload = {
        "schema_version": "evaluation_inference_profile_v1",
        "mode": mode,
        "workload": {
            "pool_id": catalog.pool.pool_id,
            "schedule_sha256": catalog.pool.manifest["schedule_sha256"],
            "policy": "0806",
            "seed": DEFAULT_SEED,
            "games": games,
            "workers": workers,
            "worker_processes": min(workers, games),
            "engine_pool_size": engine_pool_size,
            "max_live_environments": min(games, workers * engine_pool_size),
            "opponent_game_counts": list(config.games_by_opponent or ()),
            "batch_size": batch_size,
            "batch_wait_ms": batch_wait_ms,
            "inference_dtype": inference_dtype,
            "forced_action_shortcut": forced,
            "arbitrary_legal_actions": arbitrary,
        },
        "outcome": {
            "completed_games": summary["completed_games"],
            "errors": summary["errors"],
            "unfinished": summary["unfinished"],
            "wall_seconds": wall_seconds,
            "games_per_second": games / wall_seconds,
            "engine_selections": summary["performance"]["engine_selections"],
            "engine_selections_per_second": summary["performance"]["engine_selections"] / wall_seconds,
            "report": str(result.report_path.relative_to(ROOT)),
        },
        "worker": aggregate_worker_performance(
            result.game_records,
            wall_seconds=wall_seconds,
            workers=min(games, workers * engine_pool_size),
        ),
        "inference": inference,
        "hardware": {
            **_gpu_metadata(),
            "peak_process_tree_rss_bytes": peak_process_tree_rss_bytes,
        },
    }
    _atomic_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("policy", "forced", "engine"), required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--inference-dtype", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--engine-pool-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or DEFAULT_OUTPUT_ROOT / f"{args.mode}.json"
    payload = run_profile(
        mode=args.mode,
        games=args.games,
        workers=args.workers,
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
        inference_dtype=args.inference_dtype,
        output=output,
        engine_pool_size=args.engine_pool_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
