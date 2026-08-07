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
        self.cpu_ticks = 0
        self._last_cpu_ticks: dict[int, int] = {}
        self._clock_ticks_per_second = int(os.sysconf("SC_CLK_TCK"))

    def start(self) -> None:
        self._sample(baseline=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self._sample()
        return self.peak_bytes

    @property
    def cpu_seconds(self) -> float:
        return self.cpu_ticks / self._clock_ticks_per_second

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def _sample(self, *, baseline: bool = False) -> None:
        processes: dict[int, tuple[int, int, int]] = {}
        for status_path in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = {}
                for line in status_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith(("Pid:", "PPid:", "VmRSS:")):
                        key, value = line.split(":", 1)
                        fields[key] = value.strip().split()[0]
                pid = int(fields["Pid"])
                stat = status_path.with_name("stat").read_text(encoding="utf-8")
                stat_fields = stat.rsplit(")", 1)[1].split()
                cpu_ticks = int(stat_fields[11]) + int(stat_fields[12])
                processes[pid] = (
                    int(fields["PPid"]),
                    int(fields.get("VmRSS", 0)) * 1024,
                    cpu_ticks,
                )
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue
        descendants = {self._root_pid}
        changed = True
        while changed:
            changed = False
            for pid, (parent, _rss, _cpu_ticks) in processes.items():
                if parent in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        total = sum(processes.get(pid, (0, 0, 0))[1] for pid in descendants)
        self.peak_bytes = max(self.peak_bytes, total)
        for pid in descendants:
            current = processes.get(pid, (0, 0, 0))[2]
            previous = self._last_cpu_ticks.get(pid)
            if previous is None:
                previous = current if baseline or pid == self._root_pid else 0
            self.cpu_ticks += max(0, current - previous)
            self._last_cpu_ticks[pid] = current


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
        "compiler_seconds",
        "ipc_connection_wait_seconds",
        "ipc_send_seconds",
        "ipc_response_wait_seconds",
        "ipc_roundtrip_seconds",
    )
    int_fields = ("agent_calls", "engine_select_calls", "compiler_calls", "ipc_calls")
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


def derive_pipeline_diagnostics(
    inference: Mapping[str, Any],
    worker: Mapping[str, Any],
    *,
    wall_seconds: float,
    max_live_environments: int,
) -> dict[str, Any]:
    batches = int(inference.get("batches", 0))
    calls = int(worker.get("ipc_calls", 0))
    seconds = inference.get("seconds", {})
    seconds = seconds if isinstance(seconds, Mapping) else {}
    histogram_payload = inference.get("batch_histogram", {})
    histogram = {
        int(size): int(count)
        for size, count in histogram_payload.items()
    } if isinstance(histogram_payload, Mapping) else {}

    def per_batch_ms(name: str) -> float:
        return 1000.0 * float(seconds.get(name, 0.0)) / batches if batches else 0.0

    def per_request_ms(name: str) -> float:
        return 1000.0 * float(worker.get(name, 0.0)) / calls if calls else 0.0

    residual = max(
        0.0,
        float(seconds.get("batch_cycle_seconds", 0.0))
        - float(seconds.get("inference_dispatch_seconds", 0.0))
        - float(seconds.get("dispatch_profile_bookkeeping_seconds", 0.0)),
    )
    latency = inference.get("latency_ms", {})
    latency = latency if isinstance(latency, Mapping) else {}
    request_latency = inference.get("request_latency_ms", {})
    request_latency = request_latency if isinstance(request_latency, Mapping) else {}
    near_live_floor = max(1, int(max_live_environments * 0.9))
    return {
        "batching": {
            "batches": batches,
            "launches_per_second": batches / wall_seconds if wall_seconds > 0 else 0.0,
            "mean_batch_size": float(inference.get("mean_batch_size", 0.0)),
            "maximum_observed_batch_size": max(histogram, default=0),
            "deadline_fraction": (
                int((inference.get("counters") or {}).get("batches_deadline", 0)) / batches
                if batches else 0.0
            ),
            "tiny_batch_le_8_fraction": (
                sum(count for size, count in histogram.items() if size <= 8) / batches
                if batches else 0.0
            ),
            "near_live_limit_fraction": (
                sum(count for size, count in histogram.items() if size >= near_live_floor) / batches
                if batches else 0.0
            ),
        },
        "per_batch_ms": {
            name: per_batch_ms(name)
            for name in (
                "batch_coalesce_seconds",
                "batch_cycle_seconds",
                "prepare_records_seconds",
                "collate_cpu_seconds",
                "h2d_wall_seconds",
                "h2d_gpu_seconds",
                "dtype_cast_wall_seconds",
                "model_wall_seconds",
                "gpu_model_seconds",
                "d2h_decode_seconds",
            )
        },
        "per_request_ms": {
            "compiler": per_request_ms("compiler_seconds"),
            "ipc_connection_wait": per_request_ms("ipc_connection_wait_seconds"),
            "ipc_send": per_request_ms("ipc_send_seconds"),
            "ipc_response_wait": per_request_ms("ipc_response_wait_seconds"),
            "ipc_roundtrip": per_request_ms("ipc_roundtrip_seconds"),
            "server_request_mean": (
                float(request_latency.get("total", 0.0))
                / int(request_latency.get("count", 0))
                if int(request_latency.get("count", 0)) else 0.0
            ),
            "ipc_ingress_mean": float(
                (latency.get("ipc_ingress") or {}).get("mean", 0.0)
            ),
            "handler_wakeup_mean": float(
                (latency.get("handler_wakeup") or {}).get("mean", 0.0)
            ),
            "queue_wait_mean": float(
                (latency.get("queue_wait") or {}).get("mean", 0.0)
            ),
            "ipc_egress_send_concurrent_mean": (
                1000.0 * float(seconds.get("ipc_egress_send_seconds", 0.0)) / calls
                if calls else 0.0
            ),
        },
        "wall_fractions": {
            "gpu_model_active": (
                float(seconds.get("gpu_model_seconds", 0.0)) / wall_seconds
                if wall_seconds > 0 else 0.0
            ),
            "central_collate_cpu": (
                float(seconds.get("collate_cpu_seconds", 0.0)) / wall_seconds
                if wall_seconds > 0 else 0.0
            ),
            "h2d_gpu": (
                float(seconds.get("h2d_gpu_seconds", 0.0)) / wall_seconds
                if wall_seconds > 0 else 0.0
            ),
        },
        "dispatch_handoff_residual": {
            "seconds": residual,
            "milliseconds_per_batch": 1000.0 * residual / batches if batches else 0.0,
            "definition": "batch_cycle - inference_dispatch - profile_bookkeeping",
        },
    }


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
    compiler_workers: int,
    worker_local_compiler: bool,
    worker_compiler_backend: str,
    async_h2d: bool,
    resident_tensor_cache: bool,
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
            compiler_workers=compiler_workers,
            worker_local_compiler=worker_local_compiler,
            worker_compiler_backend=worker_compiler_backend,
            async_h2d=async_h2d,
            resident_tensor_cache=resident_tensor_cache,
        )
    return replace(
        base,
        inference_profile=True,
        inference_forced_action_shortcut=forced_action_shortcut,
        engine_pool_size=engine_pool_size,
        compiler_workers=compiler_workers,
        worker_local_compiler=worker_local_compiler,
        worker_compiler_backend=worker_compiler_backend,
        async_h2d=async_h2d,
        resident_tensor_cache=resident_tensor_cache,
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
    compiler_workers: int = 1,
    worker_local_compiler: bool = False,
    worker_compiler_backend: str = "policy_stateless",
    async_h2d: bool = False,
    resident_tensor_cache: bool = False,
    policy_root: Path | None = None,
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
                    root=policy_root or catalog.candidate_policy.root,
                    device="cuda:0",
                    batch_size=batch_size,
                    batch_wait_ms=batch_wait_ms,
                    label="profile",
                    ability_repeat_limit=8,
                    inference_dtype=inference_dtype,
                    profile=True,
                    forced_action_shortcut=forced,
                    compiler_workers=compiler_workers,
                    async_h2d=async_h2d,
                    resident_tensor_cache=resident_tensor_cache,
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
                compiler_workers=compiler_workers,
                worker_local_compiler=worker_local_compiler,
                worker_compiler_backend=worker_compiler_backend,
                async_h2d=async_h2d,
                resident_tensor_cache=resident_tensor_cache,
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
            compiler_workers=compiler_workers,
            worker_local_compiler=worker_local_compiler,
            worker_compiler_backend=worker_compiler_backend,
            async_h2d=async_h2d,
            resident_tensor_cache=resident_tensor_cache,
        )
        rss_monitor = _ProcessTreeRssMonitor(os.getpid())
        rss_monitor.start()
        started = time.perf_counter()
        try:
            result = run_batch(config)
            wall_seconds = time.perf_counter() - started
        finally:
            peak_process_tree_rss_bytes = rss_monitor.stop()
            process_tree_cpu_seconds = rss_monitor.cpu_seconds
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
            "compiler_workers": compiler_workers,
            "worker_local_compiler": worker_local_compiler,
            "worker_compiler_backend": worker_compiler_backend,
            "async_h2d": async_h2d,
            "resident_tensor_cache": resident_tensor_cache,
            "policy_root": str((policy_root or catalog.candidate_policy.root).resolve()),
            "inference_channels_per_role": min(engine_pool_size, 8),
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
            "process_tree_cpu_seconds": process_tree_cpu_seconds,
            "average_cpu_cores": process_tree_cpu_seconds / wall_seconds,
            "average_cpu_percent": 100.0 * process_tree_cpu_seconds / wall_seconds,
        },
    }
    payload["diagnosis"] = derive_pipeline_diagnostics(
        inference or {},
        payload["worker"],
        wall_seconds=wall_seconds,
        max_live_environments=min(games, workers * engine_pool_size),
    )
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
    parser.add_argument("--compiler-workers", type=int, default=1)
    parser.add_argument("--worker-local-compiler", action="store_true")
    parser.add_argument(
        "--worker-compiler-backend",
        choices=("policy_stateless", "0035_incremental"),
        default="policy_stateless",
    )
    parser.add_argument("--async-h2d", action="store_true")
    parser.add_argument("--resident-tensor-cache", action="store_true")
    parser.add_argument("--policy-root", type=Path)
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
        compiler_workers=args.compiler_workers,
        worker_local_compiler=args.worker_local_compiler,
        worker_compiler_backend=args.worker_compiler_backend,
        async_h2d=args.async_h2d,
        resident_tensor_cache=args.resident_tensor_cache,
        policy_root=args.policy_root,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
