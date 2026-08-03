"""Linux process diagnostics used only by the 0030 performance benchmark."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeMaps:
    torch: bool
    cuda: bool
    engine: bool


def parse_kib_field(text: str, field: str) -> int:
    prefix = f"{field}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return 0


def classify_maps(text: str) -> RuntimeMaps:
    lowered = text.lower()
    return RuntimeMaps(
        torch="libtorch" in lowered or "/torch/" in lowered,
        cuda="libcuda.so" in lowered or "libcudart.so" in lowered,
        engine="libcg.so" in lowered,
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def _spawn_children(parent_pid: int) -> list[int]:
    children = _read(Path(f"/proc/{parent_pid}/task/{parent_pid}/children"))
    output: list[int] = []
    for value in children.split():
        pid = int(value)
        command = _read(Path(f"/proc/{pid}/cmdline")).replace("\x00", " ")
        if "multiprocessing.spawn" in command and "spawn_main" in command:
            output.append(pid)
    return output


def _memory_fields() -> tuple[int, int]:
    text = _read(Path("/proc/meminfo"))
    available = parse_kib_field(text, "MemAvailable")
    swap_total = parse_kib_field(text, "SwapTotal")
    swap_free = parse_kib_field(text, "SwapFree")
    return available, max(0, swap_total - swap_free)


class WorkerProcessSampler:
    """Periodically samples direct spawn children without touching the rollout hot path."""

    def __init__(self, *, interval_seconds: float = 1.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        self.parent_pid = os.getpid()
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_count = 0
        self.peak_pss_total_bytes = 0
        self.peak_pss_mean_bytes = 0
        self.observed_pids: set[int] = set()
        self.torch_pids: set[int] = set()
        self.cuda_pids: set[int] = set()
        self.engine_pids: set[int] = set()
        available, swap_used = _memory_fields()
        self.min_available_bytes = available
        self.initial_swap_used_bytes = swap_used
        self.max_swap_used_bytes = swap_used
        self.final_swap_used_bytes = swap_used

    def _sample(self) -> None:
        pids = _spawn_children(self.parent_pid)
        pss_values: list[int] = []
        for pid in pids:
            self.observed_pids.add(pid)
            rollup = _read(Path(f"/proc/{pid}/smaps_rollup"))
            if rollup:
                pss_values.append(parse_kib_field(rollup, "Pss"))
            maps = classify_maps(_read(Path(f"/proc/{pid}/maps")))
            if maps.torch:
                self.torch_pids.add(pid)
            if maps.cuda:
                self.cuda_pids.add(pid)
            if maps.engine:
                self.engine_pids.add(pid)
        pss_total = sum(pss_values)
        self.peak_count = max(self.peak_count, len(pids))
        if pss_total > self.peak_pss_total_bytes:
            self.peak_pss_total_bytes = pss_total
            self.peak_pss_mean_bytes = (
                pss_total // len(pss_values) if pss_values else 0
            )
        available, swap_used = _memory_fields()
        if available:
            self.min_available_bytes = min(self.min_available_bytes, available)
        self.max_swap_used_bytes = max(self.max_swap_used_bytes, swap_used)
        self.final_swap_used_bytes = swap_used

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("worker sampler already started")
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="0030-worker-process-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._sample()

    def metrics(self) -> dict[str, int]:
        return {
            "worker_count_peak": self.peak_count,
            "worker_processes_observed": len(self.observed_pids),
            "worker_pss_total_peak_bytes": self.peak_pss_total_bytes,
            "worker_pss_mean_at_peak_bytes": self.peak_pss_mean_bytes,
            "worker_torch_import_count": len(self.torch_pids),
            "worker_cuda_mapping_count": len(self.cuda_pids),
            "worker_engine_mapping_count": len(self.engine_pids),
            "host_mem_available_min_bytes": self.min_available_bytes,
            "host_swap_used_initial_bytes": self.initial_swap_used_bytes,
            "host_swap_used_peak_bytes": self.max_swap_used_bytes,
            "host_swap_used_final_bytes": self.final_swap_used_bytes,
        }


__all__ = [
    "RuntimeMaps",
    "WorkerProcessSampler",
    "classify_maps",
    "parse_kib_field",
]
