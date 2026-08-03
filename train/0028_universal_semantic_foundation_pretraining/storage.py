"""Fail-closed storage guards for 0028 data materialization."""

from __future__ import annotations

import shutil
from pathlib import Path


GIB = 1024**3
DEFAULT_DATASET_LIMIT_BYTES = 100 * GIB
DEFAULT_LINUX_FREE_FLOOR_BYTES = 50 * GIB
DEFAULT_C_FREE_FLOOR_BYTES = 30 * GIB


def directory_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def guard_storage(
    dataset_root: Path,
    *,
    dataset_limit_bytes: int = DEFAULT_DATASET_LIMIT_BYTES,
    linux_free_floor_bytes: int = DEFAULT_LINUX_FREE_FLOOR_BYTES,
    c_free_floor_bytes: int = DEFAULT_C_FREE_FLOOR_BYTES,
    c_mount: Path = Path("/mnt/c"),
) -> dict[str, int]:
    probe = dataset_root if dataset_root.exists() else dataset_root.parent
    probe.mkdir(parents=True, exist_ok=True)
    dataset_bytes = directory_bytes(dataset_root)
    linux_free = shutil.disk_usage(probe).free
    c_free = shutil.disk_usage(c_mount).free if c_mount.is_dir() else 2**63 - 1
    if dataset_bytes >= dataset_limit_bytes:
        raise RuntimeError(
            f"dataset hard limit reached: {dataset_bytes} >= {dataset_limit_bytes}"
        )
    if linux_free < linux_free_floor_bytes:
        raise RuntimeError(
            f"Linux free-space floor reached: {linux_free} < {linux_free_floor_bytes}"
        )
    if c_free < c_free_floor_bytes:
        raise RuntimeError(f"C: free-space floor reached: {c_free} < {c_free_floor_bytes}")
    return {
        "dataset_bytes": dataset_bytes,
        "dataset_limit_bytes": dataset_limit_bytes,
        "linux_free_bytes": linux_free,
        "linux_free_floor_bytes": linux_free_floor_bytes,
        "c_free_bytes": c_free,
        "c_free_floor_bytes": c_free_floor_bytes,
    }


__all__ = [
    "DEFAULT_C_FREE_FLOOR_BYTES",
    "DEFAULT_DATASET_LIMIT_BYTES",
    "DEFAULT_LINUX_FREE_FLOOR_BYTES",
    "GIB",
    "directory_bytes",
    "guard_storage",
]
