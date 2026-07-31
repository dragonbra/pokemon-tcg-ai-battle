"""Bounded SSD accounting and checkpoint retention for long League runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

GIB = 1024 ** 3


@dataclass(frozen=True)
class StorageSnapshot:
    free_bytes: int
    version_bytes: int
    stop_requested: bool

    def metrics(self) -> dict[str, float]:
        return {
            "system/disk/free_gib": self.free_bytes / GIB,
            "system/disk/version_gib": self.version_bytes / GIB,
            "system/disk/stop_requested": float(self.stop_requested),
        }


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def preflight_storage(path: Path, *, minimum_free_gib: float = 100.0) -> StorageSnapshot:
    free = shutil.disk_usage(path.resolve().anchor).free
    if free < minimum_free_gib * GIB:
        raise RuntimeError(f"League launch requires {minimum_free_gib:.0f} GiB free; found {free / GIB:.2f} GiB")
    return StorageSnapshot(free, _tree_bytes(path), False)


def runtime_storage(path: Path, *, stop_free_gib: float = 80.0, version_cap_gib: float = 10.0) -> StorageSnapshot:
    free = shutil.disk_usage(path.resolve().anchor).free
    size = _tree_bytes(path)
    if size > version_cap_gib * GIB:
        raise RuntimeError(f"League version exceeded {version_cap_gib:.0f} GiB cap: {size / GIB:.2f} GiB")
    return StorageSnapshot(free, size, free < stop_free_gib * GIB)


def prune_latest(checkpoint_dir: Path, *, keep: int = 2, protected: set[Path] | None = None) -> list[Path]:
    if keep < 1:
        raise ValueError("checkpoint retention must be positive")
    protected = {item.resolve() for item in (protected or set())}
    checkpoints = sorted(checkpoint_dir.glob("update-*.pt"))
    removable = [item for item in checkpoints[:-keep] if item.resolve() not in protected]
    for item in removable:
        item.unlink(); item.with_suffix(item.suffix + ".sha256").unlink(missing_ok=True)
    return removable


__all__ = ["StorageSnapshot", "preflight_storage", "prune_latest", "runtime_storage"]
