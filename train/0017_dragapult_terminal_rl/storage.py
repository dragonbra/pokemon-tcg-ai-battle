from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


GIB = 1024**3


@dataclass(frozen=True)
class StorageSnapshot:
    root_free_gib: float
    windows_free_gib: float
    version_bytes: int
    warning: bool

    def metrics(self) -> dict[str, float]:
        return {
            "system/disk/root_free_gib": self.root_free_gib,
            "system/disk/mnt_c_free_gib": self.windows_free_gib,
            "system/disk/version_gib": self.version_bytes / GIB,
            "system/disk/warning": float(self.warning),
        }


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def storage_guard(
    version_root: Path,
    *,
    warning_free_gib: float = 80.0,
    hard_stop_free_gib: float = 50.0,
    version_cap_gib: float = 20.0,
) -> StorageSnapshot:
    root_free = shutil.disk_usage(Path("/")).free / GIB
    windows_path = Path("/mnt/c") if Path("/mnt/c").exists() else Path("/")
    windows_free = shutil.disk_usage(windows_path).free / GIB
    version_bytes = _tree_bytes(version_root)
    if root_free < hard_stop_free_gib or windows_free < hard_stop_free_gib:
        raise RuntimeError(
            f"disk hard stop: /={root_free:.2f} GiB, /mnt/c={windows_free:.2f} GiB"
        )
    if version_bytes > version_cap_gib * GIB:
        raise RuntimeError(
            f"version storage cap exceeded: {version_bytes / GIB:.2f} GiB"
        )
    return StorageSnapshot(
        root_free,
        windows_free,
        version_bytes,
        root_free < warning_free_gib or windows_free < warning_free_gib,
    )


__all__ = ["StorageSnapshot", "storage_guard"]
