from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_STORAGE_PATH = Path("/mnt/c")
DEFAULT_MIN_FREE_GIB = 20.0


@dataclass(frozen=True)
class StorageReport:
    """Filesystem capacity snapshot used before generating RL artifacts."""

    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int

    @property
    def free_gib(self) -> float:
        return self.free_bytes / (1024**3)

    def to_dict(self) -> dict[str, str | int | float]:
        return {**asdict(self), "free_gib": self.free_gib}


def inspect_storage(path: str | Path = DEFAULT_STORAGE_PATH) -> StorageReport:
    """Inspect a path, falling back to ``/`` outside WSL."""
    requested = Path(path)
    actual = requested if requested.exists() else Path("/")
    usage = shutil.disk_usage(actual)
    return StorageReport(
        path=str(actual),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
    )


def assert_storage_safe(
    path: str | Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> StorageReport:
    """Raise before a large run if the monitored filesystem is near full."""
    if min_free_gib < 0:
        raise ValueError("min_free_gib must not be negative")
    report = inspect_storage(path)
    if report.free_gib < min_free_gib:
        raise RuntimeError(
            f"insufficient storage on {report.path}: "
            f"{report.free_gib:.2f} GiB free, need at least {min_free_gib:.2f} GiB; "
            "clean old RL artifacts before continuing"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check WSL storage before an RL run")
    parser.add_argument("--path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    report = assert_storage_safe(args.path, args.min_free_gib)
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
