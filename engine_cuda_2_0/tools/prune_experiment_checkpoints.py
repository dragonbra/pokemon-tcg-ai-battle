from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


SERIAL_PATTERNS = (
    re.compile(r"^policy_iter[_-]?(\d+)\.(?:pt|pth|ckpt)$", re.IGNORECASE),
    re.compile(r"^checkpoint_iter[_-]?(\d+)\.(?:pt|pth|ckpt)$", re.IGNORECASE),
    re.compile(r"^bc_transformer_epoch[_-]?(\d+)\.(?:pt|pth|ckpt)$", re.IGNORECASE),
    re.compile(r"^policy_epoch[_-]?(\d+)\.(?:pt|pth|ckpt)$", re.IGNORECASE),
    re.compile(r"^model_epoch[_-]?(\d+)\.(?:pt|pth|ckpt)$", re.IGNORECASE),
)

BEST_NAMES = {
    "best.pt",
    "best.pth",
    "checkpoint_best.pt",
    "model_best.pt",
    "policy_best.pt",
    "bc_transformer_best.pt",
}

LAST_NAMES = {
    "checkpoint_last.pt",
    "checkpoint_latest.pt",
    "model_last.pt",
    "policy_last.pt",
}


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class RetentionGroup:
    experiment_dir: str
    keep: FileRecord
    delete: tuple[FileRecord, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep one recognized checkpoint snapshot per experiment directory."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--validate-kept-torch",
        action="store_true",
        help="Fully torch.load every retained checkpoint before writing or deleting.",
    )
    return parser.parse_args()


def serial_number(path: Path) -> int | None:
    for pattern in SERIAL_PATTERNS:
        match = pattern.match(path.name)
        if match:
            return int(match.group(1))
    return None


def recognized(path: Path) -> bool:
    lowered = path.name.lower()
    return (
        lowered in BEST_NAMES
        or lowered in LAST_NAMES
        or serial_number(path) is not None
    )


def retention_key(path: Path) -> tuple[int, int, int, str]:
    lowered = path.name.lower()
    if lowered in BEST_NAMES:
        priority = 3
        sequence = 0
    elif lowered in LAST_NAMES:
        priority = 2
        sequence = 0
    else:
        priority = 1
        sequence = serial_number(path) or 0
    stat = path.stat()
    return priority, sequence, stat.st_mtime_ns, path.name


def record(path: Path) -> FileRecord:
    stat = path.stat()
    return FileRecord(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def build_plan(root: Path) -> list[RetentionGroup]:
    by_directory: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and recognized(path):
            by_directory.setdefault(path.parent, []).append(path)

    groups: list[RetentionGroup] = []
    for directory, candidates in sorted(by_directory.items()):
        if len(candidates) < 2:
            continue
        keep = max(candidates, key=retention_key)
        delete = tuple(
            record(path) for path in sorted(candidates) if path != keep
        )
        groups.append(
            RetentionGroup(
                experiment_dir=str(directory),
                keep=record(keep),
                delete=delete,
            )
        )
    return groups


def validate_record(root: Path, item: FileRecord) -> Path:
    path = Path(item.path)
    resolved_root = root.resolve()
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise RuntimeError(f"path escaped checkpoint root: {resolved_path}")
    stat = resolved_path.stat()
    if stat.st_size != item.bytes or stat.st_mtime_ns != item.mtime_ns:
        raise RuntimeError(f"checkpoint changed since planning: {resolved_path}")
    return resolved_path


def apply_plan(root: Path, groups: list[RetentionGroup]) -> tuple[int, int]:
    validated: list[Path] = []
    for group in groups:
        validate_record(root, group.keep)
        validated.extend(validate_record(root, item) for item in group.delete)
    for path in validated:
        path.unlink()
    return len(validated), sum(
        item.bytes for group in groups for item in group.delete
    )


def validate_kept_torch(groups: list[RetentionGroup]) -> None:
    import torch

    for group in groups:
        path = Path(group.keep.path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload is None:
            raise RuntimeError(f"retained checkpoint loaded as None: {path}")
        del payload


def main() -> None:
    args = parse_args()
    root = args.root.resolve(strict=True)
    if not root.is_dir():
        raise SystemExit(f"checkpoint root is not a directory: {root}")

    groups = build_plan(root)
    if args.validate_kept_torch:
        validate_kept_torch(groups)
    delete_files = sum(len(group.delete) for group in groups)
    delete_bytes = sum(item.bytes for group in groups for item in group.delete)
    payload = {
        "version": 1,
        "root": str(root),
        "selection": "best, then last, then highest numbered iteration/epoch",
        "retained_torch_load_validated": args.validate_kept_torch,
        "groups": [asdict(group) for group in groups],
        "summary": {
            "experiment_directories": len(groups),
            "delete_files": delete_files,
            "delete_bytes": delete_bytes,
            "delete_gib": round(delete_bytes / 1024**3, 3),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.apply:
        removed_files, removed_bytes = apply_plan(root, groups)
        payload["applied"] = {
            "removed_files": removed_files,
            "removed_bytes": removed_bytes,
        }
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
