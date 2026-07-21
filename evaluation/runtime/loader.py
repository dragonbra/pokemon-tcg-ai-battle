from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from evaluation.packages.loader import (
    PackageValidationError,
    SubmissionPackage,
    _clear_cg_modules,
)


_NATIVE_SUFFIXES = {".dll", ".dylib", ".so"}


def _included_files(cg_root: Path) -> list[Path]:
    return [
        path
        for path in cg_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]


def compute_cg_manifest(
    cg_root: Path,
) -> dict[str, object]:
    """计算 cg 目录内容清单，并在完整 runtime 场景校验必要文件。"""
    resolved_root = cg_root.resolve()
    if not resolved_root.is_dir():
        raise PackageValidationError("cg runtime directory is required")

    files = _included_files(resolved_root)
    file_hashes = {
        path.relative_to(resolved_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    python_files = sorted(path for path in file_hashes if path.endswith(".py"))
    native_files = sorted(path for path in file_hashes if Path(path).suffix in _NATIVE_SUFFIXES)

    for required_path in ("__init__.py", "game.py", "api.py"):
        if required_path not in file_hashes:
            raise PackageValidationError(f"cg runtime requires {required_path}")
    if not native_files:
        raise PackageValidationError("cg runtime requires a native library")

    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(file_hashes.items()):
        digest.update(f"{relative_path}:{file_hash}\n".encode("utf-8"))
    return {
        "files": dict(sorted(file_hashes.items())),
        "file_count": len(file_hashes),
        "python_files": python_files,
        "native_files": native_files,
        "tree_hash": digest.hexdigest(),
    }


def assert_cg_compatible(candidate: SubmissionPackage, opponent: SubmissionPackage) -> None:
    if candidate.cg_manifest["tree_hash"] != opponent.cg_manifest["tree_hash"]:
        raise PackageValidationError("candidate/opponent cg hash mismatch")


def load_game_api(runtime_root: Path) -> object:
    """仅从指定 runtime 的 cg/game.py 加载游戏接口。"""
    cg_root = (runtime_root / "cg").resolve()
    _clear_cg_modules()
    try:
        compute_cg_manifest(cg_root)
        package_path = cg_root / "__init__.py"
        if package_path.is_file():
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
        else:
            raise PackageValidationError("cg/__init__.py is required")

        game_path = cg_root / "game.py"
        game_spec = importlib.util.spec_from_file_location("cg.game", game_path)
        if game_spec is None or game_spec.loader is None:
            raise PackageValidationError("cg/game.py could not be loaded")
        game_module = importlib.util.module_from_spec(game_spec)
        sys.modules["cg.game"] = game_module
        game_spec.loader.exec_module(game_module)
        return game_module
    except Exception as exc:
        if isinstance(exc, PackageValidationError):
            raise
        raise PackageValidationError(f"cg/game.py could not be loaded: {exc}") from exc
    finally:
        _clear_cg_modules()
