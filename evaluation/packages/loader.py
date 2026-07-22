from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator


class PackageValidationError(ValueError):
    """提交 package 未满足评测加载契约。"""


SUBPROCESS_VALIDATION_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class SubmissionPackage:
    name: str
    root: Path
    deck: list[int]
    entrypoint: Path
    package_hash: str
    deck_hash: str
    cg_manifest: dict[str, object]


def _clear_cg_modules() -> None:
    for module_name in [
        name for name in sys.modules if name == "cg" or name.startswith("cg.")
    ]:
        sys.modules.pop(module_name, None)


def _read_deck(deck_path: Path, official_card_ids: set[int]) -> list[int]:
    if not deck_path.is_file():
        raise PackageValidationError("deck.csv is required")

    deck: list[int] = []
    for line_number, raw_value in enumerate(
        deck_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw_value.strip()
        if not value:
            continue
        try:
            card_id = int(value)
        except ValueError as exc:
            raise PackageValidationError(
                f"deck.csv line {line_number} must be an integer card ID"
            ) from exc
        if card_id <= 0:
            raise PackageValidationError(f"deck.csv line {line_number} must be a positive card ID")
        if card_id not in official_card_ids:
            raise PackageValidationError(f"deck.csv contains unknown official card ID: {card_id}")
        deck.append(card_id)

    if not deck:
        raise PackageValidationError("deck.csv must not be empty")
    if len(deck) != 60:
        raise PackageValidationError(f"deck.csv must contain 60 cards, got {len(deck)}")
    return deck


def _load_submission_module(entrypoint: Path, root: Path) -> ModuleType:
    module_name = "_evaluation_submission_" + hashlib.sha256(
        str(entrypoint.resolve()).encode("utf-8")
    ).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise PackageValidationError("main.py could not be loaded")

    module = importlib.util.module_from_spec(spec)
    root_text = str(root)
    _clear_cg_modules()
    sys.path.insert(0, root_text)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        raise PackageValidationError(f"main.py could not be loaded: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(root_text)
        _clear_cg_modules()
    return module


def _validate_agent(entrypoint: Path, root: Path, deck: list[int]) -> None:
    module = _load_submission_module(entrypoint, root)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise PackageValidationError("main.py must define a callable agent")
    try:
        returned_deck = agent({"select": None})
    except BaseException as exc:
        raise PackageValidationError(f"agent deck callback failed: {exc}") from exc
    if not isinstance(returned_deck, list) or returned_deck != deck:
        raise PackageValidationError("agent({'select': None}) must return the deck.csv card IDs")


_VALIDATION_SUCCESS_FD = 198
_VALIDATION_SUCCESS_MARKER = b"EVALUATION_AGENT_VALIDATION_OK"

_SUBPROCESS_VALIDATION_SCRIPT = f"""
import os
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from evaluation.packages.loader import _validate_agent


try:
    entrypoint = Path(sys.argv[1])
    package_root = Path(sys.argv[2])
    deck = [int(card_id) for card_id in sys.argv[4:]]
    _validate_agent(entrypoint, package_root, deck)
except BaseException as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1) from exc
else:
    os.write({_VALIDATION_SUCCESS_FD}, {_VALIDATION_SUCCESS_MARKER!r})
finally:
    try:
        os.close({_VALIDATION_SUCCESS_FD})
    except OSError:
        pass
"""


@contextmanager
def _validation_success_pipe() -> Iterator[int]:
    saved_fixed_fd: int | None = None
    saved_fixed_fd_inheritable = False
    read_fd: int | None = None
    write_fd: int | None = None
    fixed_fd_installed = False

    try:
        try:
            saved_fixed_fd_inheritable = os.get_inheritable(_VALIDATION_SUCCESS_FD)
        except OSError:
            pass
        else:
            saved_fixed_fd = os.dup(_VALIDATION_SUCCESS_FD)

        read_fd, write_fd = os.pipe()
        if read_fd == _VALIDATION_SUCCESS_FD:
            moved_read_fd = os.dup(read_fd)
            os.close(read_fd)
            read_fd = moved_read_fd

        if write_fd == _VALIDATION_SUCCESS_FD:
            os.set_inheritable(write_fd, True)
        else:
            os.dup2(write_fd, _VALIDATION_SUCCESS_FD, inheritable=True)
        fixed_fd_installed = True
        os.set_blocking(read_fd, False)
        yield read_fd
    finally:
        fds_to_close = {fd for fd in (read_fd, write_fd) if fd is not None}
        if fixed_fd_installed:
            fds_to_close.add(_VALIDATION_SUCCESS_FD)
        for fd in fds_to_close:
            try:
                os.close(fd)
            except OSError:
                pass

        if saved_fixed_fd is not None:
            try:
                if fixed_fd_installed:
                    os.dup2(
                        saved_fixed_fd,
                        _VALIDATION_SUCCESS_FD,
                        inheritable=saved_fixed_fd_inheritable,
                    )
            finally:
                os.close(saved_fixed_fd)


def _validate_agent_in_subprocess(entrypoint: Path, root: Path, deck: list[int]) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_path = [str(repository_root)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    try:
        with _validation_success_pipe() as validation_read_fd:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        _SUBPROCESS_VALIDATION_SCRIPT,
                        str(entrypoint),
                        str(root),
                        str(repository_root),
                        *(str(card_id) for card_id in deck),
                    ],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=SUBPROCESS_VALIDATION_TIMEOUT_SECONDS,
                    pass_fds=(_VALIDATION_SUCCESS_FD,),
                )
            except subprocess.TimeoutExpired as exc:
                raise PackageValidationError("main.py subprocess validation timed out") from exc

            try:
                success_marker = os.read(
                    validation_read_fd,
                    len(_VALIDATION_SUCCESS_MARKER),
                )
            except BlockingIOError:
                success_marker = b""
    except PackageValidationError:
        raise
    except OSError as exc:
        raise PackageValidationError(f"main.py subprocess validation failed: {exc}") from exc

    if result.returncode == 0 and success_marker == _VALIDATION_SUCCESS_MARKER:
        return
    detail = result.stderr.strip() or result.stdout.strip()
    if not detail:
        detail = f"subprocess exited with code {result.returncode}"
    raise PackageValidationError(f"main.py subprocess validation failed: {detail}")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compute_package_hash(
    root: Path,
    entrypoint: Path,
    deck_path: Path,
    cg_manifest: dict[str, object],
) -> str:
    files = cg_manifest["files"]
    if not isinstance(files, dict):
        raise PackageValidationError("cg manifest files must be a mapping")
    digest = hashlib.sha256()
    components = [("deck.csv", _hash_file(deck_path)), ("main.py", _hash_file(entrypoint))]
    components.extend((f"cg/{path}", file_hash) for path, file_hash in files.items())
    for relative_path, file_hash in sorted(components):
        digest.update(f"{relative_path}:{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def load_submission_package(
    root: Path,
    official_card_ids: set[int],
    name: str | None = None,
) -> SubmissionPackage:
    """读取 package，并在隔离子进程中验证 agent 的启动回调。"""
    package_root = root.absolute()
    if not package_root.is_dir():
        raise PackageValidationError(f"package root does not exist: {package_root}")

    entrypoint = package_root / "main.py"
    if not entrypoint.is_file():
        raise PackageValidationError("main.py is required")
    deck_path = package_root / "deck.csv"
    deck = _read_deck(deck_path, official_card_ids)

    from evaluation.runtime.loader import compute_cg_manifest

    cg_manifest = compute_cg_manifest(package_root / "cg")
    _validate_agent_in_subprocess(entrypoint, package_root, deck)
    return SubmissionPackage(
        name=name or package_root.name,
        root=package_root,
        deck=deck,
        entrypoint=entrypoint,
        package_hash=_compute_package_hash(package_root, entrypoint, deck_path, cg_manifest),
        deck_hash=_hash_file(deck_path),
        cg_manifest=cg_manifest,
    )
