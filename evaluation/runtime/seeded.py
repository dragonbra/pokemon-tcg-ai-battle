from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SOURCE = REPOSITORY_ROOT / "engine/source/ptcgProgram 22"
ADAPTER_SOURCE = Path(__file__).with_name("seeded_engine.cpp")
BUILD_ROOT = REPOSITORY_ROOT / "engine/build/seeded_official"
ABI_SCHEMA = "seeded_official_engine_abi_v1"
RUNTIME_VERSION = 2
BUILD_FLAGS = (
    "-std=c++20",
    "-O2",
    "-fPIC",
    "-shared",
    "-pthread",
    "-static-libstdc++",
    "-static-libgcc",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def source_tree_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in OFFICIAL_SOURCE.rglob("*") if item.is_file()):
        relative = path.relative_to(OFFICIAL_SOURCE).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class SeededRuntimeManifest:
    runtime_version: int
    abi_schema: str
    official_source_sha256: str
    adapter_source_sha256: str
    compiler: str
    build_key: str
    library_path: Path
    library_sha256: str
    manifest_path: Path

    def json_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["library_path"] = str(self.library_path.relative_to(REPOSITORY_ROOT))
        payload["manifest_path"] = str(self.manifest_path.relative_to(REPOSITORY_ROOT))
        return payload


def _compiler_identity(compiler: str) -> str:
    result = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0]


def _manifest_from_payload(path: Path, payload: dict[str, Any]) -> SeededRuntimeManifest:
    return SeededRuntimeManifest(
        runtime_version=int(payload["runtime_version"]),
        abi_schema=str(payload["abi_schema"]),
        official_source_sha256=str(payload["official_source_sha256"]),
        adapter_source_sha256=str(payload["adapter_source_sha256"]),
        compiler=str(payload["compiler"]),
        build_key=str(payload["build_key"]),
        library_path=REPOSITORY_ROOT / str(payload["library_path"]),
        library_sha256=str(payload["library_sha256"]),
        manifest_path=path,
    )


def build_seeded_runtime(*, compiler: str = "g++") -> SeededRuntimeManifest:
    official_hash = source_tree_sha256()
    adapter_hash = _sha256_file(ADAPTER_SOURCE)
    compiler_identity = _compiler_identity(compiler)
    build_key = _sha256_bytes(
        f"{ABI_SCHEMA}\0{official_hash}\0{adapter_hash}\0{compiler_identity}\0"
        f"{' '.join(BUILD_FLAGS)}".encode("utf-8")
    )[:20]
    output_dir = BUILD_ROOT / f"{RUNTIME_VERSION:04d}"
    library_path = output_dir / "libcg.so"
    manifest_path = output_dir / "manifest.json"
    lock_path = BUILD_ROOT / ".build.lock"
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if library_path.is_file() and manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = _manifest_from_payload(manifest_path, payload)
            if (
                manifest.runtime_version == RUNTIME_VERSION
                and manifest.abi_schema == ABI_SCHEMA
                and manifest.official_source_sha256 == official_hash
                and manifest.adapter_source_sha256 == adapter_hash
                and manifest.compiler == compiler_identity
                and manifest.build_key == build_key
                and manifest.library_sha256 == _sha256_file(library_path)
            ):
                return manifest

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output_dir, prefix="libcg-", suffix=".so", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            subprocess.run(
                [
                    compiler,
                    *BUILD_FLAGS,
                    "-I",
                    str(OFFICIAL_SOURCE),
                    str(ADAPTER_SOURCE),
                    "-o",
                    str(temporary_path),
                ],
                check=True,
                cwd=REPOSITORY_ROOT,
            )
            os.replace(temporary_path, library_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        manifest = SeededRuntimeManifest(
            runtime_version=RUNTIME_VERSION,
            abi_schema=ABI_SCHEMA,
            official_source_sha256=official_hash,
            adapter_source_sha256=adapter_hash,
            compiler=compiler_identity,
            build_key=build_key,
            library_path=library_path,
            library_sha256=_sha256_file(library_path),
            manifest_path=manifest_path,
        )
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest.json_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        return manifest


class _StartData(ctypes.Structure):
    _fields_ = [
        ("battle_ptr", ctypes.c_void_p),
        ("error_player", ctypes.c_int),
        ("error_type", ctypes.c_int),
    ]


class _SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("select_player", ctypes.c_int),
    ]


_LOADED: dict[Path, Any] = {}


def load_seeded_library(path: Path) -> Any:
    resolved = path.resolve()
    if resolved in _LOADED:
        return _LOADED[resolved]
    library = ctypes.cdll.LoadLibrary(str(resolved))
    library.GameInitialize.restype = None
    library.GameInitialize.argtypes = []
    library.SeededRuntimeAbi.restype = ctypes.c_char_p
    library.SeededRuntimeAbi.argtypes = []
    library.ConfigureSeeds.restype = ctypes.c_int
    library.ConfigureSeeds.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong]
    library.BattleStartSeeded.restype = _StartData
    library.BattleStartSeeded.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_ulonglong]
    library.GetBattleData.restype = _SerialData
    library.GetBattleData.argtypes = [ctypes.c_void_p]
    library.Select.restype = ctypes.c_int
    library.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    library.BattleFinish.restype = None
    library.BattleFinish.argtypes = [ctypes.c_void_p]
    library.GameInitialize()
    actual_abi = library.SeededRuntimeAbi().decode("ascii")
    if actual_abi != ABI_SCHEMA:
        raise RuntimeError(f"seeded runtime ABI mismatch: {actual_abi}")
    _LOADED[resolved] = library
    return library


def configure_seeded_library(
    library: Any, *, engine_seed: int, search_seed: int
) -> None:
    if not 0 < engine_seed <= 0xFFFFFFFF:
        raise ValueError("engine_seed must be in [1, 2**32-1]")
    if not 0 < search_seed <= 0xFFFFFFFF:
        raise ValueError("search_seed must be in [1, 2**32-1]")
    error = int(library.ConfigureSeeds(engine_seed, search_seed))
    if error != 0:
        raise RuntimeError(f"seeded runtime configuration failed: {error}")


class SeededBattle:
    def __init__(
        self,
        library_path: Path,
        deck0: list[int],
        deck1: list[int],
        *,
        engine_seed: int,
        search_seed: int,
    ) -> None:
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("both decks must contain exactly 60 cards")
        self.library = load_seeded_library(library_path)
        configure_seeded_library(
            self.library, engine_seed=engine_seed, search_seed=search_seed
        )
        self.cards = tuple(int(card_id) for card_id in (*deck0, *deck1))
        self.engine_seed = engine_seed
        self.pointer: int | None = None

    def _observation(self) -> dict[str, Any]:
        if self.pointer is None:
            raise RuntimeError("battle is not started")
        serial = self.library.GetBattleData(self.pointer)
        if not serial.json:
            raise RuntimeError("official engine returned no observation")
        observation = json.loads(serial.json.decode("utf-8"))
        observation["search_begin_input"] = ctypes.string_at(
            serial.data, serial.count
        ).decode("ascii")
        return observation

    def start(self) -> dict[str, Any]:
        if self.pointer is not None:
            raise RuntimeError("battle is already started")
        argument = (ctypes.c_int * len(self.cards))(*self.cards)
        result = self.library.BattleStartSeeded(argument, self.engine_seed)
        if not result.battle_ptr or result.error_type:
            raise RuntimeError(
                "seeded battle start failed: "
                f"player={result.error_player} type={result.error_type}"
            )
        self.pointer = int(result.battle_ptr)
        return self._observation()

    def select(self, action: list[int]) -> dict[str, Any]:
        if self.pointer is None:
            raise RuntimeError("battle is not started")
        argument = (ctypes.c_int * len(action))(*action)
        error = int(self.library.Select(self.pointer, argument, len(action)))
        if error != 0:
            raise RuntimeError(f"official engine rejected action: {error}")
        return self._observation()

    def finish(self) -> None:
        if self.pointer is None:
            return
        self.library.BattleFinish(self.pointer)
        self.pointer = None


def main() -> int:
    manifest = build_seeded_runtime()
    print(json.dumps(manifest.json_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
