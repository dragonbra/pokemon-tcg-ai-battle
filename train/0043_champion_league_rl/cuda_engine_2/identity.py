"""Fail-closed source, ABI, build, GPU, and private-rule identity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from ..assets import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")


class CudaEngineIdentityError(RuntimeError):
    pass


def _tracked_source_hash(repository_root: Path, engine_root: Path) -> str:
    """Hash the bytes actually present for every tracked engine source file."""

    relative = engine_root.relative_to(repository_root).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", relative],
        cwd=repository_root, capture_output=True, check=True,
    )
    names = [item.decode() for item in result.stdout.split(b"\0") if item]
    if not names:
        raise CudaEngineIdentityError("CUDA Engine 2.0 has no tracked source files")
    digest = hashlib.sha256()
    for name in sorted(names):
        path = repository_root / name
        if not path.is_file():
            raise CudaEngineIdentityError(f"tracked CUDA Engine 2.0 file is missing: {name}")
        encoded = name.encode()
        data = path.read_bytes()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _constant(path: Path, name: str) -> int:
    match = re.search(rf"constexpr\s+std::uint32_t\s+{re.escape(name)}\s*=\s*(\d+)\s*;", path.read_text())
    if not match:
        raise CudaEngineIdentityError(f"missing CUDA ABI constant {name}: {path}")
    return int(match.group(1))


@dataclass(frozen=True, slots=True)
class CudaEngineIdentity:
    engine_label: str
    engine_source_sha256: str
    official_state_abi: int
    official_rule_abi: int
    official_rule_pack_sha256: str
    compute_capability: tuple[int, int]
    device_name: str
    runtime_dtype: str
    semantic_codec: str
    binary_path: str | None
    binary_sha256: str | None
    extension_path: str | None
    extension_sha256: str | None
    status: str = "PASS"
    schema_version: str = "0043_cuda_engine_2_identity_v1"

    @classmethod
    def resolve(
        cls, repository_root: Path, *, rule_pack: Path,
        binary: Path | None = None, extension: Path | None = None,
        require_gpu: bool = True, require_extension: bool = False,
    ) -> "CudaEngineIdentity":
        import torch

        root = Path(repository_root).resolve()
        engine = root / "engine_cuda_2_0"
        config = json.loads(CONFIG_PATH.read_text())
        if config.get("schema_version") != "0043_cuda_engine_2_config_v1":
            raise CudaEngineIdentityError("unsupported 0043 CUDA Engine 2.0 config")
        tree = _tracked_source_hash(root, engine)
        if tree != config["engine_source_sha256"]:
            raise CudaEngineIdentityError(f"CUDA Engine 2.0 source identity mismatch: {tree}")
        state_abi = _constant(engine / "include/ptcg_cuda/official_state_pod.cuh", "kOfficialStateAbiVersion")
        rule_abi = _constant(engine / "include/ptcg_cuda/official_rule_layout.cuh", "kOfficialRuleAbiVersion")
        if (state_abi, rule_abi) != (config["official_state_abi"], config["official_rule_abi"]):
            raise CudaEngineIdentityError("CUDA Engine 2.0 ABI identity mismatch")
        rule_hash = sha256_file(rule_pack)
        if rule_hash != config["official_rule_pack_sha256"]:
            raise CudaEngineIdentityError("private official rule-pack identity mismatch")
        if require_gpu and not torch.cuda.is_available():
            raise CudaEngineIdentityError("CUDA Engine 2.0 requires an available CUDA device")
        capability = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (0, 0)
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
        if require_gpu and capability != tuple(config["required_compute_capability"]):
            raise CudaEngineIdentityError(f"unsupported GPU compute capability: {capability}")
        for label, path in (("binary", binary), ("extension", extension)):
            if path is not None and (not path.is_file() or path.stat().st_size == 0):
                raise CudaEngineIdentityError(f"missing CUDA Engine 2.0 {label}: {path}")
        if require_extension and extension is None:
            raise CudaEngineIdentityError("CUDA Engine 2.0 PyTorch extension is required")
        return cls(
            engine_label=config["engine_label"], engine_source_sha256=tree,
            official_state_abi=state_abi, official_rule_abi=rule_abi,
            official_rule_pack_sha256=rule_hash,
            compute_capability=capability, device_name=device,
            runtime_dtype=config["runtime_dtype"], semantic_codec=config["semantic_codec"],
            binary_path=str(binary.resolve()) if binary else None,
            binary_sha256=sha256_file(binary) if binary else None,
            extension_path=str(extension.resolve()) if extension else None,
            extension_sha256=sha256_file(extension) if extension else None,
        )

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["CudaEngineIdentity", "CudaEngineIdentityError"]
