from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SOURCE_SCHEMA = "0031_model_only_checkpoint_v1"
OUTPUT_SCHEMA = "0031_compact_shared_prototype_fp32_v1"
PROJECT_ID = "0031_rule_faithful_semantic_foundation_pretraining"
CANONICAL_PREFIX = "prototype_encoder."
ALIAS_PREFIXES = (
    "state_encoder.prototypes.",
    "option_encoder.prototypes.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(state_dict: dict[str, Any], torch: Any) -> int:
    return sum(
        int(value.numel() * value.element_size())
        for value in state_dict.values()
        if torch.is_tensor(value)
    )


def extract_checkpoint(source: Path, destination: Path) -> Path:
    if source.suffix == ".pt":
        return source
    with tarfile.open(source, mode="r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        if len(files) != 1:
            raise ValueError("source archive must contain exactly one checkpoint file")
        member = files[0]
        name = PurePosixPath(member.name)
        if name.name != member.name or name.suffix != ".pt":
            raise ValueError("source checkpoint must be a top-level .pt file")
        extracted = destination / name.name
        reader = archive.extractfile(member)
        if reader is None:
            raise ValueError("cannot read checkpoint archive member")
        with reader, extracted.open("xb") as output:
            while chunk := reader.read(1024 * 1024):
                output.write(chunk)
        return extracted


def compact_state_dict(state_dict: dict[str, Any], torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = {
        name[len(CANONICAL_PREFIX) :]: tensor
        for name, tensor in state_dict.items()
        if name.startswith(CANONICAL_PREFIX)
    }
    if not canonical:
        raise ValueError("checkpoint has no prototype_encoder weights")
    duplicate_keys: list[str] = []
    for suffix, tensor in canonical.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"prototype weight is not a tensor: {suffix}")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"prototype weight is not FP32: {suffix} ({tensor.dtype})")
        for prefix in ALIAS_PREFIXES:
            alias = f"{prefix}{suffix}"
            other = state_dict.get(alias)
            if (
                not torch.is_tensor(other)
                or other.shape != tensor.shape
                or other.dtype != tensor.dtype
                or not torch.equal(other, tensor)
            ):
                raise ValueError(f"prototype alias is not exactly equal: {alias}")
            duplicate_keys.append(alias)
    compact = {name: tensor for name, tensor in state_dict.items() if name not in duplicate_keys}
    floating_dtypes = sorted(
        {str(tensor.dtype) for tensor in compact.values() if torch.is_tensor(tensor) and tensor.is_floating_point()}
    )
    if floating_dtypes != ["torch.float32"]:
        raise ValueError(f"checkpoint floating storage must remain FP32: {floating_dtypes}")
    return compact, {
        "canonical_prefix": CANONICAL_PREFIX,
        "alias_prefixes": list(ALIAS_PREFIXES),
        "canonical_tensor_count": len(canonical),
        "dropped_duplicate_keys": len(duplicate_keys),
        "floating_storage_dtype": "float32",
        "runtime_dtype": "float32",
    }


def compact_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    import torch

    with tempfile.TemporaryDirectory(prefix="ptcg_semantic0031_compact_") as temporary:
        checkpoint = extract_checkpoint(source, Path(temporary))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("unsupported semantic0031 source checkpoint")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, dict) or metadata.get("project_id") != PROJECT_ID:
        raise ValueError("semantic0031 project mismatch")
    if not isinstance(state_dict, dict):
        raise ValueError("semantic0031 state_dict is missing")
    source_bytes = tensor_bytes(state_dict, torch)
    compact, storage = compact_state_dict(state_dict, torch)
    compact_bytes = tensor_bytes(compact, torch)
    storage.update(
        {
            "source_schema_version": SOURCE_SCHEMA,
            "source_checkpoint_sha256": sha256_file(source),
            "source_state_keys": len(state_dict),
            "compact_state_keys": len(compact),
            "source_tensor_bytes": source_bytes,
            "compact_tensor_bytes": compact_bytes,
            "removed_tensor_bytes": source_bytes - compact_bytes,
        }
    )
    result = {
        "schema_version": OUTPUT_SCHEMA,
        "state_dict": compact,
        "metadata": metadata,
        "storage": storage,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(result, output)
    return {
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "storage": storage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep one exact FP32 semantic0031 prototype encoder and restore aliases at load time."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = compact_checkpoint(args.source.resolve(), args.output.resolve())
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
