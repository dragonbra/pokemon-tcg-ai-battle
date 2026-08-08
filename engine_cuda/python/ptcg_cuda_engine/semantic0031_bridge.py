from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
import types
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SEMANTIC0031_SCHEMA = "0031_rule_faithful_semantic_decision_v2"
SEMANTIC0031_PACKAGE_SCHEMA = "0031_pt0805_compact_candidate_v1"
SEMANTIC0031_CHECKPOINT_SCHEMA = "0031_compact_fp16_storage_v1"
SEMANTIC0031_FP32_SHARED_CHECKPOINT_SCHEMA = "0031_compact_shared_prototype_fp32_v1"
SEMANTIC0031_CHECKPOINT_SCHEMAS = frozenset(
    {SEMANTIC0031_CHECKPOINT_SCHEMA, SEMANTIC0031_FP32_SHARED_CHECKPOINT_SCHEMA}
)
SEMANTIC0031_PROJECT = "0031_rule_faithful_semantic_foundation_pretraining"
KNOWN_0031_ARCHIVE_SHA256 = "b28b6217c8e412974128672dbdf193ebdd7a1521f5a5eb9f1ce9b34e8a6cb7d6"

FIELD_PAD = 0
FIELD_PRESENT = 1
FIELD_UNKNOWN = 2
FIELD_NOT_APPLICABLE = 3

_REQUIRED_PACKAGE_FILES = frozenset(
    {
        "deck.csv",
        "main.py",
        "manifest.json",
        "semantic0031/__init__.py",
        "semantic0031/assets/official_public_prototypes_v1.json",
        "semantic0031/assets/official_full_engine_prototypes_v2.json",
        "semantic0031/contracts/batch.py",
        "semantic0031/contracts/fields.py",
        "semantic0031/domain/prototypes.py",
        "semantic0031/model/config.py",
        "semantic0031/model/policy.py",
    }
)
_MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 1024 * 1024 * 1024

_GLOBAL_CAT_WIDTH = 12
_GLOBAL_NUM_WIDTH = 24
_CARD_CAT_WIDTH = 9
_CARD_NUM_WIDTH = 7
_RESOURCE_CAT_WIDTH = 4
_RESOURCE_NUM_WIDTH = 15
_EVENT_CAT_WIDTH = 31
_EVENT_NUM_WIDTH = 4
_OPTION_CAT_WIDTH = 19
_OPTION_NUM_WIDTH = 2
_POLICY_GLOBAL_CAT_WIDTH = 8
_POLICY_GLOBAL_NUM_WIDTH = 16
_POLICY_ENTITY_CAT_WIDTH = 6
_POLICY_ENTITY_NUM_WIDTH = 10
_POLICY_OPTION_CAT_WIDTH = 12
_POLICY_OPTION_NUM_WIDTH = 4
_MAX_CARD_ID = 2048
_MAX_ATTACK_ID = 2048
_SEMANTIC_HISTORY_CAPACITY = 64
_SEMANTIC_HISTORY_PARAM_CAPACITY = 7
_SEMANTIC_HISTORY_KEYS = frozenset(
    {
        "semantic0031_history_total_count",
        "semantic0031_history_write_index",
        "semantic0031_history_log_type",
        "semantic0031_history_param_count",
        "semantic0031_history_params",
    }
)
_SEMANTIC0031_V2_REQUIRED_KEYS = frozenset(
    {
        "global_cat",
        "global_num",
        "global_state",
        "card_cat",
        "card_num",
        "card_state",
        "card_parent",
        "card_mask",
        "resource_cat",
        "resource_num",
        "resource_state",
        "resource_mask",
        "event_cat",
        "event_num",
        "event_state",
        "event_mask",
        "event_source",
        "event_target",
        "event_before",
        "event_after",
        "option_cat",
        "option_num",
        "option_state",
        "option_mask",
        "option_source",
        "option_target",
        "option_context",
        "option_effect_card",
        "option_skill_id",
        "option_skill_role",
        "option_skill_parent",
        "option_skill_mask",
        "option_effect_id",
        "option_effect_role",
        "option_effect_parent",
        "option_effect_mask",
        "min_count",
        "max_count",
        "targets",
    }
)
_SEMANTIC0031_V2_MASK_KEYS = frozenset(
    {
        "card_mask",
        "resource_mask",
        "event_mask",
        "option_mask",
        "option_skill_mask",
        "option_effect_mask",
    }
)
_SEMANTIC0031_V2_FLOAT_KEYS = frozenset(
    {
        "global_num",
        "card_num",
        "resource_num",
        "event_num",
        "option_num",
    }
)
_SEMANTIC0031_V2_LONG_KEYS = _SEMANTIC0031_V2_REQUIRED_KEYS - (
    _SEMANTIC0031_V2_MASK_KEYS | _SEMANTIC0031_V2_FLOAT_KEYS
)

# PolicyCodecV1 zones -> semantic0031 compiler zones. Prize rows are deliberately
# omitted below because the canonical 0031 compiler represents prize uncertainty
# in its resource ledger, not as visible card instances.
_ZONE_MAP = (0, 1, 2, 3, 4, 0, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member_name(raw_name: str) -> str:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ValueError(f"unsafe archive member name: {raw_name!r}")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {raw_name!r}")
    if path.parts and ":" in path.parts[0]:
        raise ValueError(f"drive-qualified archive member path: {raw_name!r}")
    return path.as_posix()


def inspect_semantic0031_archive(path: str | Path) -> dict[str, Any]:
    """Inspect a candidate archive without extracting or importing its code."""

    archive = Path(path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    names: set[str] = set()
    total_bytes = 0
    files = 0
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle.getmembers():
            name = _safe_member_name(member.name)
            if name in names:
                raise ValueError(f"duplicate archive member: {name}")
            names.add(name)
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are forbidden: {name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported archive member type: {name}")
            if member.size < 0 or member.size > _MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(f"archive member is too large: {name}")
            if member.isfile():
                files += 1
                total_bytes += int(member.size)
                if total_bytes > _MAX_ARCHIVE_TOTAL_BYTES:
                    raise ValueError("archive expands beyond the size limit")
    missing = sorted(_REQUIRED_PACKAGE_FILES - names)
    if missing:
        raise ValueError(f"semantic0031 archive is missing required files: {missing}")
    return {
        "path": str(archive),
        "sha256": _sha256_file(archive),
        "members": len(names),
        "files": files,
        "uncompressed_bytes": total_bytes,
    }


def _extract_archive_safely(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle.getmembers():
            name = _safe_member_name(member.name)
            target = (root / Path(*PurePosixPath(name).parts)).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"archive member escapes extraction root: {name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"unsupported archive member type: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {name}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _read_manifest(root: Path) -> dict[str, Any]:
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("semantic0031 manifest root must be an object")
    if raw.get("schema_version") != SEMANTIC0031_PACKAGE_SCHEMA:
        raise ValueError("unsupported semantic0031 package schema")
    if raw.get("project_id") != SEMANTIC0031_PROJECT:
        raise ValueError("semantic0031 package project mismatch")
    return raw


def _read_deck(root: Path, manifest: Mapping[str, Any]) -> list[int]:
    deck_path = root / "deck.csv"
    deck = [
        int(value) for value in deck_path.read_text(encoding="ascii").splitlines() if value.strip()
    ]
    if len(deck) != 60 or any(card <= 0 or card > _MAX_CARD_ID for card in deck):
        raise ValueError("semantic0031 deck must contain 60 supported positive card IDs")
    canonical_deck = ",".join(str(card) for card in sorted(deck)).encode("ascii")
    if hashlib.sha256(canonical_deck).hexdigest() != manifest.get("deck_sha256"):
        raise ValueError("semantic0031 deck hash mismatch")
    expected_counts = Counter(
        {int(card_id): int(count) for card_id, count in manifest.get("deck_counts", ())}
    )
    if Counter(deck) != expected_counts:
        raise ValueError("semantic0031 deck counts do not match the manifest")
    return deck


def _validate_package_files(root: Path, manifest: Mapping[str, Any]) -> Path:
    missing = sorted(name for name in _REQUIRED_PACKAGE_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"semantic0031 package is missing required files: {missing}")
    checkpoint_name = manifest.get("compact_checkpoint")
    if (
        not isinstance(checkpoint_name, str)
        or not checkpoint_name.endswith(".pt")
        or PurePosixPath(checkpoint_name).name != checkpoint_name
    ):
        raise ValueError("semantic0031 checkpoint path is invalid")
    checkpoint = root / checkpoint_name
    if not checkpoint.is_file():
        raise ValueError("semantic0031 checkpoint path is invalid")
    if checkpoint.stat().st_size != int(manifest.get("compact_checkpoint_bytes", -1)):
        raise ValueError("semantic0031 checkpoint size mismatch")
    if _sha256_file(checkpoint) != manifest.get("compact_checkpoint_sha256"):
        raise ValueError("semantic0031 checkpoint hash mismatch")
    return checkpoint


def _load_package_modules(root: Path, identity: str) -> dict[str, Any]:
    path_digest = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:12]
    namespace = f"_ptcg_cuda_0031_{identity[:16]}_{path_digest}"
    existing = sys.modules.get(namespace)
    if existing is not None:
        if getattr(existing, "_ptcg_package_root", None) != str(root.resolve()):
            raise RuntimeError(f"dynamic semantic0031 namespace collision: {namespace}")
    else:
        package = types.ModuleType(namespace)
        package.__package__ = namespace
        package.__path__ = [str(root)]
        package._ptcg_package_root = str(root.resolve())
        sys.modules[namespace] = package

        semantic_name = f"{namespace}.semantic0031"
        spec = importlib.util.spec_from_file_location(
            semantic_name,
            root / "semantic0031" / "__init__.py",
            submodule_search_locations=[str(root / "semantic0031")],
        )
        if spec is None or spec.loader is None:
            sys.modules.pop(namespace, None)
            raise RuntimeError("cannot create semantic0031 package import spec")
        semantic = importlib.util.module_from_spec(spec)
        sys.modules[semantic_name] = semantic
        try:
            spec.loader.exec_module(semantic)
        except Exception:
            for name in tuple(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    sys.modules.pop(name, None)
            raise

    return {
        "namespace": namespace,
        "prototypes": importlib.import_module(f"{namespace}.semantic0031.domain.prototypes"),
        "model": importlib.import_module(f"{namespace}.semantic0031.model"),
    }


_PROTOTYPE_CANONICAL_PREFIX = "prototype_encoder."
_PROTOTYPE_ALIAS_PREFIXES = (
    "state_encoder.prototypes.",
    "option_encoder.prototypes.",
)


def _expand_shared_prototype_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Restore registered prototype aliases from one exact FP32 canonical copy."""

    import torch

    canonical = {
        name[len(_PROTOTYPE_CANONICAL_PREFIX) :]: tensor
        for name, tensor in state_dict.items()
        if name.startswith(_PROTOTYPE_CANONICAL_PREFIX)
    }
    if not canonical:
        raise ValueError("semantic0031 checkpoint has no canonical prototype_encoder weights")
    expanded = {
        name: tensor
        for name, tensor in state_dict.items()
        if not any(name.startswith(prefix) for prefix in _PROTOTYPE_ALIAS_PREFIXES)
    }
    for suffix, tensor in canonical.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"semantic0031 prototype weight is not a tensor: {suffix}")
        for prefix in _PROTOTYPE_ALIAS_PREFIXES:
            alias = f"{prefix}{suffix}"
            existing = state_dict.get(alias)
            if existing is not None and (
                not torch.is_tensor(existing)
                or existing.shape != tensor.shape
                or existing.dtype != tensor.dtype
                or not torch.equal(existing, tensor)
            ):
                raise ValueError(f"semantic0031 prototype alias differs from canonical: {alias}")
            expanded[alias] = tensor
    return expanded


@dataclass
class Semantic0031Package:
    root: Path
    manifest: dict[str, Any]
    deck: list[int]
    model: Any
    config: Any
    prototypes: Any
    archive_sha256: str | None
    namespace: str
    _temporary: tempfile.TemporaryDirectory[str] | None = None

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def load_semantic0031_package(
    path: str | Path,
    device: Any,
    *,
    expected_archive_sha256: str = KNOWN_0031_ARCHIVE_SHA256,
    trusted_directory: bool = False,
    checkpoint_override: str | Path | None = None,
) -> Semantic0031Package:
    """Load the compact 0031 package with strict hashes and FP32 runtime weights.

    Archive source code is imported only after the caller-supplied commitment
    matches. Directories require an explicit trust opt-in because their Python
    source has no archive-level commitment.
    """

    import torch

    source = Path(path).resolve()
    temporary: tempfile.TemporaryDirectory[str] | None = None
    archive_sha256: str | None = None
    if source.is_file():
        report = inspect_semantic0031_archive(source)
        archive_sha256 = str(report["sha256"])
        if not expected_archive_sha256 or archive_sha256 != expected_archive_sha256.lower():
            raise ValueError("semantic0031 archive hash mismatch; refusing to import package code")
        temporary = tempfile.TemporaryDirectory(prefix="ptcg_semantic0031_")
        root = Path(temporary.name) / "package"
        _extract_archive_safely(source, root)
    elif source.is_dir():
        if not trusted_directory:
            raise ValueError("loading a semantic0031 directory requires trusted_directory=True")
        root = source
    else:
        raise FileNotFoundError(source)

    try:
        manifest = _read_manifest(root)
        package_checkpoint = _validate_package_files(root, manifest)
        deck = _read_deck(root, manifest)
        checkpoint = (
            Path(checkpoint_override).resolve()
            if checkpoint_override is not None
            else package_checkpoint
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha256 = _sha256_file(checkpoint)
        identity = archive_sha256 or checkpoint_sha256
        modules = _load_package_modules(root, identity)
        prototype_index = modules["prototypes"].PrototypeIndex.load(
            root / "semantic0031/assets/official_public_prototypes_v1.json",
            root / "semantic0031/assets/official_full_engine_prototypes_v2.json",
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") not in SEMANTIC0031_CHECKPOINT_SCHEMAS
        ):
            raise ValueError("unsupported semantic0031 checkpoint schema")
        metadata = payload.get("metadata")
        storage = payload.get("storage")
        if not isinstance(metadata, dict) or metadata.get("project_id") != SEMANTIC0031_PROJECT:
            raise ValueError("semantic0031 checkpoint project mismatch")
        if not isinstance(storage, dict) or storage.get("runtime_dtype") != "float32":
            raise ValueError("semantic0031 checkpoint runtime dtype mismatch")
        contract = metadata.get("model_config")
        config_payload = contract.get("config") if isinstance(contract, dict) else None
        if not isinstance(config_payload, dict):
            raise ValueError("semantic0031 checkpoint has no model config")
        config = modules["model"].ModelConfig(**config_payload)
        model = modules["model"].SemanticPolicy(config, prototype_index).eval()
        compact_state = payload.get("state_dict")
        if not isinstance(compact_state, dict):
            raise ValueError("semantic0031 checkpoint state_dict is missing")
        expanded_state = _expand_shared_prototype_state_dict(compact_state)
        payload["state_dict"] = expanded_state
        expected = set(model.state_dict())
        if set(expanded_state) != expected:
            raise ValueError(
                "expanded semantic0031 checkpoint does not match the model; "
                f"missing={sorted(expected - set(expanded_state))[:8]} "
                f"extra={sorted(set(expanded_state) - expected)[:8]}"
            )
        model.load_state_dict(expanded_state, strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        expected_count = int(
            contract.get(
                "parameter_count",
                manifest.get("import_validation", {}).get("parameter_count", -1),
            )
        )
        if parameter_count != expected_count:
            raise ValueError("semantic0031 model parameter count mismatch")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("semantic0031 parameters did not expand to FP32")
        model.requires_grad_(False)
        model.to(device=device, dtype=torch.float32)
        model.eval()
        return Semantic0031Package(
            root=root,
            manifest=manifest,
            deck=deck,
            model=model,
            config=config,
            prototypes=prototype_index,
            archive_sha256=archive_sha256,
            namespace=str(modules["namespace"]),
            _temporary=temporary,
        )
    except Exception:
        if temporary is not None:
            temporary.cleanup()
        raise


def semantic0031_static_fields(registered_deck: Sequence[int], device: Any) -> dict[str, Any]:
    """Build the exact registered-deck facts once on the target device."""

    import torch

    deck = [int(card) for card in registered_deck]
    if len(deck) != 60 or any(card <= 0 or card > _MAX_CARD_ID for card in deck):
        raise ValueError("registered deck must contain 60 supported positive card IDs")
    counts = Counter(deck)
    identities = sorted(counts)
    width = len(identities)
    return {
        "semantic0031_resource_card_ids": torch.tensor(
            identities, dtype=torch.long, device=device
        ).view(1, width),
        "semantic0031_resource_initial": torch.tensor(
            [counts[identity] for identity in identities],
            dtype=torch.float32,
            device=device,
        ).view(1, width),
        "semantic0031_resource_mask": torch.ones((1, width), dtype=torch.bool, device=device),
    }


def _require_shape(name: str, value: Any, rank: int, width: int | None = None) -> None:
    if value.ndim != rank or (width is not None and value.shape[-1] != width):
        raise ValueError(
            f"{name} must have rank {rank}" + (f" and width {width}" if width is not None else "")
        )


def _expand_static(value: Any, batch_size: int, name: str) -> Any:
    if value.ndim < 1:
        raise ValueError(f"static field {name} must have a batch axis")
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] == 1:
        return value.expand(batch_size, *value.shape[1:])
    raise ValueError(f"static field {name} has an incompatible batch axis")


def _one_based_entity_relation(index: Any, entity_mask: Any) -> Any:
    import torch

    maximum = entity_mask.shape[1]
    safe = (index.long() - 1).clamp(min=0, max=max(0, maximum - 1))
    valid = index.gt(0) & index.le(maximum)
    valid = valid & entity_mask.gather(1, safe)
    return torch.where(valid, index.long(), torch.zeros_like(index, dtype=torch.long))


def _relation_by_one_based_serial(serial: Any, card_cat: Any, card_mask: Any) -> Any:
    import torch

    card_serial = card_cat[..., 1].long()
    wanted = serial.long()
    matches = (
        wanted.gt(0).unsqueeze(-1)
        & card_mask.unsqueeze(1)
        & card_serial.unsqueeze(1).eq(wanted.unsqueeze(-1))
    )
    first = matches.long().argmax(dim=2) + 1
    return torch.where(matches.any(dim=2), first, torch.zeros_like(first))


def _positive_enum(value: Any, cap: int) -> Any:
    import torch

    shifted = (value.long() + 1).clamp(max=cap)
    return torch.where(value.long().ge(0), shifted, torch.zeros_like(shifted))


def _semantic_history_slots(total_count: Any, write_index: Any, capacity: int) -> tuple[Any, Any, Any]:
    import torch

    batch_size = total_count.shape[0]
    device = total_count.device
    positions = torch.arange(capacity, dtype=torch.long, device=device).view(1, capacity)
    live_count = total_count.long().clamp(min=0, max=capacity)
    start = torch.where(
        total_count.long().le(capacity),
        torch.zeros_like(write_index.long()),
        write_index.long().remainder(capacity),
    )
    slots = (start.view(batch_size, 1) + positions).remainder(capacity)
    live = positions.lt(live_count.view(batch_size, 1))
    return slots, live, live_count


def _semantic0031_events_from_history(
    batch: Mapping[str, Any],
    source_global_cat: Any,
    card_cat: Any,
    card_mask: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    import torch

    present = _SEMANTIC_HISTORY_KEYS & set(batch)
    batch_size = source_global_cat.shape[0]
    device = source_global_cat.device
    if not present:
        event_cat = torch.zeros((batch_size, 1, _EVENT_CAT_WIDTH), dtype=torch.long, device=device)
        event_num = torch.zeros((batch_size, 1, _EVENT_NUM_WIDTH), dtype=torch.float32, device=device)
        event_state = torch.zeros((batch_size, 1, _EVENT_NUM_WIDTH), dtype=torch.long, device=device)
        event_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        event_relation = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        return (
            event_cat,
            event_num,
            event_state,
            event_mask,
            event_relation,
            event_relation,
            event_relation,
            event_relation,
        )
    if present != _SEMANTIC_HISTORY_KEYS:
        missing = sorted(_SEMANTIC_HISTORY_KEYS - present)
        raise KeyError(f"semantic0031 history fields are incomplete: {missing}")

    total_count = batch["semantic0031_history_total_count"]
    write_index = batch["semantic0031_history_write_index"]
    log_type = batch["semantic0031_history_log_type"]
    param_count = batch["semantic0031_history_param_count"]
    params = batch["semantic0031_history_params"]
    if total_count.shape != (batch_size,) or write_index.shape != (batch_size,):
        raise ValueError("semantic0031 history counters must have shape [batch]")
    if log_type.shape != (batch_size, _SEMANTIC_HISTORY_CAPACITY):
        raise ValueError("semantic0031 history log_type must have shape [batch,64]")
    if param_count.shape != log_type.shape:
        raise ValueError("semantic0031 history param_count shape disagrees")
    if params.shape != (
        batch_size,
        _SEMANTIC_HISTORY_CAPACITY,
        _SEMANTIC_HISTORY_PARAM_CAPACITY,
    ):
        raise ValueError("semantic0031 history params must have shape [batch,64,7]")

    slots, live, live_count = _semantic_history_slots(
        total_count,
        write_index,
        _SEMANTIC_HISTORY_CAPACITY,
    )
    ordered_type = log_type.long().gather(1, slots)
    ordered_param_count = param_count.long().gather(1, slots)
    ordered_params = params.long().gather(
        1,
        slots.unsqueeze(-1).expand(
            batch_size,
            _SEMANTIC_HISTORY_CAPACITY,
            _SEMANTIC_HISTORY_PARAM_CAPACITY,
        ),
    )
    known_log = live & ordered_type.ge(0) & ordered_type.lt(24)

    event_cat = torch.zeros(
        (batch_size, _SEMANTIC_HISTORY_CAPACITY, _EVENT_CAT_WIDTH),
        dtype=torch.long,
        device=device,
    )
    event_num = torch.zeros(
        (batch_size, _SEMANTIC_HISTORY_CAPACITY, _EVENT_NUM_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    event_state = torch.zeros(
        (batch_size, _SEMANTIC_HISTORY_CAPACITY, _EVENT_NUM_WIDTH),
        dtype=torch.long,
        device=device,
    )
    event_mask = known_log
    event_cat[..., 0] = torch.where(known_log, ordered_type + 1, torch.zeros_like(ordered_type))

    actor = (source_global_cat[:, 3].long() - 1).clamp(min=0, max=1).view(batch_size, 1)
    player = ordered_params[..., 0]
    has_player = (
        known_log
        & ordered_param_count.ge(1)
        & (
            ordered_type.eq(0)
            | ordered_type.eq(1)
            | ordered_type.eq(2)
            | ordered_type.eq(3)
            | ordered_type.eq(4)
            | ordered_type.eq(6)
            | ordered_type.eq(7)
            | ordered_type.eq(8)
            | ordered_type.eq(9)
            | ordered_type.eq(10)
            | ordered_type.eq(11)
            | ordered_type.eq(12)
            | ordered_type.eq(13)
            | ordered_type.eq(14)
            | ordered_type.eq(15)
            | ordered_type.eq(16)
            | ordered_type.eq(17)
            | ordered_type.eq(18)
            | ordered_type.eq(19)
            | ordered_type.eq(20)
            | ordered_type.eq(21)
            | ordered_type.eq(22)
        )
        & player.ge(0)
        & player.le(1)
    )
    relative_player = torch.where(
        player.eq(actor),
        torch.ones_like(player),
        torch.full_like(player, 2),
    )
    event_cat[..., 1] = torch.where(
        has_player,
        relative_player,
        torch.where(known_log, torch.full_like(player, 3), torch.zeros_like(player)),
    )

    is_has_basic = known_log & ordered_type.eq(1) & ordered_param_count.ge(2)
    event_cat[..., 21] = torch.where(
        is_has_basic,
        torch.where(
            ordered_params[..., 1].eq(0),
            torch.ones_like(player),
            torch.full_like(player, 2),
        ),
        event_cat[..., 21],
    )

    is_draw = known_log & ordered_type.eq(4) & ordered_param_count.ge(3)
    is_hidden_draw = (
        is_draw
        & player.ge(0)
        & player.le(1)
        & player.ne(actor)
    )
    is_visible_draw = is_draw & ~is_hidden_draw
    event_cat[..., 0] = torch.where(
        is_hidden_draw,
        torch.full_like(ordered_type, 6),
        event_cat[..., 0],
    )
    is_move_card = known_log & ordered_type.eq(6) & ordered_param_count.ge(5)
    is_move_hidden = known_log & ordered_type.eq(7) & ordered_param_count.ge(3)
    is_switch = known_log & ordered_type.eq(8) & ordered_param_count.ge(5)
    is_change = known_log & ordered_type.eq(9) & ordered_param_count.ge(5)
    is_play = known_log & ordered_type.eq(10) & ordered_param_count.ge(3)
    is_target_event = (
        known_log
        & (ordered_type.eq(11) | ordered_type.eq(12) | ordered_type.eq(13))
        & ordered_param_count.ge(5)
    )
    is_move_attached = known_log & ordered_type.eq(14) & ordered_param_count.ge(7)
    is_attack = known_log & ordered_type.eq(15) & ordered_param_count.ge(4)
    is_hp_change = known_log & ordered_type.eq(16) & ordered_param_count.ge(5)
    is_status = (
        known_log
        & ordered_type.ge(17)
        & ordered_type.le(21)
        & ordered_param_count.ge(4)
    )
    draw_card_id = ordered_params[..., 1].clamp(min=0, max=_MAX_CARD_ID)
    draw_serial = ordered_params[..., 2].clamp(min=0, max=255)
    move_card_id = ordered_params[..., 1].clamp(min=0, max=_MAX_CARD_ID)
    move_serial = ordered_params[..., 2].clamp(min=0, max=255)
    event_card_id = torch.where(is_move_card, move_card_id, torch.where(is_visible_draw, draw_card_id, torch.zeros_like(draw_card_id)))
    event_serial = torch.where(is_move_card, move_serial, torch.where(is_visible_draw, draw_serial, torch.zeros_like(draw_serial)))
    has_visible_identity = (is_visible_draw & draw_card_id.gt(0)) | (is_move_card & move_card_id.gt(0))
    has_serial = is_visible_draw | is_move_card
    event_cat[..., 2] = event_card_id
    event_cat[..., 5] = torch.where(
        is_move_card,
        _positive_enum(ordered_params[..., 3], 32),
        torch.where(is_move_hidden, _positive_enum(ordered_params[..., 1], 32), torch.zeros_like(draw_serial)),
    )
    event_cat[..., 6] = torch.where(
        is_move_card,
        _positive_enum(ordered_params[..., 4], 32),
        torch.where(is_move_hidden, _positive_enum(ordered_params[..., 2], 32), torch.zeros_like(draw_serial)),
    )
    event_cat[..., 7] = torch.where(
        known_log,
        torch.where(has_visible_identity, torch.full_like(draw_card_id, 2), torch.ones_like(draw_card_id)),
        torch.zeros_like(draw_card_id),
    )
    event_cat[..., 8] = torch.where(
        known_log,
        torch.where(has_serial, torch.full_like(draw_serial, 2), torch.ones_like(draw_serial)),
        torch.zeros_like(draw_serial),
    )
    event_cat[..., 9] = torch.where(known_log, torch.ones_like(draw_serial), torch.zeros_like(draw_serial))
    event_cat[..., 14] = torch.where(has_serial, event_serial + 1, torch.zeros_like(draw_serial))

    generic_card_event = is_play | is_target_event | is_move_attached | is_attack | is_hp_change
    generic_card_id = ordered_params[..., 1].clamp(min=0, max=_MAX_CARD_ID)
    generic_serial = ordered_params[..., 2].clamp(min=0, max=255)
    status_card_id = ordered_params[..., 2].clamp(min=0, max=_MAX_CARD_ID)
    status_serial = ordered_params[..., 3].clamp(min=0, max=255)
    event_cat[..., 2] = torch.where(
        generic_card_event,
        generic_card_id,
        torch.where(is_status, status_card_id, event_cat[..., 2]),
    )
    visible_generic = (generic_card_event & generic_card_id.gt(0)) | (is_status & status_card_id.gt(0))
    event_cat[..., 7] = torch.where(
        generic_card_event | is_status,
        torch.where(visible_generic, torch.full_like(player, 2), torch.ones_like(player)),
        event_cat[..., 7],
    )
    event_cat[..., 8] = torch.where(
        generic_card_event | is_status,
        torch.full_like(player, 2),
        event_cat[..., 8],
    )
    event_cat[..., 14] = torch.where(
        generic_card_event,
        generic_serial + 1,
        torch.where(is_status, status_serial + 1, event_cat[..., 14]),
    )

    event_cat[..., 10] = torch.where(
        is_switch,
        ordered_params[..., 1].clamp(min=0, max=_MAX_CARD_ID),
        event_cat[..., 10],
    )
    event_cat[..., 11] = torch.where(
        is_switch,
        ordered_params[..., 3].clamp(min=0, max=_MAX_CARD_ID),
        event_cat[..., 11],
    )
    event_cat[..., 16] = torch.where(
        is_switch,
        ordered_params[..., 2].clamp(min=0, max=255) + 1,
        event_cat[..., 16],
    )
    event_cat[..., 17] = torch.where(
        is_switch,
        ordered_params[..., 4].clamp(min=0, max=255) + 1,
        event_cat[..., 17],
    )
    event_cat[..., 12] = torch.where(
        is_change,
        ordered_params[..., 1].clamp(min=0, max=_MAX_CARD_ID),
        torch.where(
            is_move_attached,
            ordered_params[..., 3].clamp(min=0, max=_MAX_CARD_ID),
            event_cat[..., 12],
        ),
    )
    event_cat[..., 13] = torch.where(
        is_change,
        ordered_params[..., 3].clamp(min=0, max=_MAX_CARD_ID),
        torch.where(
            is_move_attached,
            ordered_params[..., 5].clamp(min=0, max=_MAX_CARD_ID),
            event_cat[..., 13],
        ),
    )
    event_cat[..., 18] = torch.where(
        is_change,
        ordered_params[..., 2].clamp(min=0, max=255) + 1,
        torch.where(
            is_move_attached,
            ordered_params[..., 4].clamp(min=0, max=255) + 1,
            event_cat[..., 18],
        ),
    )
    event_cat[..., 19] = torch.where(
        is_change,
        ordered_params[..., 4].clamp(min=0, max=255) + 1,
        torch.where(
            is_move_attached,
            ordered_params[..., 6].clamp(min=0, max=255) + 1,
            event_cat[..., 19],
        ),
    )
    event_cat[..., 3] = torch.where(
        is_target_event,
        ordered_params[..., 3].clamp(min=0, max=_MAX_CARD_ID),
        event_cat[..., 3],
    )
    event_cat[..., 9] = torch.where(
        is_target_event,
        torch.full_like(player, 2),
        event_cat[..., 9],
    )
    event_cat[..., 15] = torch.where(
        is_target_event,
        ordered_params[..., 4].clamp(min=0, max=255) + 1,
        event_cat[..., 15],
    )
    event_cat[..., 4] = torch.where(
        is_attack,
        ordered_params[..., 3].clamp(min=0, max=4096),
        event_cat[..., 4],
    )
    event_cat[..., 20] = torch.where(
        is_status,
        torch.where(
            ordered_params[..., 1].eq(0),
            torch.ones_like(player),
            torch.full_like(player, 2),
        ),
        event_cat[..., 20],
    )
    event_cat[..., 23] = torch.where(
        is_hp_change,
        torch.where(
            ordered_params[..., 4].eq(0),
            torch.ones_like(player),
            torch.full_like(player, 2),
        ),
        event_cat[..., 23],
    )

    is_result = known_log & ordered_type.eq(23) & ordered_param_count.ge(2)
    result = ordered_params[..., 0]
    reason = ordered_params[..., 1]
    event_cat[..., 24] = torch.where(
        is_result & result.ge(0) & result.lt(15),
        result + 1,
        torch.zeros_like(result),
    )
    event_cat[..., 25] = torch.where(
        is_result & reason.ge(0) & reason.lt(31),
        reason + 1,
        torch.zeros_like(reason),
    )
    is_coin = known_log & ordered_type.eq(22) & ordered_param_count.ge(2)
    head = ordered_params[..., 1]
    event_cat[..., 22] = torch.where(
        is_coin & head.ge(0) & head.le(1),
        head + 1,
        torch.zeros_like(head),
    )

    positions = torch.arange(
        _SEMANTIC_HISTORY_CAPACITY, dtype=torch.long, device=device
    ).view(1, _SEMANTIC_HISTORY_CAPACITY)
    age = (live_count.view(batch_size, 1) - 1 - positions).clamp(min=0)
    event_num[..., 0] = torch.where(known_log, age.float(), torch.zeros_like(age).float())
    event_state[..., 0] = torch.where(
        known_log,
        torch.full_like(age, FIELD_PRESENT),
        torch.zeros_like(age),
    )
    event_state[..., 1:] = torch.where(
        known_log.unsqueeze(-1),
        torch.full_like(event_state[..., 1:], FIELD_UNKNOWN),
        event_state[..., 1:],
    )
    event_num[..., 1] = torch.where(
        is_hp_change,
        ordered_params[..., 3].float(),
        event_num[..., 1],
    )
    event_state[..., 1] = torch.where(
        is_hp_change,
        torch.full_like(event_state[..., 1], FIELD_PRESENT),
        event_state[..., 1],
    )

    source_serial = torch.where(
        is_status,
        status_serial + 1,
        generic_serial + 1,
    )
    event_source = torch.where(
        is_visible_draw | is_move_card | generic_card_event | is_status,
        _relation_by_one_based_serial(source_serial, card_cat, card_mask),
        torch.zeros_like(event_cat[..., 14]),
    )
    event_source = torch.where(
        is_switch,
        _relation_by_one_based_serial(event_cat[..., 16], card_cat, card_mask),
        event_source,
    )
    event_target = torch.where(
        is_switch,
        _relation_by_one_based_serial(event_cat[..., 17], card_cat, card_mask),
        torch.where(
            is_target_event,
            _relation_by_one_based_serial(event_cat[..., 15], card_cat, card_mask),
            torch.zeros_like(event_source),
        ),
    )
    event_before = torch.where(
        is_change | is_move_attached,
        _relation_by_one_based_serial(event_cat[..., 18], card_cat, card_mask),
        torch.zeros_like(event_source),
    )
    event_after = torch.where(
        is_change | is_move_attached,
        _relation_by_one_based_serial(event_cat[..., 19], card_cat, card_mask),
        torch.zeros_like(event_source),
    )
    event_source = torch.where(
        is_visible_draw | is_move_card,
        _relation_by_one_based_serial(event_cat[..., 14], card_cat, card_mask),
        event_source,
    )
    return (
        event_cat,
        event_num,
        event_state,
        event_mask,
        event_source,
        event_target,
        event_before,
        event_after,
    )


def policy_codec_v1_to_semantic0031_v2(
    batch: Mapping[str, Any],
    *,
    max_action_steps: int = 64,
) -> dict[str, Any]:
    """Project PolicyCodecV1 into the truthful subset of the 0031 contract.

    Exact current facts retain PRESENT state. Facts unavailable in the CUDA
    codec use UNKNOWN or a false sequence mask; zeros are never advertised as
    known historical facts.
    """

    import torch

    required = {
        "global_cat",
        "global_num",
        "entity_cat",
        "entity_num",
        "entity_parent",
        "entity_mask",
        "option_cat",
        "option_num",
        "option_mask",
        "min_count",
        "max_count",
        "semantic0031_resource_card_ids",
        "semantic0031_resource_initial",
        "semantic0031_resource_mask",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"PolicyCodecV1 batch is missing semantic0031 fields: {missing}")
    history_keys = _SEMANTIC_HISTORY_KEYS & set(batch)
    source_global_cat = batch["global_cat"]
    source_global_num = batch["global_num"]
    entity_cat = batch["entity_cat"]
    entity_num = batch["entity_num"]
    entity_parent = batch["entity_parent"]
    entity_mask = batch["entity_mask"].bool()
    source_option_cat = batch["option_cat"]
    source_option_num = batch["option_num"]
    option_mask = batch["option_mask"].bool()
    _require_shape("global_cat", source_global_cat, 2, _POLICY_GLOBAL_CAT_WIDTH)
    _require_shape("global_num", source_global_num, 2, _POLICY_GLOBAL_NUM_WIDTH)
    _require_shape("entity_cat", entity_cat, 3, _POLICY_ENTITY_CAT_WIDTH)
    _require_shape("entity_num", entity_num, 3, _POLICY_ENTITY_NUM_WIDTH)
    _require_shape("entity_parent", entity_parent, 2)
    _require_shape("entity_mask", entity_mask, 2)
    _require_shape("option_cat", source_option_cat, 3, _POLICY_OPTION_CAT_WIDTH)
    _require_shape("option_num", source_option_num, 3, _POLICY_OPTION_NUM_WIDTH)
    _require_shape("option_mask", option_mask, 2)
    batch_size, card_capacity = entity_mask.shape
    if entity_cat.shape[:2] != entity_mask.shape or entity_num.shape[:2] != entity_mask.shape:
        raise ValueError("PolicyCodecV1 entity tensors disagree")
    if entity_parent.shape != entity_mask.shape:
        raise ValueError("PolicyCodecV1 entity parent shape disagrees")
    if (
        source_option_cat.shape[:2] != option_mask.shape
        or source_option_num.shape[:2] != option_mask.shape
    ):
        raise ValueError("PolicyCodecV1 option tensors disagree")
    option_capacity = option_mask.shape[1]
    if option_capacity > 128:
        raise ValueError("semantic0031 supports at most 128 options")
    device = option_mask.device
    for name in required | history_keys:
        if batch[name].device != device:
            raise ValueError(f"batch field {name} is on the wrong device")

    global_cat = torch.zeros((batch_size, _GLOBAL_CAT_WIDTH), dtype=torch.long, device=device)
    global_cat[:, 0] = source_global_cat[:, 0].long().clamp(min=0, max=65)
    context = source_global_cat[:, 1].long()
    global_cat[:, 1] = torch.where(context.lt(130), context, torch.zeros_like(context))
    first = source_global_cat[:, 2].long()
    actor = source_global_cat[:, 3].long()
    global_cat[:, 2] = torch.where(
        first.gt(0) & actor.gt(0),
        torch.where(first.eq(actor), torch.ones_like(first), torch.full_like(first, 2)),
        torch.zeros_like(first),
    )
    flags = source_global_cat[:, 7].long()
    for target, bit in zip((3, 4, 5, 6), range(4)):
        global_cat[:, target] = flags.bitwise_and(1 << bit).ne(0).long() + 1
    source_zone = entity_cat[..., 2].long()
    source_status = entity_cat[..., 5].long().clamp(min=0, max=31)
    own_status = torch.where(
        entity_mask & source_zone.eq(1), source_status, torch.zeros_like(source_status)
    ).amax(dim=1)
    opponent_status = torch.where(
        entity_mask & source_zone.eq(6), source_status, torch.zeros_like(source_status)
    ).amax(dim=1)
    global_cat[:, 7] = own_status + 1
    global_cat[:, 8] = opponent_status + 1
    global_cat[:, 9] = 2  # Exact registered deck membership is known.
    global_cat[:, 10] = 1  # No deck order is asserted by PolicyCodecV1.
    global_cat[:, 11] = 0  # Hidden/visible looking semantics are not carried.

    global_num = torch.zeros((batch_size, _GLOBAL_NUM_WIDTH), dtype=torch.float32, device=device)
    global_state = torch.full(
        (batch_size, _GLOBAL_NUM_WIDTH),
        FIELD_UNKNOWN,
        dtype=torch.long,
        device=device,
    )

    def exact_global(target: int, source: int, scale: float) -> None:
        global_num[:, target] = torch.round(source_global_num[:, source].float() * scale)
        global_state[:, target] = FIELD_PRESENT

    for target, source, scale in (
        (0, 0, 20.0),
        (1, 1, 50.0),
        (2, 4, 60.0),
        (3, 5, 60.0),
        (4, 6, 20.0),
        (5, 7, 20.0),
        (6, 8, 6.0),
        (7, 9, 6.0),
        (8, 14, 5.0),
        (9, 15, 5.0),
        (10, 10, 128.0),
        (13, 11, 300.0),
        (14, 12, 10.0),
        (16, 7, 20.0),
        (19, 13, 60.0),
    ):
        exact_global(target, source, scale)
    global_num[:, 11] = batch["min_count"].float()
    global_num[:, 12] = batch["max_count"].float()
    global_state[:, 11:13] = FIELD_PRESENT
    global_num[:, 20] = (entity_mask & source_zone.eq(11)).sum(dim=1).float()
    global_state[:, 20] = FIELD_PRESENT

    zone_table = torch.tensor(_ZONE_MAP, dtype=torch.long, device=device)
    bounded_zone = source_zone.clamp(min=0, max=len(_ZONE_MAP) - 1)
    mapped_zone = zone_table[bounded_zone]
    supported_zone = source_zone.ge(0) & source_zone.lt(len(_ZONE_MAP))
    card_mask = entity_mask & supported_zone & mapped_zone.ne(0)
    raw_card_id = entity_cat[..., 0].long()
    supported_identity = raw_card_id.gt(0) & raw_card_id.le(_MAX_CARD_ID)
    card_id = torch.where(supported_identity, raw_card_id, torch.zeros_like(raw_card_id))
    card_cat = torch.zeros(
        (batch_size, card_capacity, _CARD_CAT_WIDTH),
        dtype=torch.long,
        device=device,
    )
    card_cat[..., 0] = card_id
    owner = entity_cat[..., 1].long().clamp(min=0, max=3)
    card_cat[..., 2] = torch.where(owner.eq(0) & card_mask, torch.full_like(owner, 3), owner)
    card_cat[..., 3] = mapped_zone
    card_cat[..., 4] = entity_cat[..., 3].long().clamp(min=0, max=256)
    kind = entity_cat[..., 4].long().clamp(min=0, max=10)
    card_cat[..., 5] = kind
    card_cat[..., 6] = source_status
    card_cat[..., 8] = torch.where(
        card_mask & supported_identity,
        torch.ones_like(raw_card_id),
        torch.zeros_like(raw_card_id),
    )
    card_cat *= card_mask.unsqueeze(-1)

    card_num = torch.zeros(
        (batch_size, card_capacity, _CARD_NUM_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    card_num[..., 0] = torch.round(entity_num[..., 0].float() * 400.0)
    card_num[..., 1] = torch.round(entity_num[..., 1].float() * 400.0)
    card_num[..., 2] = torch.round(entity_num[..., 3].float() * 10.0)
    card_num[..., 4] = torch.round(entity_num[..., 4].float() * 4.0)
    card_num[..., 5] = torch.round(entity_num[..., 5].float() * 4.0)
    card_num[..., 6] = entity_num[..., 6].float()
    card_num *= card_mask.unsqueeze(-1)
    pokemon = card_mask & kind.eq(2)
    card_state = torch.full(
        (batch_size, card_capacity, _CARD_NUM_WIDTH),
        FIELD_NOT_APPLICABLE,
        dtype=torch.long,
        device=device,
    )
    for field in (0, 1, 2, 4, 5, 6):
        card_state[..., field] = torch.where(
            pokemon,
            torch.full_like(card_state[..., field], FIELD_PRESENT),
            card_state[..., field],
        )
    card_state[..., 3] = torch.where(
        pokemon,
        torch.full_like(card_state[..., 3], FIELD_UNKNOWN),
        card_state[..., 3],
    )
    card_state = torch.where(card_mask.unsqueeze(-1), card_state, torch.zeros_like(card_state))
    parent_valid = entity_parent.ge(0) & entity_parent.lt(card_capacity)
    parent_one_based = entity_parent.long() + 1
    card_parent = torch.where(
        card_mask & parent_valid,
        parent_one_based,
        torch.zeros_like(parent_one_based),
    )

    resource_card_ids = _expand_static(
        batch["semantic0031_resource_card_ids"], batch_size, "resource_card_ids"
    ).long()
    resource_initial = _expand_static(
        batch["semantic0031_resource_initial"], batch_size, "resource_initial"
    ).float()
    resource_mask = _expand_static(
        batch["semantic0031_resource_mask"], batch_size, "resource_mask"
    ).bool()
    if (
        resource_card_ids.shape != resource_initial.shape
        or resource_card_ids.shape != resource_mask.shape
    ):
        raise ValueError("semantic0031 resource static fields disagree")
    resource_capacity = resource_mask.shape[1]
    resource_cat = torch.zeros(
        (batch_size, resource_capacity, _RESOURCE_CAT_WIDTH),
        dtype=torch.long,
        device=device,
    )
    resource_cat[..., 0] = resource_card_ids
    resource_cat[..., 1] = 6  # Unknown per-identity deck allocation.
    resource_cat[..., 2] = 6  # Unknown per-identity prize allocation.
    resource_cat[..., 3] = 1  # Explicitly not order-known.
    resource_cat *= resource_mask.unsqueeze(-1)
    resource_num = torch.zeros(
        (batch_size, resource_capacity, _RESOURCE_NUM_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    resource_num[..., 0] = resource_initial
    resource_state = torch.full(
        (batch_size, resource_capacity, _RESOURCE_NUM_WIDTH),
        FIELD_UNKNOWN,
        dtype=torch.long,
        device=device,
    )
    resource_state[..., 0] = FIELD_PRESENT
    resource_state = torch.where(
        resource_mask.unsqueeze(-1), resource_state, torch.zeros_like(resource_state)
    )

    (
        event_cat,
        event_num,
        event_state,
        event_mask,
        event_source,
        event_target,
        event_before,
        event_after,
    ) = _semantic0031_events_from_history(batch, source_global_cat, card_cat, card_mask)

    option_cat = torch.zeros(
        (batch_size, option_capacity, _OPTION_CAT_WIDTH),
        dtype=torch.long,
        device=device,
    )
    option_cat[..., 0] = source_option_cat[..., 0].long().clamp(min=0, max=65)
    option_cat[..., 1] = source_option_cat[..., 3].long().clamp(min=0, max=3)
    option_cat[..., 2] = source_option_cat[..., 1].long().clamp(min=0, max=33)
    target_relation = _one_based_entity_relation(source_option_cat[..., 9], card_mask)
    target_safe = (target_relation - 1).clamp(min=0, max=max(0, card_capacity - 1))
    target_owner = card_cat[..., 2].gather(1, target_safe)
    option_cat[..., 3] = torch.where(
        target_relation.gt(0), target_owner, torch.full_like(target_owner, 3)
    )
    option_cat[..., 4] = source_option_cat[..., 2].long().clamp(min=0, max=33)
    for target, source in ((5, 4), (6, 5)):
        identity = source_option_cat[..., source].long()
        option_cat[..., target] = torch.where(
            identity.le(_MAX_CARD_ID), identity.clamp_min(0), torch.zeros_like(identity)
        )
    attack_id = source_option_cat[..., 6].long()
    option_cat[..., 7] = torch.where(
        attack_id.le(_MAX_ATTACK_ID), attack_id.clamp_min(0), torch.zeros_like(attack_id)
    )
    option_cat[..., 9] = global_cat[:, 0].unsqueeze(1)
    option_cat[..., 10] = global_cat[:, 1].unsqueeze(1)
    option_cat[..., 13] = torch.arange(
        1, option_capacity + 1, dtype=torch.long, device=device
    ).view(1, -1)
    option_cat[..., 14] = source_option_cat[..., 10].long().clamp(min=0, max=256)
    option_cat[..., 15] = source_option_cat[..., 11].long().clamp(min=0, max=256)
    option_cat *= option_mask.unsqueeze(-1)

    option_num = torch.zeros(
        (batch_size, option_capacity, _OPTION_NUM_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    encoded_number = source_option_cat[..., 7].long()
    number_present = option_mask & encoded_number.gt(0)
    option_num[..., 0] = torch.where(
        number_present, (encoded_number - 1).float(), torch.zeros_like(encoded_number).float()
    )
    option_state = torch.full(
        (batch_size, option_capacity, _OPTION_NUM_WIDTH),
        FIELD_UNKNOWN,
        dtype=torch.long,
        device=device,
    )
    option_state[..., 0] = torch.where(
        number_present,
        torch.full_like(option_state[..., 0], FIELD_PRESENT),
        option_state[..., 0],
    )
    option_state = torch.where(
        option_mask.unsqueeze(-1), option_state, torch.zeros_like(option_state)
    )
    option_source = _one_based_entity_relation(source_option_cat[..., 8], card_mask)
    option_target = target_relation
    option_relation = torch.zeros((batch_size, option_capacity), dtype=torch.long, device=device)
    prototype_relation = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    prototype_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    minimum = batch["min_count"].long().view(batch_size).clamp(min=0, max=max_action_steps)
    maximum = batch["max_count"].long().view(batch_size).clamp(min=0, max=max_action_steps)
    maximum = torch.maximum(minimum, maximum)
    targets = torch.full((batch_size, 1), -100, dtype=torch.long, device=device)

    return {
        "global_cat": global_cat,
        "global_num": global_num,
        "global_state": global_state,
        "card_cat": card_cat,
        "card_num": card_num,
        "card_state": card_state,
        "card_parent": card_parent,
        "card_mask": card_mask,
        "resource_cat": resource_cat,
        "resource_num": resource_num,
        "resource_state": resource_state,
        "resource_mask": resource_mask,
        "event_cat": event_cat,
        "event_num": event_num,
        "event_state": event_state,
        "event_mask": event_mask,
        "event_source": event_source,
        "event_target": event_target,
        "event_before": event_before,
        "event_after": event_after,
        "option_cat": option_cat,
        "option_num": option_num,
        "option_state": option_state,
        "option_mask": option_mask,
        "option_source": option_source,
        "option_target": option_target,
        "option_context": option_relation,
        "option_effect_card": option_relation,
        "option_skill_id": prototype_relation,
        "option_skill_role": prototype_relation,
        "option_skill_parent": prototype_relation,
        "option_skill_mask": prototype_mask,
        "option_effect_id": prototype_relation,
        "option_effect_role": prototype_relation,
        "option_effect_parent": prototype_relation,
        "option_effect_mask": prototype_mask,
        "min_count": minimum,
        "max_count": maximum,
        "targets": targets,
    }


def semantic0031_v2_ready_batch(
    batch: Mapping[str, Any],
    *,
    max_action_steps: int = 64,
) -> dict[str, Any]:
    """Normalize already-materialized semantic0031 v2 CUDA tensors for the model contract."""

    import torch

    if not 1 <= max_action_steps <= 64:
        raise ValueError("semantic0031 max_action_steps must be in [1, 64]")
    missing = sorted(_SEMANTIC0031_V2_REQUIRED_KEYS - set(batch))
    if missing:
        raise KeyError(f"semantic0031 v2 batch is missing fields: {missing}")
    device = batch["option_mask"].device
    if device.type != "cuda":
        raise ValueError("semantic0031 v2 batch must stay on a CUDA device")
    for name in _SEMANTIC0031_V2_REQUIRED_KEYS:
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"semantic0031 v2 field {name} must be a torch.Tensor")
        if value.device != device:
            raise ValueError(f"semantic0031 v2 field {name} is on a different device")

    option_mask = batch["option_mask"]
    _require_shape("global_cat", batch["global_cat"], 2, _GLOBAL_CAT_WIDTH)
    _require_shape("global_num", batch["global_num"], 2, _GLOBAL_NUM_WIDTH)
    _require_shape("global_state", batch["global_state"], 2, _GLOBAL_NUM_WIDTH)
    _require_shape("card_cat", batch["card_cat"], 3, _CARD_CAT_WIDTH)
    _require_shape("card_num", batch["card_num"], 3, _CARD_NUM_WIDTH)
    _require_shape("card_state", batch["card_state"], 3, _CARD_NUM_WIDTH)
    _require_shape("resource_cat", batch["resource_cat"], 3, _RESOURCE_CAT_WIDTH)
    _require_shape("resource_num", batch["resource_num"], 3, _RESOURCE_NUM_WIDTH)
    _require_shape("resource_state", batch["resource_state"], 3, _RESOURCE_NUM_WIDTH)
    _require_shape("event_cat", batch["event_cat"], 3, _EVENT_CAT_WIDTH)
    _require_shape("event_num", batch["event_num"], 3, _EVENT_NUM_WIDTH)
    _require_shape("event_state", batch["event_state"], 3, _EVENT_NUM_WIDTH)
    _require_shape("option_cat", batch["option_cat"], 3, _OPTION_CAT_WIDTH)
    _require_shape("option_num", batch["option_num"], 3, _OPTION_NUM_WIDTH)
    _require_shape("option_state", batch["option_state"], 3, _OPTION_NUM_WIDTH)
    _require_shape("option_mask", option_mask, 2)
    batch_size = option_mask.shape[0]
    if batch["global_cat"].shape[0] != batch_size:
        raise ValueError("semantic0031 v2 global batch axis disagrees")

    sequence_fields = {
        "card_mask": batch["card_cat"].shape[:2],
        "card_parent": batch["card_cat"].shape[:2],
        "resource_mask": batch["resource_cat"].shape[:2],
        "event_mask": batch["event_cat"].shape[:2],
        "event_source": batch["event_cat"].shape[:2],
        "event_target": batch["event_cat"].shape[:2],
        "event_before": batch["event_cat"].shape[:2],
        "event_after": batch["event_cat"].shape[:2],
        "option_source": batch["option_cat"].shape[:2],
        "option_target": batch["option_cat"].shape[:2],
        "option_context": batch["option_cat"].shape[:2],
        "option_effect_card": batch["option_cat"].shape[:2],
        "option_skill_id": batch["option_skill_id"].shape[:2],
        "option_skill_role": batch["option_skill_id"].shape[:2],
        "option_skill_parent": batch["option_skill_id"].shape[:2],
        "option_skill_mask": batch["option_skill_id"].shape[:2],
        "option_effect_id": batch["option_effect_id"].shape[:2],
        "option_effect_role": batch["option_effect_id"].shape[:2],
        "option_effect_parent": batch["option_effect_id"].shape[:2],
        "option_effect_mask": batch["option_effect_id"].shape[:2],
    }
    for name, expected in sequence_fields.items():
        value = batch[name]
        if value.ndim != 2 or value.shape != expected:
            raise ValueError(
                f"semantic0031 v2 {name} shape {tuple(value.shape)} does not match {tuple(expected)}"
            )

    output: dict[str, Any] = {}
    for name in sorted(_SEMANTIC0031_V2_REQUIRED_KEYS):
        value = batch[name]
        if name in _SEMANTIC0031_V2_MASK_KEYS:
            output[name] = value.bool()
        elif name in _SEMANTIC0031_V2_FLOAT_KEYS:
            output[name] = value.float()
        elif name in _SEMANTIC0031_V2_LONG_KEYS:
            output[name] = value.long()
        else:
            raise KeyError(f"semantic0031 v2 field {name} has no dtype rule")

    minimum = output["min_count"].view(batch_size).clamp(min=0, max=max_action_steps)
    maximum = output["max_count"].view(batch_size).clamp(min=0, max=max_action_steps)
    output["min_count"] = minimum
    output["max_count"] = torch.maximum(minimum, maximum)
    targets = output["targets"]
    if targets.ndim == 1:
        targets = targets.view(batch_size, 1)
    if targets.ndim != 2 or targets.shape[0] != batch_size:
        raise ValueError("semantic0031 v2 targets must have shape [batch, action_steps]")
    output["targets"] = targets
    return output


def semantic0031_decode_device(
    action_decoder: Any,
    batch: Any,
    options: Any,
    state_summary: Any,
    *,
    max_select: int,
    greedy: bool,
    route_mask: Any | None = None,
    compute_stats: bool = True,
    sampling_seeds: Any | None = None,
    sampling_counters: Any | None = None,
) -> dict[str, Any]:
    """Fixed-shape ordered decode with actions and statistics kept on device."""

    import math
    import torch

    option_mask = batch.option_mask.bool()
    batch_size, option_count = option_mask.shape
    if route_mask is None:
        route_mask = torch.ones(batch_size, dtype=torch.bool, device=options.device)
    else:
        route_mask = route_mask.bool().view(batch_size)
    hidden = torch.tanh(action_decoder.initial(state_summary))
    option_keys = action_decoder.key(options)
    option_bias = action_decoder.option_bias(options).squeeze(-1)
    available = option_mask.clone()
    actions = torch.full((batch_size, max_select), -1, dtype=torch.long, device=options.device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=options.device)
    logprob = torch.zeros(batch_size, dtype=torch.float32, device=options.device)
    entropy = torch.zeros_like(logprob)
    maximum = batch.max_count.long().view(batch_size).clamp(min=0, max=max_select)
    minimum = batch.min_count.long().view(batch_size).clamp(min=0, max=max_select)
    active = route_mask & maximum.gt(0)
    if not greedy:
        if sampling_seeds is None or sampling_counters is None:
            raise ValueError("stochastic CUDA decode requires per-row seeds and counters")
        sampling_seeds = sampling_seeds.long().view(batch_size)
        sampling_counters = sampling_counters.long().view(batch_size)

    for step in range(max_select):
        pointer = (action_decoder.query(hidden).unsqueeze(1) * option_keys).sum(-1) / math.sqrt(
            options.shape[-1]
        )
        pointer = pointer + option_bias
        pointer = pointer.masked_fill(
            ~(available & active.unsqueeze(1)), torch.finfo(pointer.dtype).min
        )
        stop = action_decoder.stop(hidden).squeeze(-1)
        stop = stop.masked_fill(~(active & lengths.ge(minimum)), torch.finfo(stop.dtype).min)
        logits = torch.cat((pointer, stop.unsqueeze(1)), dim=1)
        logits = torch.where(route_mask[:, None], logits, torch.zeros_like(logits))
        distribution = None
        if not greedy or compute_stats:
            distribution = torch.distributions.Categorical(logits=logits.float())
        if greedy:
            choice = logits.argmax(dim=1)
        else:
            modulus = 2_147_483_647
            key = torch.remainder(sampling_seeds, modulus)
            key = torch.remainder(
                key + sampling_counters * 1_000_003 + (step + 1) * 9_176, modulus
            )
            for _ in range(3):
                key = torch.remainder(key * 48_271, modulus)
            uniform = (key.to(torch.float64) + 0.5) / modulus
            cumulative = torch.softmax(logits.float(), dim=1).double().cumsum(dim=1)
            choice = cumulative.ge(uniform.unsqueeze(1)).long().argmax(dim=1)
        if compute_stats:
            logprob += torch.where(active, distribution.log_prob(choice), 0.0)
            entropy += torch.where(active, distribution.entropy(), 0.0)
        chosen_valid = active & choice.lt(option_count)
        stopped |= active & choice.eq(option_count)
        safe = choice.clamp(min=0, max=max(0, option_count - 1))
        actions[:, step] = torch.where(chosen_valid, safe, torch.full_like(safe, -1))
        chosen_mask = torch.nn.functional.one_hot(safe, num_classes=option_count).bool()
        available = available & ~(chosen_mask & chosen_valid.unsqueeze(1))
        selected = options.gather(1, safe.view(-1, 1, 1).expand(-1, 1, options.shape[-1])).squeeze(
            1
        )
        candidate_hidden = action_decoder.recurrent(selected, hidden)
        hidden = torch.where(chosen_valid.unsqueeze(-1), candidate_hidden, hidden)
        lengths = lengths + chosen_valid.long()
        active = chosen_valid & lengths.lt(maximum) & route_mask
    return {
        "actions": actions,
        "lengths": lengths,
        "stopped": stopped,
        "logprob": logprob,
        "entropy": entropy,
    }


def semantic0031_greedy_decode_device(
    action_decoder: Any,
    batch: Any,
    options: Any,
    state_summary: Any,
    *,
    max_select: int,
    route_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Compatibility wrapper for fixed-shape ordered greedy decoding."""

    decoded = semantic0031_decode_device(
        action_decoder,
        batch,
        options,
        state_summary,
        max_select=max_select,
        greedy=True,
        route_mask=route_mask,
        compute_stats=False,
    )
    return decoded["actions"], decoded["lengths"]


def semantic0031_mean_pool_by_parent_device(
    values: Any,
    one_based_parent: Any,
    mask: Any,
    parent_count: int,
) -> Any:
    """Match the canonical explicit-relation pooling without prototype inference."""

    batch, _, width = values.shape
    output = values.new_zeros((batch, parent_count, width))
    counts = values.new_zeros((batch, parent_count, 1))
    valid = mask & one_based_parent.gt(0) & one_based_parent.le(parent_count)
    index = (one_based_parent - 1).clamp(min=0, max=max(0, parent_count - 1))
    output.scatter_add_(
        1,
        index.unsqueeze(-1).expand(-1, -1, width),
        values * valid.unsqueeze(-1),
    )
    counts.scatter_add_(1, index.unsqueeze(-1), valid.unsqueeze(-1).to(values.dtype))
    return output / counts.clamp_min(1.0)


class Semantic0031DeviceAdapter:
    """GPU-resident adapter for the compact 0031 semantic policy."""

    outputs_normalized = True

    def __init__(
        self,
        model: Any,
        registered_deck: Sequence[int],
        *,
        max_select: int = 64,
    ) -> None:
        import torch

        if not 1 <= max_select <= 64:
            raise ValueError("semantic0031 max_select must be in [1, 64]")
        parameters = tuple(model.parameters())
        if not parameters:
            raise ValueError("semantic0031 model has no parameters")
        devices = {parameter.device for parameter in parameters}
        dtypes = {parameter.dtype for parameter in parameters if parameter.is_floating_point()}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("semantic0031 model parameters disagree on device/dtype")
        self.model = model
        self.device = next(iter(devices))
        if self.device.type != "cuda":
            raise ValueError("Semantic0031DeviceAdapter requires a CUDA model")
        if dtypes != {torch.float32}:
            raise ValueError("semantic0031 runtime model must remain FP32")
        self.max_select = max_select
        self._static_fields = semantic0031_static_fields(registered_deck, self.device)
        with torch.inference_mode():
            self.prototype_memory = model.prototype_encoder.encode_all()

    @property
    def shared_batch_key(self) -> tuple[str, int, int]:
        return (SEMANTIC0031_SCHEMA, id(self.model), self.max_select)

    @property
    def shared_static_fields(self) -> Mapping[str, Any]:
        return self._static_fields

    def _encode_option_inputs(self, batch: Any, state: Any) -> Any:
        option_encoder = self.model.option_encoder
        option_cat = batch.option_cat
        options = option_encoder.categorical(option_cat)
        options = options + option_encoder.numeric(batch.option_num, batch.option_state)
        for card_column in (5, 6, 11, 12):
            options = options + self.prototype_memory.cards[option_cat[..., card_column]]
        options = options + self.prototype_memory.attacks[option_cat[..., 7]]
        options = options + option_encoder.source_relation(state.gather_cards(batch.option_source))
        options = options + option_encoder.target_relation(state.gather_cards(batch.option_target))
        options = options + option_encoder.context_relation(
            state.gather_cards(batch.option_context)
        )
        options = options + option_encoder.effect_card_relation(
            state.gather_cards(batch.option_effect_card)
        )

        skill_tokens = (
            self.prototype_memory.skills[batch.option_skill_id]
            + option_encoder.skill_role(batch.option_skill_role)
        )
        skill_context = semantic0031_mean_pool_by_parent_device(
            skill_tokens,
            batch.option_skill_parent,
            batch.option_skill_mask,
            batch.option_count,
        )
        effect_tokens = (
            self.prototype_memory.effects[batch.option_effect_id]
            + option_encoder.effect_role(batch.option_effect_role)
        )
        effect_context = semantic0031_mean_pool_by_parent_device(
            effect_tokens,
            batch.option_effect_parent,
            batch.option_effect_mask,
            batch.option_count,
        )
        options = option_encoder.input_norm(
            options
            + option_encoder.skill_relation(skill_context)
            + option_encoder.effect_relation(effect_context)
        )
        return options

    def _encode_options(self, batch: Any, state: Any) -> Any:
        option_encoder = self.model.option_encoder
        options = self._encode_option_inputs(batch, state)
        encoded = option_encoder.cross_attention_transformer(
            tgt=options,
            memory=state.tokens,
            tgt_key_padding_mask=~batch.option_mask,
            memory_key_padding_mask=~state.mask,
        )
        return encoded * batch.option_mask.unsqueeze(-1)

    def _inject_static(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        option_mask = batch["option_mask"]
        output = dict(batch)
        for name, value in self._static_fields.items():
            if value.device != option_mask.device:
                raise ValueError(f"semantic0031 static field {name} is on the wrong device")
            output[name] = value.expand(option_mask.shape[0], *value.shape[1:])
        return output

    def act_device(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        return self.act_device_shared_static(self._inject_static(batch))

    def act_device_shared_static(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        import torch

        if batch["option_mask"].device != self.device:
            raise ValueError("semantic0031 batch and model must share one CUDA device")
        with torch.inference_mode():
            semantic = policy_codec_v1_to_semantic0031_v2(batch, max_action_steps=self.max_select)
            validated = self.model.validate_batch(semantic)
            state = self.model.state_encoder(validated, self.prototype_memory)
            options = self._encode_options(validated, state)
            return semantic0031_greedy_decode_device(
                self.model.action_decoder,
                validated,
                options,
                state.summary,
                max_select=self.max_select,
                route_mask=batch.get("_route_mask"),
            )

    def act_device_semantic0031_v2(
        self,
        batch: Mapping[str, Any],
        *,
        route_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        import torch

        if batch["option_mask"].device != self.device:
            raise ValueError("semantic0031 v2 batch and model must share one CUDA device")
        if route_mask is None:
            route_mask = batch.get("_route_mask")
        with torch.inference_mode():
            semantic = semantic0031_v2_ready_batch(batch, max_action_steps=self.max_select)
            validated = self.model.validate_batch(semantic)
            state = self.model.state_encoder(validated, self.prototype_memory)
            options = self._encode_options(validated, state)
            return semantic0031_greedy_decode_device(
                self.model.action_decoder,
                validated,
                options,
                state.summary,
                max_select=self.max_select,
                route_mask=route_mask,
            )

    def act_ready_lanes_semantic0031_v2(
        self,
        engine: Any,
        lane_indices: Any,
        *,
        route_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        batch = engine.encode_semantic0031_v2_lanes(lane_indices.contiguous())
        return self.act_device_semantic0031_v2(batch, route_mask=route_mask)


__all__ = [
    "KNOWN_0031_ARCHIVE_SHA256",
    "SEMANTIC0031_CHECKPOINT_SCHEMA",
    "SEMANTIC0031_PACKAGE_SCHEMA",
    "SEMANTIC0031_SCHEMA",
    "Semantic0031DeviceAdapter",
    "Semantic0031Package",
    "inspect_semantic0031_archive",
    "load_semantic0031_package",
    "policy_codec_v1_to_semantic0031_v2",
    "semantic0031_decode_device",
    "semantic0031_greedy_decode_device",
    "semantic0031_static_fields",
    "semantic0031_v2_ready_batch",
]
