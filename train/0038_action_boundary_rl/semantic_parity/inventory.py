"""Immutable path, weight, and runtime inventory for the 0038 parity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import torch

from ..checkpoint import validate_model_only_payload
from ..action_boundary.contracts import (
    ACTION_BOUNDARY_SCHEMA_VERSION,
    CANONICALIZER_VERSION,
    DECISION_GATE_VERSION,
    FEATURE_PREPROCESSING_VERSION,
    OFFICIAL_PROTOCOL_ADAPTER_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
)
from ..policy.actor_critic import load_actor_critic
from ..policy.adaptation import AdaptationConfig
from ..integrated.config import IntegratedFlags
from ..training.storage_full_semantic import _checkpoint_tensor


ROOT = Path(__file__).resolve().parents[3]
BASE_CHECKPOINT = (
    ROOT / "rl_runs/0037_dragapult_value_initialized_rl/source/"
    "friend_0806_epoch11/model.pt"
)
VALUE_CHECKPOINT = (
    ROOT / "rl_runs/0037_dragapult_value_initialized_rl/source/"
    "pre_rl_value/model.pt"
)
FOCAL_DECK = (
    ROOT / "train/0038_action_boundary_rl/league/decks/"
    "dragapult_ex_07bedfffbfad/deck.csv"
)
CUDA_RULES = ROOT / ".tmp/cuda_0032_rules/official_rules.bin"
CUDA_EXTENSION = (
    ROOT / ".tmp/engine_cuda_benchmark/build_sm120_staged/_ptcg_cuda.so"
)
OFFICIAL_CPU_LIBRARY = ROOT / "engine/build/seeded_official/0002/libcg.so"
CARD_DATABASE = ROOT / "data/official/EN_Card_Data.csv"
ARCHIVED_U230_INVENTORY_FIXTURE = (
    ROOT
    / "train/0038_action_boundary_rl/tests/fixtures/semantic_parity_v1/"
    "u230_package_load_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(rows: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    seen = 0
    for name, value in sorted(rows):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        seen += 1
    if seen == 0:
        raise ValueError("component tensor set is empty")
    return digest.hexdigest()


def _deck() -> tuple[int, ...]:
    cards = tuple(int(line) for line in FOCAL_DECK.read_text().splitlines() if line.strip())
    if len(cards) != 60:
        raise ValueError("0038 focal deck is not exactly 60 cards")
    return cards


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout)


def _path_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _source_has_silent_macro_fallback(package_root: Path) -> bool:
    source = (
        package_root / "strategy/deployment/compound_inference.py"
    ).read_text(encoding="utf-8")
    # Current fallback clears an invalid transaction and immediately performs a
    # fresh full-policy decision on an internal callback.  The audit treats this
    # as a semantic fallback even though it is restricted to MacroProtocolError.
    if "except MacroProtocolError:" not in source:
        return False
    handler = source.split("except MacroProtocolError:", 1)[1].split("else:", 1)[0]
    return "raise" not in handler and "self.pending = None" in handler


def _strict_package_startup(package_root: Path) -> dict[str, Any]:
    code = """
import json, runpy
namespace = runpy.run_path('main.py')
policy = namespace['POLICY']
print(json.dumps({
    'checkpoint_update': int(policy.metadata['checkpoint_update']),
    'actor_tensors': len(policy.actor.state_dict()),
    'value_tensors': len(policy.value_head.state_dict()) if hasattr(policy, 'value_head') else 0,
    'allocation_tensors': len(policy.allocation_head.state_dict()),
    'meta_head_tensors': len(policy.meta_head.state_dict()),
    'meta_conditioner_tensors': len(policy.meta_conditioner.state_dict()),
    'model_eval': not policy.actor.training,
    'value_eval': (not policy.value_head.training) if hasattr(policy, 'value_head') else False,
    'deck_cards': len(namespace['read_deck_csv']()),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("strict package startup emitted no load report")
    return json.loads(lines[-1])


def load_archived_runtime_inventory_fixture(
    fixture: Path = ARCHIVED_U230_INVENTORY_FIXTURE,
) -> dict[str, Any]:
    """Load a hash-validated inventory captured before package retirement."""

    from .freeze_regression_fixtures import validate_fixtures

    fixture = fixture.resolve()
    manifest = validate_fixtures(fixture.parent)
    registered = {entry["path"] for entry in manifest["entries"]}
    if fixture.name not in registered:
        raise ValueError("runtime inventory fixture is not registered in its manifest")
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "0038_u230_package_load_prefx_fixture_v1":
        raise ValueError("archived runtime inventory fixture schema mismatch")
    report = payload.get("load_report")
    if not isinstance(report, dict):
        raise ValueError("archived runtime inventory fixture has no load report")
    required_sections = {"checkpoint", "contracts", "package", "runtime"}
    if not required_sections.issubset(report):
        raise ValueError("archived runtime inventory load report is incomplete")
    source_manifest_sha256 = manifest["sources"]["package_manifest_sha256"]
    if report["package"].get("manifest_sha256") != source_manifest_sha256:
        raise ValueError("archived package manifest identity mismatch")
    return report


def build_runtime_inventory(*, checkpoint: Path, package_root: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    package_root = package_root.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_model_only_payload(payload)
    state: dict[str, torch.Tensor] = payload["state_dict"]
    adaptation = AdaptationConfig(**payload["adaptation"])
    flags = IntegratedFlags(**payload["integrated_flags"])

    model, identity = load_actor_critic(
        BASE_CHECKPOINT, _deck(), "cpu", adaptation=adaptation,
        integrated_flags=flags,
    )
    expected = {
        name for name in model.state_dict()
        if _checkpoint_tensor(model, name)
    }
    actual = set(state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    base_payload = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    base_state: dict[str, torch.Tensor] = base_payload["state_dict"]
    package_model = torch.load(
        package_root / "strategy/model.bin", map_location="cpu", weights_only=True
    )
    package_manifest = json.loads(
        (package_root / "manifest.json").read_text(encoding="utf-8")
    )
    portable_metadata = package_model.get("metadata") or {}
    portable_required_top_level = {
        "schema_version", "actor_state_dict", "value_head_state_dict",
        "allocation_head_state_dict",
        "opponent_meta_head_state_dict", "opponent_meta_conditioner_state_dict",
        "metadata",
    }
    portable_legacy_top_level = portable_required_top_level - {
        "value_head_state_dict"
    }
    portable_top_level = set(package_model)
    if portable_top_level not in (
        portable_required_top_level, portable_legacy_top_level
    ):
        raise ValueError("portable package top-level state inventory mismatch")
    deck = [
        int(line) for line in (package_root / "deck.csv").read_text().splitlines()
        if line.strip()
    ]
    # Exercise the formal package entrypoint in an isolated process.  This also
    # validates the immutable file inventory before every strict state load.
    package_load_report = _strict_package_startup(package_root)

    component_hashes = {
        "base_encoder": _tensor_state_sha256(
            (name, value) for name, value in base_state.items()
            if name.startswith(("prototype_encoder.", "state_encoder."))
        ),
        "option_encoder_lora": _tensor_state_sha256(
            (name, value) for name, value in state.items()
            if ".parametrizations." in name
        ),
        "action_decoder": _tensor_state_sha256(
            (name, value) for name, value in state.items()
            if name.startswith("actor.action_decoder.")
        ),
        "value": _tensor_state_sha256(
            (name, value) for name, value in state.items()
            if name.startswith("value_head.")
        ),
        "allocation_head": _tensor_state_sha256(
            (name, value) for name, value in state.items()
            if name.startswith("allocation_head.")
        ),
    }

    metadata = payload.get("metadata") or {}
    portable_component_hashes = {
        "actor": _tensor_state_sha256(package_model["actor_state_dict"].items()),
        "allocation_head": _tensor_state_sha256(
            package_model["allocation_head_state_dict"].items()
        ),
        "opponent_meta_head": _tensor_state_sha256(
            package_model["opponent_meta_head_state_dict"].items()
        ),
        "opponent_meta_conditioner": _tensor_state_sha256(
            package_model["opponent_meta_conditioner_state_dict"].items()
        ),
    }
    if "value_head_state_dict" in package_model:
        portable_component_hashes["value"] = _tensor_state_sha256(
            package_model["value_head_state_dict"].items()
        )
    return {
        "schema_version": "0038_semantic_parity_runtime_inventory_v1",
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "paths": {
            "cuda_rollout_engine_entry": "train/0038_action_boundary_rl/rollout/cuda_collector.py:CudaFullSemanticRolloutCollector.collect",
            "cuda_agent_entry": "engine_cuda/python/ptcg_cuda_engine/semantic0031_resident.py:run_resident_greedy_jobs",
            "cuda_model_entry": "train/0038_action_boundary_rl/training/run_full_semantic.py:build_collector",
            "frozen_schedule_entry": "train/0038_action_boundary_rl/evaluation/frozen_jobs.py:build_frozen_jobs",
            "package_build_entry": "train/0038_action_boundary_rl/export_full_semantic_candidate.py:export_candidate",
            "package_agent_entry": "main.py:agent",
            "decision_gate": "train/0038_action_boundary_rl/action_boundary/decision_gate.py:DecisionGate",
            "cuda_action_boundary": "train/0038_action_boundary_rl/rollout/cuda_action_boundary.py:CudaActionBoundaryAdapter",
            "package_action_boundary": "strategy/deployment/compound_inference.py:PortableCompoundSemanticPolicy",
            "macro_planner": "train/0038_action_boundary_rl/action_boundary/macro_planner.py:MacroPlanner",
            "official_pending_transaction": "train/0038_action_boundary_rl/action_boundary/macro_protocol.py:PendingMacroTransaction",
            "official_observe_only": "train/0038_action_boundary_rl/rollout/worker_compiler.py:WorkerLocalCompiler.observe_only",
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": _sha256(checkpoint),
            "schema_version": payload["schema_version"],
            "actor_schema": payload["actor_schema"],
            "update": int(payload["update"]),
            "source_actor_sha256": identity.checkpoint_sha256,
            "source_value_sha256": metadata.get("source_value_sha256"),
            "component_sha256": component_hashes,
            "critical_missing_keys": missing,
            "critical_unexpected_keys": unexpected,
            "state_tensor_count": len(state),
            "adaptation": payload["adaptation"],
            "integrated_flags": payload["integrated_flags"],
            "training_feature_preprocessing_version": metadata.get(
                "feature_preprocessing_version"
            ),
            "training_distribution_compatible": (
                metadata.get("feature_preprocessing_version")
                == FEATURE_PREPROCESSING_VERSION
            ),
        },
        "contracts": {
            "action_boundary": ACTION_BOUNDARY_SCHEMA_VERSION,
            "decision_gate": DECISION_GATE_VERSION,
            "canonicalizer": CANONICALIZER_VERSION,
            "trajectory": TRAJECTORY_SCHEMA_VERSION,
            "official_protocol_adapter": OFFICIAL_PROTOCOL_ADAPTER_VERSION,
            "feature_preprocessing": FEATURE_PREPROCESSING_VERSION,
            "source_checkpoint": {
                "action_boundary": metadata.get("action_schema_version"),
                "decision_gate": metadata.get("decision_gate_version"),
                "canonicalizer": metadata.get("canonicalizer_version"),
                "trajectory": metadata.get("trajectory_schema_version"),
                "official_protocol_adapter": metadata.get(
                    "official_protocol_adapter_version"
                ),
            },
        },
        "package": {
            "root": str(package_root.relative_to(ROOT)),
            "manifest_sha256": _sha256(package_root / "manifest.json"),
            "model_sha256": _sha256(package_root / "strategy/model.bin"),
            "portable_schema": package_model["schema_version"],
            "action_boundary": portable_metadata.get("action_schema_version"),
            "decision_gate": portable_metadata.get("decision_gate_version"),
            "canonicalizer": portable_metadata.get("canonicalizer_version"),
            "official_protocol_adapter": portable_metadata.get(
                "official_protocol_adapter_version"
            ),
            "checkpoint_update": portable_metadata.get("checkpoint_update"),
            "strict_load": (
                portable_top_level in (
                    portable_required_top_level, portable_legacy_top_level
                )
                and not missing and not unexpected
                and package_load_report["checkpoint_update"] == int(payload["update"])
                and package_load_report["model_eval"]
            ),
            "strict_load_report": package_load_report,
            "silent_legacy_fallback": _source_has_silent_macro_fallback(package_root),
            "model_eval": ".eval()" in (
                package_root / "strategy/deployment/compound_inference.py"
            ).read_text(encoding="utf-8"),
            "critic_deployed": bool(package_manifest.get("critic_deployed")),
            "storage_dtype": package_manifest.get("storage_dtype"),
            "runtime_dtype": package_manifest.get("runtime_dtype"),
            "deck_card_count": len(deck),
            "deck_sha256": package_manifest.get("deck_sha256"),
            "top_level_state_keys": sorted(package_model),
            "component_sha256": portable_component_hashes,
        },
        "runtime": {
            "cuda_rules_sha256": _sha256(CUDA_RULES),
            "cuda_extension_sha256": _sha256(CUDA_EXTENSION),
            "official_cpu_library_sha256": _sha256(OFFICIAL_CPU_LIBRARY),
            "card_database_sha256": _sha256(CARD_DATABASE),
            "cuda_rules": _path_record(CUDA_RULES),
            "cuda_extension": _path_record(CUDA_EXTENSION),
            "official_cpu_library": _path_record(OFFICIAL_CPU_LIBRARY),
            "card_database": _path_record(CARD_DATABASE),
            "cuda_execution": "one official primitive select per callback; macro only caches target plan",
            "package_lifecycle": "one module-global POLICY per worker process; reset on select=null; pending retained across callbacks",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_runtime_inventory(
        checkpoint=args.checkpoint, package_root=args.package
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVED_U230_INVENTORY_FIXTURE",
    "build_runtime_inventory",
    "load_archived_runtime_inventory_fixture",
]
