"""V16: reconstructable U105 continuation with doubled StateEncoder LoRA LR."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..policy.actor_critic import (
    DEFAULT_0814_ACTOR_CHECKPOINT,
    DEFAULT_0814_VALUE_CHECKPOINT,
)
from ..policy_identity import materialize_policy_bundle
from . import run_v13_policy0814_shared_encoder as contract
from .checkpointing import (
    atomic_save_complete_delta,
    audit_reconstruction,
    load_complete_delta,
    trainable_parameter_names,
)
from .lr_profiles import LearningRateProfile
from .run_v1 import PROJECT_ROOT, ROOT, RULES, _checkpoint, run


VERSION = "V16_policy0814_u105_complete_delta_state_lora_recovery"
VERSION_ROOT = ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions" / VERSION
SOURCE_ROOT = VERSION_ROOT / "source"
PARENT_VERSION = "V14_policy0814_identity_gate_fix"
PARENT_UPDATE = 105
PARENT_ROOT = ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions" / PARENT_VERSION
LEGACY_PARENT = PARENT_ROOT / f"checkpoint/update-{PARENT_UPDATE:06d}.pt"
PARENT_STATUS = PARENT_ROOT / "artifact/status.json"
EXPECTED_PARENT_SHA256 = "1df883f14ee026025b98ac519d25442e9497e59fda513d55663889eaa620a5a3"
STATE_LORA_INITIALIZATION_SEED = 4_516_105
RECONSTRUCTION_SEED = 4_516_106
REFERENCE_INITIALIZATION_SEED = 4_516_000
WANDB_RUN_ID = "0045-v16-policy0814-u105-complete-delta-recovery"

LEARNING_RATE_PROFILE = LearningRateProfile(
    profile_id="0045_v16_state_encoder_lora_recovery_2x_v1",
    base_profile_id="0045_expert_cold_start_lr_v1",
    scale=None,
    scope="v16_shared_encoder_lora_only",
    default_after_run="0045_expert_cold_start_lr_v1",
    decoder_learning_rate=1.0e-5,
    allocation_learning_rate=1.0e-5,
    option_lora_learning_rate=2.0e-5,
    shared_encoder_learning_rate=4.0e-5,
    value_learning_rate=2.0e-5,
    prize_learning_rate=2.0e-5,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _seeded_model(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return contract._model_and_audit()


def _state_lora_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(sorted(
        name for name in trainable_parameter_names(model)
        if name.startswith("actor.state_encoder.")
        and ".parametrizations." in name
    ))


def readiness(*, require_pristine: bool = True) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = registry.validate_all()
    required = (
        RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so", LEGACY_PARENT, PARENT_STATUS,
        DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V16 launch artifacts missing: {missing}")
    if sha256_file(LEGACY_PARENT) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("V16 parent is not exact V14 U105")
    status = json.loads(PARENT_STATUS.read_text(encoding="utf-8"))
    if (
        status.get("state") != "stopped_by_user"
        or status.get("checkpoint_update") != PARENT_UPDATE
        or status.get("wandb", {}).get("state") != "synced"
    ):
        raise RuntimeError("V16 requires stopped and synced V14 U105 provenance")
    if require_pristine and VERSION_ROOT.exists() and any(VERSION_ROOT.rglob("*")):
        raise FileExistsError(f"V16 formal version path is already used: {VERSION_ROOT}")
    bundle = materialize_policy_bundle(
        PROJECT_ROOT, contract.OPPONENT_POLICY_ID, purpose="0045_v16_readiness"
    )
    model, source, trainable = _seeded_model(STATE_LORA_INITIALIZATION_SEED)
    state_lora = _state_lora_names(model)
    if len(state_lora) != 16:
        raise RuntimeError("V16 StateEncoder LoRA tensor boundary changed")
    return {
        "schema_version": "0045_v16_complete_delta_u105_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "update": PARENT_UPDATE,
            "checkpoint": str(LEGACY_PARENT.relative_to(ROOT)),
            "checkpoint_sha256": EXPECTED_PARENT_SHA256,
            "checkpoint_schema": "0045_minimal_lora_model_only_v1",
            "known_missing_state_encoder_lora_tensors": 16,
        },
        "state_encoder_lora_reinitialization": {
            "seed": STATE_LORA_INITIALIZATION_SEED,
            "tensor_count": len(state_lora),
            "parameter_count": sum(
                dict(model.named_parameters())[name].numel() for name in state_lora
            ),
            "learning_rate": LEARNING_RATE_PROFILE.shared_encoder_learning_rate,
            "semantics": "new_zero_effective_lora_not_recovered_v14_weights",
        },
        "source": asdict(source),
        "trainable": trainable,
        "learning_rate_profile": LEARNING_RATE_PROFILE.metadata(),
        "opponent": {
            "policy_id": contract.OPPONENT_POLICY_ID,
            "effective_policy_sha256": bundle.audit.effective_policy_sha256,
        },
        "rollout": {
            "games": contract.ROLLOUT_GAMES,
            "lanes": contract.ROLLOUT_GAMES,
            "fixed_deck_quotas": contract.FIXED_DECK_QUOTAS,
            "random_remainder_deck_ids": list(contract.RANDOM_DECK_IDS),
        },
        "eval": {
            "initial_checkpoint": PARENT_UPDATE,
            "every_updates": 5,
            "games": 512,
            "profile": "policy0814_exact_deck_cuda512",
        },
        "checkpoint_contract": {
            "schema": "0045_complete_delta_model_only_v2",
            "required_trainable_tensors": trainable["tensor_count"],
            "full_reconstruction_hash_required": True,
            "checkpoint_retention": "all",
        },
        "asset_audit": asdict(asset_audit),
        "wandb": {
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    }


def materialize_sources(output_root: Path = SOURCE_ROOT) -> dict[str, Path]:
    if output_root.exists() and any(output_root.rglob("*")):
        raise FileExistsError(f"V16 source materialization is already used: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    model, _, trainable_audit = _seeded_model(STATE_LORA_INITIALIZATION_SEED)
    base_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    legacy = torch.load(LEGACY_PARENT, map_location="cpu", weights_only=True)
    if legacy.get("schema_version") != "0045_minimal_lora_model_only_v1":
        raise RuntimeError("V16 parent must remain the historical V14 schema")
    trainable_names = set(trainable_parameter_names(model))
    legacy_names = set(legacy["state_dict"])
    missing_trainable = sorted(trainable_names - legacy_names)
    state_lora = list(_state_lora_names(model))
    if missing_trainable != state_lora:
        raise RuntimeError("V14 missing trainable boundary is not exact StateEncoder LoRA")
    extra_non_trainable = sorted(legacy_names - trainable_names)
    changed_extra = [
        name for name in extra_non_trainable
        if name not in base_state
        or not torch.equal(base_state[name], legacy["state_dict"][name])
    ]
    if changed_extra:
        raise RuntimeError(f"V14 has changed non-trainable delta tensors: {changed_extra}")
    incompatible = model.load_state_dict(legacy["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"V14 U105 has unexpected tensors: {incompatible.unexpected_keys}")
    parameters = dict(model.named_parameters())
    b_names = [name for name in state_lora if name.rsplit(".", 1)[-1].endswith("b")]
    all_b_zero = bool(b_names) and all(
        torch.count_nonzero(parameters[name]).item() == 0 for name in b_names
    )
    if not all_b_zero:
        raise RuntimeError("new StateEncoder LoRA must start zero-effective")

    parent_payload = _checkpoint(
        model, PARENT_UPDATE, version=VERSION,
        focal_deck_id=contract.FOCAL_DECK_ID,
        focal_deck_ids=(contract.FOCAL_DECK_ID,), focal_schedule_mode="fixed",
        source_parent_version=PARENT_VERSION, source_parent_update=PARENT_UPDATE,
        parent_policy_id=contract.OPPONENT_POLICY_ID, parent_policy_update=None,
    )
    parent_path = output_root / f"complete-parent-update-{PARENT_UPDATE:06d}.pt"
    atomic_save_complete_delta(parent_path, parent_payload, model)
    reconstructed, _, _ = _seeded_model(RECONSTRUCTION_SEED)
    load_complete_delta(reconstructed, parent_payload)
    reconstruction = audit_reconstruction(model, reconstructed, parent_payload)

    reference_model, _, reference_trainable = _seeded_model(
        REFERENCE_INITIALIZATION_SEED
    )
    reference_payload = _checkpoint(
        reference_model, 0, version=VERSION,
        focal_deck_id=contract.FOCAL_DECK_ID,
        focal_deck_ids=(contract.FOCAL_DECK_ID,), focal_schedule_mode="fixed",
        source_parent_version="Policy-0814-pretrained", source_parent_update=None,
        parent_policy_id=contract.OPPONENT_POLICY_ID, parent_policy_update=None,
    )
    reference_path = output_root / "reference-policy0814-complete-u000000.pt"
    atomic_save_complete_delta(reference_path, reference_payload, reference_model)
    reference_reconstructed, _, _ = _seeded_model(RECONSTRUCTION_SEED + 1)
    load_complete_delta(reference_reconstructed, reference_payload)
    reference_reconstruction = audit_reconstruction(
        reference_model, reference_reconstructed, reference_payload
    )

    audit = {
        "schema_version": "0045_v16_u105_handoff_audit_v1",
        "status": "PASS",
        "legacy_parent": {
            "path": str(LEGACY_PARENT.relative_to(ROOT)),
            "sha256": EXPECTED_PARENT_SHA256,
            "preserved_trainable_tensor_count": len(trainable_names & legacy_names),
            "missing_trainable_tensor_count": len(missing_trainable),
            "missing_trainable_tensor_names": missing_trainable,
            "unchanged_non_trainable_tensor_count": len(extra_non_trainable),
        },
        "state_encoder_lora_initialization": {
            "seed": STATE_LORA_INITIALIZATION_SEED,
            "tensor_count": len(state_lora),
            "tensor_names": state_lora,
            "all_b_tensors_zero": all_b_zero,
            "learning_rate": LEARNING_RATE_PROFILE.shared_encoder_learning_rate,
        },
        "complete_parent": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "trainable_tensor_count": len(parent_payload["state_dict"]),
            "full_state_sha256": parent_payload["checkpoint_integrity"][
                "source_full_state_sha256"
            ],
            "reconstruction_status": reconstruction["status"],
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "trainable_tensor_count": reference_trainable["tensor_count"],
            "full_state_sha256": reference_payload["checkpoint_integrity"][
                "source_full_state_sha256"
            ],
            "reconstruction_status": reference_reconstruction["status"],
        },
        "trainable": trainable_audit,
        "learning_rate_profile": LEARNING_RATE_PROFILE.metadata(),
    }
    audit_path = output_root / "handoff_audit.json"
    _atomic_json(audit_path, audit)
    return {
        "parent_checkpoint": parent_path,
        "reference_checkpoint": reference_path,
        "audit": audit_path,
    }


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    readiness(require_pristine=True)
    sources = materialize_sources(SOURCE_ROOT)
    run(
        updates=updates, wandb_mode=wandb_mode, launch_formal=True,
        version=VERSION, start_update=PARENT_UPDATE,
        parent_checkpoint=sources["parent_checkpoint"],
        reference_checkpoint=sources["reference_checkpoint"],
        parent_pfsp_state=None, baseline_evaluation_checkpoint=PARENT_UPDATE,
        periodic_evaluation_enabled=True, periodic_evaluation_interval_updates=5,
        periodic_evaluation_profile="policy0814_exact_deck_cuda512",
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0045 · V16 · U105 complete delta · StateEncoder LoRA recovery 2x",
        focal_deck_ids=(contract.FOCAL_DECK_ID,),
        focal_deck_id=contract.FOCAL_DECK_ID,
        evaluation_focal_deck_id=contract.FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION, source_parent_update=PARENT_UPDATE,
        reference_anchor_update=0,
        reference_anchor_identity="Policy-0814-V16-Complete-U0",
        allow_pristine_restart=True,
        opponent_sampling_mode="exact_deck_quota_training_pool",
        opponent_policy_ids=(contract.OPPONENT_POLICY_ID,),
        latest_champion_policy_id=contract.OPPONENT_POLICY_ID,
        rollout_games=contract.ROLLOUT_GAMES,
        opponent_deck_quotas=contract.FIXED_DECK_QUOTAS,
        opponent_random_deck_ids=contract.RANDOM_DECK_IDS,
        ppo_batch_size=contract.PPO_BATCH_SIZE,
        forward_microbatch_size=contract.FORWARD_MICROBATCH_SIZE,
        behavior_probe_batch_size=contract.BEHAVIOR_PROBE_BATCH_SIZE,
        offload_reference_after_cache=True,
        learning_rate_profile=LEARNING_RATE_PROFILE,
        focal_base_checkpoint=DEFAULT_0814_ACTOR_CHECKPOINT,
        focal_value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT,
        adaptation_config=contract.ADAPTATION,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    if args.launch_formal:
        launch(wandb_mode=args.wandb_mode, updates=args.updates)
    else:
        print(json.dumps(readiness(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
