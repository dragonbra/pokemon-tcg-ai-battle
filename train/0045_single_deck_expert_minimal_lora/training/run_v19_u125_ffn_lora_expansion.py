"""V19: zero-delta FFN LoRA/Option norm expansion from exact V17 U125."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..policy import AdaptationConfig
from ..policy.actor_critic import (
    DEFAULT_0814_ACTOR_CHECKPOINT,
    DEFAULT_0814_VALUE_CHECKPOINT,
    load_actor_critic,
)
from ..policy_identity import materialize_policy_bundle
from ..runtime import synthetic_batch
from . import run_v13_policy0814_shared_encoder as contract
from .checkpointing import (
    atomic_save_complete_delta,
    audit_reconstruction,
    load_complete_delta,
)
from .lr_profiles import LearningRateProfile
from .run_v1 import PROJECT_ROOT, ROOT, RULES, _cards, _checkpoint, run
from .trainable_audit import trainable_tensor_audit


VERSION = "V19_policy0814_u125_ffn_lora_norm_expansion"
VERSION_ROOT = ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions" / VERSION
SOURCE_ROOT = VERSION_ROOT / "source"
PARENT_VERSION = "V17_policy0814_v16_u112_complete_delta_restart"
PARENT_UPDATE = 125
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions" / PARENT_VERSION
    / f"checkpoint/update-{PARENT_UPDATE:06d}.pt"
)
REFERENCE_CHECKPOINT = (
    ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions"
    / "V16_policy0814_u105_complete_delta_state_lora_recovery"
    / "source/reference-policy0814-complete-u000000.pt"
)
EXPECTED_PARENT_SHA256 = "cad14c2dcb9c3dffed8fa739969cfb0952f664607854dda16fdc03f1d127dc2b"
EXPECTED_REFERENCE_SHA256 = "2b437222b0061d9bb98c4df5fc15c49e02d46fbdf6391a1085aae1ab4f46cdc3"
EXPANSION_SEED = 4_519_125
RECONSTRUCTION_SEED = 4_519_126
WANDB_RUN_ID = "0045-v19-policy0814-u125-ffn-lora-norm-expansion"

LEGACY_ADAPTATION = contract.ADAPTATION
EXPANDED_ADAPTATION = AdaptationConfig(
    rank=16,
    alpha=16.0,
    output_projection=True,
    shared_state_encoder=True,
    option_ffn_lora=True,
    state_ffn_lora=True,
    layernorm_tuning=True,
)
LEARNING_RATE_PROFILE = LearningRateProfile(
    profile_id="0045_v19_u125_ffn_lora_norm_expansion_v1",
    base_profile_id="0045_v16_state_encoder_lora_recovery_2x_v1",
    scale=None,
    scope="v19_u125_zero_delta_expansion",
    default_after_run="0045_expert_cold_start_lr_v1",
    decoder_learning_rate=1.0e-5,
    allocation_learning_rate=1.0e-5,
    option_lora_learning_rate=2.0e-5,
    shared_encoder_learning_rate=2.0e-5,
    value_learning_rate=2.0e-5,
    prize_learning_rate=2.0e-5,
    option_ffn_learning_rate=3.0e-5,
    state_ffn_learning_rate=1.5e-5,
    option_norm_learning_rate=5.0e-6,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _model(adaptation: AdaptationConfig, seed: int):
    registry = AssetRegistry.load(PROJECT_ROOT)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, source = load_actor_critic(
            checkpoint=DEFAULT_0814_ACTOR_CHECKPOINT,
            value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT,
            deck=_cards(registry, contract.FOCAL_DECK_ID),
            deck_id=contract.FOCAL_DECK_ID,
            device="cpu",
            adaptation=adaptation,
        )
    return model, source


def _load_legacy(path: Path, seed: int):
    model, _ = _model(LEGACY_ADAPTATION, seed)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    load_complete_delta(model, payload)
    return model.eval(), payload


def _expand(legacy_model, legacy_payload, seed: int):
    expanded, source = _model(EXPANDED_ADAPTATION, seed)
    incompatible = expanded.load_state_dict(legacy_payload["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"U125 expansion has unexpected tensors: {incompatible.unexpected_keys}")
    legacy_names = set(legacy_payload["state_dict"])
    if not legacy_names.issubset(expanded.state_dict()):
        raise RuntimeError("U125 expansion failed to preserve every legacy trainable tensor")
    expanded.assert_trainable_contract()
    return expanded.eval(), source


def _policy_outputs(model, batch):
    validated, state, options = model.encode_policy(batch)
    decoder_state = model.actor.action_decoder.initialize(
        validated, model.actor_summary(state)
    )
    logits = model.head.logits(validated, options, decoder_state)
    actions = model.actor.action_decoder.greedy(
        validated, options, model.actor_summary(state)
    )
    return logits, actions


def _parity(legacy_model, expanded_model) -> dict[str, object]:
    batch = synthetic_batch(batch_size=2)
    with torch.no_grad():
        old_logits, old_actions = _policy_outputs(legacy_model, batch)
        new_logits, new_actions = _policy_outputs(expanded_model, batch)
    logits_equal = torch.equal(old_logits, new_logits)
    action_equal = all(torch.equal(getattr(old_actions, field), getattr(new_actions, field)) for field in (
        "sequences", "lengths", "legal",
    ))
    if not logits_equal or not action_equal:
        raise RuntimeError("zero-delta expansion changed U125 policy behavior")
    return {
        "logits_exact_equal": logits_equal,
        "greedy_actions_exact_equal": action_equal,
        "max_abs_logit_difference": float((old_logits - new_logits).abs().max()),
    }


def materialize_sources(output_root: Path = SOURCE_ROOT) -> dict[str, object]:
    if output_root.exists() and any(output_root.rglob("*")):
        raise FileExistsError(f"V19 source materialization is already used: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    old_parent, parent_legacy_payload = _load_legacy(PARENT_CHECKPOINT, EXPANSION_SEED)
    parent, source = _expand(old_parent, parent_legacy_payload, EXPANSION_SEED)
    behavior = _parity(old_parent, parent)
    trainable = trainable_tensor_audit(parent)
    if trainable["tensor_count"] != 125 or trainable["total_trainable_params"] != 5_597_590:
        raise RuntimeError(f"V19 expanded trainable boundary changed: {trainable}")

    parent_payload = _checkpoint(
        parent, PARENT_UPDATE, version=VERSION,
        focal_deck_id=contract.FOCAL_DECK_ID,
        focal_deck_ids=(contract.FOCAL_DECK_ID,), focal_schedule_mode="fixed",
        source_parent_version=PARENT_VERSION, source_parent_update=PARENT_UPDATE,
        parent_policy_id=contract.OPPONENT_POLICY_ID, parent_policy_update=None,
    )
    parent_path = output_root / f"expanded-parent-update-{PARENT_UPDATE:06d}.pt"
    atomic_save_complete_delta(parent_path, parent_payload, parent)
    reconstructed, _ = _model(EXPANDED_ADAPTATION, RECONSTRUCTION_SEED)
    load_complete_delta(reconstructed, parent_payload)
    parent_reconstruction = audit_reconstruction(parent, reconstructed, parent_payload)

    old_reference, reference_legacy_payload = _load_legacy(
        REFERENCE_CHECKPOINT, EXPANSION_SEED + 1
    )
    reference, _ = _expand(old_reference, reference_legacy_payload, EXPANSION_SEED + 1)
    reference_behavior = _parity(old_reference, reference)
    reference_payload = _checkpoint(
        reference, 0, version=VERSION,
        focal_deck_id=contract.FOCAL_DECK_ID,
        focal_deck_ids=(contract.FOCAL_DECK_ID,), focal_schedule_mode="fixed",
        source_parent_version="Policy-0814-pretrained", source_parent_update=None,
        parent_policy_id=contract.OPPONENT_POLICY_ID, parent_policy_update=None,
    )
    reference_path = output_root / "expanded-reference-update-000000.pt"
    atomic_save_complete_delta(reference_path, reference_payload, reference)
    reference_reconstructed, _ = _model(EXPANDED_ADAPTATION, RECONSTRUCTION_SEED + 1)
    load_complete_delta(reference_reconstructed, reference_payload)
    reference_reconstruction = audit_reconstruction(
        reference, reference_reconstructed, reference_payload
    )

    audit = {
        "schema_version": "0045_v19_u125_zero_delta_expansion_audit_v1",
        "status": "PASS",
        **behavior,
        "reference_logits_exact_equal": reference_behavior["logits_exact_equal"],
        "reference_greedy_actions_exact_equal": reference_behavior["greedy_actions_exact_equal"],
        "parent_source_sha256": EXPECTED_PARENT_SHA256,
        "parent_expanded_sha256": sha256_file(parent_path),
        "parent_reconstruction": parent_reconstruction,
        "reference_reconstruction": reference_reconstruction,
        "trainable": trainable,
        "source_identity": asdict(source),
        "learning_rate_profile": LEARNING_RATE_PROFILE.metadata(),
    }
    _atomic_json(output_root / "expansion_audit.json", audit)
    return {
        "parent_checkpoint": parent_path,
        "reference_checkpoint": reference_path,
        "audit": audit,
    }


def readiness(*, require_pristine: bool = True) -> dict[str, object]:
    required = (
        RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so", PARENT_CHECKPOINT,
        REFERENCE_CHECKPOINT, DEFAULT_0814_ACTOR_CHECKPOINT,
        DEFAULT_0814_VALUE_CHECKPOINT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V19 launch artifacts missing: {missing}")
    if sha256_file(PARENT_CHECKPOINT) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("V19 parent is not exact V17 U125")
    if sha256_file(REFERENCE_CHECKPOINT) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("V19 reference identity changed")
    if require_pristine and VERSION_ROOT.exists() and any(VERSION_ROOT.rglob("*")):
        raise FileExistsError(f"V19 formal version path is already used: {VERSION_ROOT}")
    bundle = materialize_policy_bundle(
        PROJECT_ROOT, contract.OPPONENT_POLICY_ID, purpose="0045_v19_readiness"
    )
    model, source = _model(EXPANDED_ADAPTATION, EXPANSION_SEED)
    trainable = trainable_tensor_audit(model)
    return {
        "schema_version": "0045_v19_u125_ffn_lora_norm_readiness_v1",
        "status": "READY_AWAITING_MATERIALIZATION_AND_LAUNCH",
        "version": VERSION,
        "parent": {"version": PARENT_VERSION, "update": PARENT_UPDATE,
                   "sha256": EXPECTED_PARENT_SHA256},
        "optimizer": {"fresh": True, "groups": LEARNING_RATE_PROFILE.metadata()},
        "zero_delta_behavior_gate": "exact_logits_and_greedy_actions_required",
        "checkpoint_gate": "125_trainable_tensors_plus_full_state_reconstruction_hash",
        "trainable": trainable,
        "source": asdict(source),
        "opponent_effective_policy_sha256": bundle.audit.effective_policy_sha256,
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
        wandb_name="0045 - V19 - U125 zero-delta FFN LoRA and Option norm expansion",
        focal_deck_ids=(contract.FOCAL_DECK_ID,), focal_deck_id=contract.FOCAL_DECK_ID,
        evaluation_focal_deck_id=contract.FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION, source_parent_update=PARENT_UPDATE,
        reference_anchor_update=0,
        reference_anchor_identity="Policy-0814-V19-Expanded-Complete-U0",
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
        adaptation_config=EXPANDED_ADAPTATION,
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
