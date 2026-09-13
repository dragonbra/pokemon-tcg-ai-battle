"""Canonical public trainer: Policy-0814 BC initialization plus PPO."""

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
    DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT,
    load_actor_critic,
)
from ..policy_identity import materialize_policy_bundle
from .runner import PROJECT_ROOT, ROOT, RULES, _atomic_torch, _cards, _checkpoint, _config, run
from .lr_profiles import LearningRateProfile
from .trainable_audit import trainable_tensor_audit


VERSION = "dragapult_policy0814_bc_ppo"
VERSION_ROOT = ROOT / "runs/versions" / VERSION
U0_CHECKPOINT = VERSION_ROOT / "checkpoint/update-000000.pt"
FOCAL_DECK_ID = "007"
OPPONENT_POLICY_ID = "Policy-0814"
ROLLOUT_GAMES = 512
PPO_BATCH_SIZE = 4096
FORWARD_MICROBATCH_SIZE = 256
BEHAVIOR_PROBE_BATCH_SIZE = 256
WANDB_RUN_ID = "pokemon-tcg-ai-public-dragapult-policy0814-bc-ppo"
FIXED_DECK_QUOTAS = {
    "001": 143, "002": 68, "003": 48, "007": 64,
    "071": 30, "008": 14, "009": 16, "011": 7,
}
RANDOM_DECK_IDS = ("071", "008", "009", "011")
ADAPTATION = AdaptationConfig(
    rank=16,
    alpha=16.0,
    output_projection=True,
    shared_state_encoder=True,
    option_ffn_lora=True,
    state_ffn_lora=True,
    layernorm_tuning=True,
)
EXPECTED_TRAINABLE_PARAMS = 5_597_590
EXPECTED_TRAINABLE_TENSORS = 125
LEARNING_RATE_PROFILE = LearningRateProfile(
    profile_id="public_expanded_lora_v1",
    base_profile_id="policy0814_u125_ffn_lora_norm_expansion_v1",
    scale=None,
    scope="public_reproducible_training",
    default_after_run="public_expanded_lora_v1",
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


def _model_and_audit():
    registry = AssetRegistry.load(PROJECT_ROOT)
    model, source = load_actor_critic(
        checkpoint=DEFAULT_0814_ACTOR_CHECKPOINT,
        value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT,
        deck=_cards(registry, FOCAL_DECK_ID), deck_id=FOCAL_DECK_ID,
        device="cpu", adaptation=ADAPTATION,
    )
    audit = trainable_tensor_audit(model)
    if (
        audit["tensor_count"] != EXPECTED_TRAINABLE_TENSORS
        or audit["total_trainable_params"] != EXPECTED_TRAINABLE_PARAMS
    ):
        raise RuntimeError(f"public trainable boundary changed: {audit}")
    model.assert_trainable_contract()
    return model, source, audit


def readiness(*, require_pristine: bool = True) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = registry.validate_all()
    required = (
        RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
        DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"public training artifacts missing: {missing}")
    if require_pristine and VERSION_ROOT.exists() and any(VERSION_ROOT.rglob("*")):
        raise FileExistsError(f"public training path is already used: {VERSION_ROOT}")
    bundle = materialize_policy_bundle(
        PROJECT_ROOT, OPPONENT_POLICY_ID, purpose="public_training_readiness"
    )
    model, source, trainable = _model_and_audit()
    ppo = asdict(_config(
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
        behavior_probe_batch_size=BEHAVIOR_PROBE_BATCH_SIZE,
        batch_size=PPO_BATCH_SIZE,
        learning_rate_profile=LEARNING_RATE_PROFILE,
    ))
    del model
    return {
        "schema_version": "public_policy0814_bc_ppo_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "focal": {"deck_id": FOCAL_DECK_ID, "initial_policy": OPPONENT_POLICY_ID},
        "source": {
            "actor_checkpoint": str(DEFAULT_0814_ACTOR_CHECKPOINT),
            "actor_sha256": sha256_file(DEFAULT_0814_ACTOR_CHECKPOINT),
            "value_checkpoint": str(DEFAULT_0814_VALUE_CHECKPOINT),
            "value_sha256": sha256_file(DEFAULT_0814_VALUE_CHECKPOINT),
            "identity": asdict(source),
        },
        "opponent": {
            "policy_id": OPPONENT_POLICY_ID,
            "effective_policy_sha256": bundle.audit.effective_policy_sha256,
            "policy_count": 1,
        },
        "rollout": {
            "games": ROLLOUT_GAMES, "lanes": 512,
            "fixed_deck_quotas": FIXED_DECK_QUOTAS,
            "random_remainder_deck_ids": list(RANDOM_DECK_IDS),
        },
        "eval": {
            "every_updates": 5, "games": 512,
            "opponent_policy_id": OPPONENT_POLICY_ID,
            "schedule": "same fixed quotas plus frozen random remainder",
            "outputs": ["summary", "per_deck"],
        },
        "ppo": ppo,
        "trainable": trainable,
        "asset_audit": asdict(asset_audit),
        "wandb": {
            "entity": "dragon_bra", "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    }


def prepare_u0() -> dict[str, object]:
    evidence = readiness(require_pristine=True)
    model, _, audit = _model_and_audit()
    for name in ("artifact", "checkpoint", "tensorboard", "wandb"):
        (VERSION_ROOT / name).mkdir(parents=True, exist_ok=True)
    _atomic_torch(U0_CHECKPOINT, _checkpoint(
        model, 0, version=VERSION, focal_deck_id=FOCAL_DECK_ID,
        focal_deck_ids=(FOCAL_DECK_ID,), focal_schedule_mode="fixed",
        source_parent_version="Policy-0814-pretrained",
        source_parent_update=None, parent_policy_id=OPPONENT_POLICY_ID,
        parent_policy_update=None,
    ))
    _atomic_json(VERSION_ROOT / "artifact/trainable_tensor_audit.json", audit)
    _atomic_json(VERSION_ROOT / "artifact/readiness.json", evidence)
    return evidence


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    prepare_u0()
    run(
        updates=updates, wandb_mode=wandb_mode, launch_formal=True,
        version=VERSION, start_update=0, parent_checkpoint=U0_CHECKPOINT,
        reference_checkpoint=U0_CHECKPOINT, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=None,
        periodic_evaluation_enabled=True, periodic_evaluation_interval_updates=5,
        periodic_evaluation_profile="policy0814_exact_deck_cuda512",
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="public · deck 007 · Policy-0814 BC + PPO",
        focal_deck_ids=(FOCAL_DECK_ID,), focal_deck_id=FOCAL_DECK_ID,
        evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version="Policy-0814-pretrained", source_parent_update=None,
        reference_anchor_update=None, reference_anchor_identity="Policy-0814-Public-U0",
        opponent_sampling_mode="exact_deck_quota_training_pool",
        opponent_policy_ids=(OPPONENT_POLICY_ID,),
        latest_champion_policy_id=OPPONENT_POLICY_ID,
        rollout_games=ROLLOUT_GAMES,
        opponent_deck_quotas=FIXED_DECK_QUOTAS,
        opponent_random_deck_ids=RANDOM_DECK_IDS,
        ppo_batch_size=PPO_BATCH_SIZE,
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
        behavior_probe_batch_size=BEHAVIOR_PROBE_BATCH_SIZE,
        offload_reference_after_cache=True,
        learning_rate_profile=LEARNING_RATE_PROFILE,
        focal_base_checkpoint=DEFAULT_0814_ACTOR_CHECKPOINT,
        focal_value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT,
        adaptation_config=ADAPTATION,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--readiness-output", type=Path)
    args = parser.parse_args()
    if args.launch_formal:
        launch(wandb_mode=args.wandb_mode, updates=args.updates)
        return 0
    result = readiness()
    if args.readiness_output:
        _atomic_json(args.readiness_output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
