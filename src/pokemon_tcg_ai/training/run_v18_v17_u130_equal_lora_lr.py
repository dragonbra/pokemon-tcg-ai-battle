"""V18: continue V17 U130 with StateEncoder LoRA LR equal to Option LoRA."""

from __future__ import annotations

import argparse
import json

from ..assets import sha256_file
from ..policy.actor_critic import DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT
from . import run_v13_policy0814_shared_encoder as contract
from .lr_profiles import LearningRateProfile
from .run_v1 import ROOT, run
from .run_v17_v16_u112_restart import REFERENCE_CHECKPOINT

VERSION = "V19_policy0814_v17_u130_state_option_lr_equal"
PARENT_VERSION = "V17_policy0814_v16_u112_complete_delta_restart"
PARENT_UPDATE = 130
PARENT_ROOT = ROOT / "runs/versions" / PARENT_VERSION
PARENT_CHECKPOINT = PARENT_ROOT / f"checkpoint/update-{PARENT_UPDATE:06d}.pt"
EXPECTED_PARENT_SHA256 = "70343580b660a9fc923be976fb285f4b32804fa7fdfd18444475d692f719134d"
WANDB_RUN_ID = "0045-v19-policy0814-v17-u130-equal-lora-lr"

LEARNING_RATE_PROFILE = LearningRateProfile(
    profile_id="0045_v18_state_option_lora_equal_lr_v1",
    base_profile_id="0045_v16_state_encoder_lora_recovery_2x_v1",
    scale=None,
    scope="v18_state_encoder_lora_equal_option_lora",
    default_after_run="0045_expert_cold_start_lr_v1",
    decoder_learning_rate=1.0e-5,
    allocation_learning_rate=1.0e-5,
    option_lora_learning_rate=2.0e-5,
    shared_encoder_learning_rate=2.0e-5,
    value_learning_rate=2.0e-5,
    prize_learning_rate=2.0e-5,
)


def readiness() -> dict[str, object]:
    if not PARENT_CHECKPOINT.is_file():
        raise FileNotFoundError(PARENT_CHECKPOINT)
    actual = sha256_file(PARENT_CHECKPOINT)
    if actual != EXPECTED_PARENT_SHA256:
        raise RuntimeError(f"V18 parent identity mismatch: expected {EXPECTED_PARENT_SHA256}, got {actual}")
    return {
        "status": "READY",
        "version": VERSION,
        "parent_version": PARENT_VERSION,
        "parent_update": PARENT_UPDATE,
        "parent_checkpoint_sha256": actual,
        "learning_rate_profile": LEARNING_RATE_PROFILE.metadata(),
        "baseline_eval": {"checkpoint_update": PARENT_UPDATE, "profile": "policy0814_exact_deck_cuda512", "games": 512},
        "optimizer_semantics": "fresh_optimizer_from_model_only_checkpoint",
    }


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    readiness()
    run(
        updates=updates, wandb_mode=wandb_mode, launch_formal=True,
        version=VERSION, start_update=PARENT_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT, reference_checkpoint=REFERENCE_CHECKPOINT,
        baseline_evaluation_checkpoint=PARENT_UPDATE,
        periodic_evaluation_enabled=True, periodic_evaluation_interval_updates=5,
        periodic_evaluation_profile="policy0814_exact_deck_cuda512",
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0045 - V18 - V17 U130 equal State/Option LoRA LR",
        focal_deck_ids=(contract.FOCAL_DECK_ID,), focal_deck_id=contract.FOCAL_DECK_ID,
        evaluation_focal_deck_id=contract.FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION, source_parent_update=PARENT_UPDATE,
        reference_anchor_update=0, reference_anchor_identity="Policy-0814-V16-Complete-U0",
        opponent_sampling_mode="exact_deck_quota_training_pool",
        opponent_policy_ids=(contract.OPPONENT_POLICY_ID,), latest_champion_policy_id=contract.OPPONENT_POLICY_ID,
        rollout_games=contract.ROLLOUT_GAMES, opponent_deck_quotas=contract.FIXED_DECK_QUOTAS,
        opponent_random_deck_ids=contract.RANDOM_DECK_IDS, ppo_batch_size=contract.PPO_BATCH_SIZE,
        forward_microbatch_size=contract.FORWARD_MICROBATCH_SIZE,
        behavior_probe_batch_size=contract.BEHAVIOR_PROBE_BATCH_SIZE,
        offload_reference_after_cache=True, learning_rate_profile=LEARNING_RATE_PROFILE,
        focal_base_checkpoint=DEFAULT_0814_ACTOR_CHECKPOINT,
        focal_value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT, adaptation_config=contract.ADAPTATION,
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
