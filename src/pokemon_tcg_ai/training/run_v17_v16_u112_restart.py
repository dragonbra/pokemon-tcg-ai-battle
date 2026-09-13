"""V17: restart the interrupted V16 policy from its complete U112 delta."""

from __future__ import annotations

import argparse
import json

from ..assets import sha256_file
from ..policy.actor_critic import (
    DEFAULT_0814_ACTOR_CHECKPOINT,
    DEFAULT_0814_VALUE_CHECKPOINT,
)
from . import run_v13_policy0814_shared_encoder as contract
from .run_v1 import ROOT, run
from .run_v16_complete_delta_u105 import LEARNING_RATE_PROFILE


VERSION = "V17_policy0814_v16_u112_complete_delta_restart"
PARENT_VERSION = "V16_policy0814_u105_complete_delta_state_lora_recovery"
PARENT_UPDATE = 112
PARENT_ROOT = ROOT / "runs/versions" / PARENT_VERSION
PARENT_CHECKPOINT = PARENT_ROOT / f"checkpoint/update-{PARENT_UPDATE:06d}.pt"
REFERENCE_CHECKPOINT = PARENT_ROOT / "source/reference-policy0814-complete-u000000.pt"
EXPECTED_PARENT_SHA256 = "56367c949061f4fc47cddfc6515e9f2aff106eb4701a3c8f3f4e8653febb9613"
EXPECTED_REFERENCE_SHA256 = "2b437222b0061d9bb98c4df5fc15c49e02d46fbdf6391a1085aae1ab4f46cdc3"
WANDB_RUN_ID = "0045-v17-policy0814-v16-u112-complete-delta-restart"


def readiness() -> dict[str, object]:
    if sha256_file(PARENT_CHECKPOINT) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("V17 parent is not exact V16 U112")
    if sha256_file(REFERENCE_CHECKPOINT) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("V17 reference is not exact V16 Policy-0814 U0")
    return {
        "status": "READY",
        "version": VERSION,
        "parent_version": PARENT_VERSION,
        "parent_update": PARENT_UPDATE,
        "parent_checkpoint_sha256": EXPECTED_PARENT_SHA256,
        "reference_checkpoint_sha256": EXPECTED_REFERENCE_SHA256,
        "optimizer_semantics": "fresh_optimizer_from_model_only_checkpoint",
    }


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    readiness()
    run(
        updates=updates,
        wandb_mode=wandb_mode,
        launch_formal=True,
        version=VERSION,
        start_update=PARENT_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT,
        reference_checkpoint=REFERENCE_CHECKPOINT,
        baseline_evaluation_checkpoint=PARENT_UPDATE,
        periodic_evaluation_enabled=True,
        periodic_evaluation_interval_updates=5,
        periodic_evaluation_profile="policy0814_exact_deck_cuda512",
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0045 - V17 - V16 U112 complete-delta restart",
        focal_deck_ids=(contract.FOCAL_DECK_ID,),
        focal_deck_id=contract.FOCAL_DECK_ID,
        evaluation_focal_deck_id=contract.FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=PARENT_UPDATE,
        reference_anchor_update=0,
        reference_anchor_identity="Policy-0814-V16-Complete-U0",
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
