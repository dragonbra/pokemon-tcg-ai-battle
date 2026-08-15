"""V15/G4: recover V14 U2 after a transient CUDA device-not-ready failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..assets import sha256_file
from .run_v1 import ROOT, run
from .run_v14 import readiness as v14_readiness
from .run_v12 import (
    ACTOR_LEARNING_RATE_SCALE,
    FOCAL_DECK_IDS,
    FOCAL_SCHEDULE,
    G3_PORTABLE,
    INITIALIZATION_DECK_ID,
    META_RESIDUAL_LEARNING_RATE,
    OPPONENT_POLICY_ID,
    OPPONENT_SCHEDULE,
    PPO_MINIBATCH_SIZE,
    ROLLOUT_GAMES,
    VALUE_LEARNING_RATE,
)


VERSION = "V15_g4_v14_u2_rollout_only_cuda_recovery"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
PARENT_VERSION = "V14_g4_g3_meta_balanced_512_rollout_only"
SOURCE_PARENT_UPDATE = 2
REFERENCE_ANCHOR_UPDATE = 110
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
    / "checkpoint/update-000002.pt"
)
PARENT_CHECKPOINT_SHA256 = "e20f30bd59cdf632c23929f45df5365e7807f48342b94443fd4972448e76ecc1"
FORWARD_MICROBATCH_SIZE = 768
WANDB_RUN_ID = "0044-v15-g4-v14-u2-rollout-only-cuda-recovery"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v14_readiness(version_root=version_root)
    if not PARENT_CHECKPOINT.is_file():
        raise FileNotFoundError("V15 exact V14 U2 parent checkpoint is missing")
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V15 V14-U2 parent checkpoint identity changed")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V15 parent is not exact V14 U2")
    if (PARENT_CHECKPOINT.parent / "update-000003.pt").exists():
        raise RuntimeError("V15 recovery boundary escaped durable V14 U2")
    result["ppo"]["forward_microbatch_size"] = FORWARD_MICROBATCH_SIZE
    result.update({
        "schema_version": "0044_v15_cuda_recovery_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "retry_of": PARENT_VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "source_update": SOURCE_PARENT_UPDATE,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer": "fresh",
            "discarded_partial_update": 3,
        },
        "reference_anchor": {
            "policy_id": OPPONENT_POLICY_ID,
            "promoted_source_update": REFERENCE_ANCHOR_UPDATE,
        },
        "cuda_recovery": {
            "v14_error": "CUDA driver error: device not ready",
            "persistent_device_failure": False,
            "post_failure_cuda_smoke": "PASS",
            "logical_minibatch": PPO_MINIBATCH_SIZE,
            "physical_microbatch": FORWARD_MICROBATCH_SIZE,
        },
        "wandb": {
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    })
    return result


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    readiness()
    run(
        updates=updates, wandb_mode=wandb_mode, launch_formal=True,
        version=VERSION, start_update=0,
        parent_checkpoint=PARENT_CHECKPOINT, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=None,
        periodic_evaluation_enabled=False,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V15 G4 · V14 U2 CUDA recovery · rollout only",
        focal_deck_ids=FOCAL_DECK_IDS,
        focal_deck_id=INITIALIZATION_DECK_ID,
        focal_schedule_mode=FOCAL_SCHEDULE,
        evaluation_focal_deck_id=INITIALIZATION_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        reference_anchor_update=REFERENCE_ANCHOR_UPDATE,
        opponent_sampling_mode=OPPONENT_SCHEDULE,
        opponent_policy_ids=(OPPONENT_POLICY_ID,),
        latest_champion_policy_id=OPPONENT_POLICY_ID,
        rollout_games=ROLLOUT_GAMES,
        ppo_batch_size=PPO_MINIBATCH_SIZE,
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
        actor_learning_rate_scale=ACTOR_LEARNING_RATE_SCALE,
        value_learning_rate=VALUE_LEARNING_RATE,
        prize_learning_rate=VALUE_LEARNING_RATE,
        meta_actor_residual_learning_rate=META_RESIDUAL_LEARNING_RATE,
        focal_base_checkpoint=G3_PORTABLE,
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
        args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
        args.readiness_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
