"""V19: continue deck 069 from V18 U14 with half-standard LR and 1024 games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..assets import sha256_file
from .lr_profiles import scaled_standard_profile
from .run_v1 import ROOT, run
from .run_v17 import (
    FOCAL_DECK_ID,
    FORWARD_MICROBATCH_SIZE,
    G3_PORTABLE,
    OPPONENT_POLICY_ID,
    OPPONENT_SCHEDULE,
    PPO_MINIBATCH_SIZE,
)
from .run_v18 import readiness as v18_readiness


VERSION = "V19_v18_u14_deck069_half_standard_lr_1024_rollout"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
PARENT_VERSION = "V18_g4_u57_deck069_telemetry_fix"
SOURCE_PARENT_UPDATE = 14
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
    / "checkpoint/update-000014.pt"
)
PARENT_CHECKPOINT_SHA256 = "e43152d75ccbf1fd1a9c3967b3b2543f3e4ea478a339e24b69c7cc4ece00ffc9"
ROLLOUT_GAMES = 1024
OPPONENT_META_WEIGHTS = {0: 2.0}
LEARNING_RATE_PROFILE = scaled_standard_profile(
    0.5, override_id="0044_v19_half_standard_lr_run_override"
)
WANDB_RUN_ID = "0044-v19-v18-u14-deck069-half-standard-lr-1024-rollout"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v18_readiness(version_root=version_root)
    if not PARENT_CHECKPOINT.is_file():
        raise FileNotFoundError("V19 exact V18 U14 parent checkpoint is missing")
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V19 V18-U14 parent checkpoint identity changed")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V19 parent is not exact V18 U14")
    result["ppo"].update({
        "rollout_games": ROLLOUT_GAMES,
        "batch_size": PPO_MINIBATCH_SIZE,
        "forward_microbatch_size": FORWARD_MICROBATCH_SIZE,
        "learning_rate_profile": LEARNING_RATE_PROFILE.metadata(),
    })
    result.update({
        "schema_version": "0044_v19_u14_half_standard_lr_1024_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "source_update": SOURCE_PARENT_UPDATE,
            "local_start_update": 0,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "selection_evidence": (
                "Benchmark V2 Policy-0809 CUDA-2048 deck069; "
                "U14=1269-749-30 (61.9629%), best of U14-U17 and EMA"
            ),
            "optimizer": "fresh",
            "pfsp_state": None,
        },
        "learning_rate_contract": {
            **LEARNING_RATE_PROFILE.metadata(),
            "deck_identity_bound": False,
        },
        "rollout_contract": {
            "games_per_update": ROLLOUT_GAMES,
            "reason": "reduce gradient variance",
            "opponent_policy_id": OPPONENT_POLICY_ID,
            "opponent_sampling": OPPONENT_SCHEDULE,
            "opponent_meta_weights": OPPONENT_META_WEIGHTS,
            "display_meta_01_internal_archetype_id": 0,
            "weighted_meta_member_decks": ["007", "018", "067"],
            "expected_weighted_meta_lanes": 71,
            "evaluation_enabled": False,
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
        updates=updates,
        wandb_mode=wandb_mode,
        launch_formal=True,
        version=VERSION,
        start_update=0,
        parent_checkpoint=PARENT_CHECKPOINT,
        parent_pfsp_state=None,
        baseline_evaluation_checkpoint=None,
        periodic_evaluation_enabled=False,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V19 · V18 U14 → deck 069 · half standard LR · rollout 1024",
        focal_deck_ids=(FOCAL_DECK_ID,),
        focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        reference_anchor_update=SOURCE_PARENT_UPDATE,
        reference_anchor_identity=f"{PARENT_VERSION}@update-{SOURCE_PARENT_UPDATE:06d}",
        opponent_sampling_mode=OPPONENT_SCHEDULE,
        opponent_policy_ids=(OPPONENT_POLICY_ID,),
        latest_champion_policy_id=OPPONENT_POLICY_ID,
        rollout_games=ROLLOUT_GAMES,
        opponent_meta_weights=OPPONENT_META_WEIGHTS,
        ppo_batch_size=PPO_MINIBATCH_SIZE,
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
        learning_rate_profile=LEARNING_RATE_PROFILE,
        focal_base_checkpoint=G3_PORTABLE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"), default="online"
    )
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
