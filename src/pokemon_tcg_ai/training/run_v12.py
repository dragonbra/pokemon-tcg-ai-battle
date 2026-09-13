"""V12/G4: immutable Champion-G3 to a Meta-balanced 67-deck generalist."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..policy_identity import materialize_policy_bundle
from .run_v1 import PROJECT_ROOT, ROOT, RULES, _config, run


VERSION = "V12_g4_g3_meta_balanced_512_meta_residual"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
PARENT_VERSION = "V10_g3_v9_u50_generalist_001_067_core16_benchmark_v2"
SOURCE_PARENT_UPDATE = 110
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
    / "checkpoint/update-000110.pt"
)
PARENT_CHECKPOINT_SHA256 = "e1101f1712f2e4751aa56ba983b1a1165a9852b4cb1ab18913fbcd6108beac79"
G3_PORTABLE = PROJECT_ROOT / "assets/policies/definitions/champion_g003/model.bin"
OPPONENT_POLICY_ID = "Champion-G3"
FOCAL_DECK_IDS = tuple(f"{value:03d}" for value in range(1, 68))
INITIALIZATION_DECK_ID = "001"
EVALUATION_SENTINEL_DECK_ID = "007"
FOCAL_SCHEDULE = "meta_balanced_001_067"
OPPONENT_SCHEDULE = "meta_balanced_001_067"
ROLLOUT_GAMES = 512
PPO_MINIBATCH_SIZE = 4096
ACTOR_LEARNING_RATE_SCALE = 0.5
VALUE_LEARNING_RATE = 2.0e-5
META_RESIDUAL_LEARNING_RATE = 5.0e-6
WANDB_RUN_ID = "0044-v12-g4-g3-meta-balanced-512-meta-residual"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    audit = registry.validate_all()
    if (
        registry.latest_champion_policy_id != OPPONENT_POLICY_ID
        or registry.active_policy_ids != (OPPONENT_POLICY_ID,)
    ):
        raise RuntimeError("V12 opponent pool must be singleton immutable Champion-G3")
    required = (
        PARENT_CHECKPOINT, G3_PORTABLE, RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V12 launch artifacts missing: {missing}")
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V12 formal version path is already used: {version_root}")
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V12 G3 source checkpoint identity changed")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V12 parent is not exact G3 U110")
    if (PARENT_CHECKPOINT.parent / "update-000111.pt").exists():
        raise RuntimeError("V12 G3 source escaped the exact U110 boundary")
    bundle = materialize_policy_bundle(
        PROJECT_ROOT, OPPONENT_POLICY_ID, purpose="0044_v12_readiness"
    )
    ppo = _config(
        actor_learning_rate_scale=ACTOR_LEARNING_RATE_SCALE,
        batch_size=PPO_MINIBATCH_SIZE,
        value_learning_rate=VALUE_LEARNING_RATE,
        prize_learning_rate=VALUE_LEARNING_RATE,
        meta_actor_residual_learning_rate=META_RESIDUAL_LEARNING_RATE,
    )
    return {
        "schema_version": "0044_v12_g3_to_g4_launch_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "generation": {"source": "Champion-G3", "target": "Champion-G4"},
        "parent": {
            "version": PARENT_VERSION, "source_update": SOURCE_PARENT_UPDATE,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer": "fresh", "pfsp_state": None,
        },
        "policy_identity": {
            "focal_initialization": OPPONENT_POLICY_ID,
            "reference_anchor": OPPONENT_POLICY_ID,
            "opponent_policy_ids": [OPPONENT_POLICY_ID],
            "opponent_effective_policy_sha256": bundle.audit.effective_policy_sha256,
        },
        "sampling": {
            "rollout_games": ROLLOUT_GAMES,
            "focal": "Meta-first balanced then exact-deck-balanced within Meta",
            "opponent": "Meta-first balanced then exact-deck-balanced within Meta",
            "shared_meta_marginal": True,
            "independent_lane_order": True,
            "pfsp": False,
        },
        "first_player": {
            "contract": "seeded_toss_winner_agent_context_41_choice_v1",
            "scheduled_actual_seat_quota": None,
        },
        "model": {
            "meta_actor_residual": {
                "scope": "policy_only", "classes": 29, "rank": 4,
                "parameters": 74240, "initialization": "zero_residual",
            },
            "opponent_meta_head": "frozen_15_way_unchanged",
        },
        "ppo": asdict(ppo),
        "asset_audit": asdict(audit),
        "wandb": {
            "entity": "dragon_bra", "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    }


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    readiness()
    run(
        updates=updates, wandb_mode=wandb_mode, launch_formal=True,
        version=VERSION, start_update=0,
        parent_checkpoint=PARENT_CHECKPOINT, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=0,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V12 G4 · G3 → Meta-balanced 67-deck · 512",
        focal_deck_ids=FOCAL_DECK_IDS,
        focal_deck_id=INITIALIZATION_DECK_ID,
        focal_schedule_mode=FOCAL_SCHEDULE,
        evaluation_focal_deck_id=EVALUATION_SENTINEL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        opponent_sampling_mode=OPPONENT_SCHEDULE,
        opponent_policy_ids=(OPPONENT_POLICY_ID,),
        latest_champion_policy_id=OPPONENT_POLICY_ID,
        rollout_games=ROLLOUT_GAMES,
        ppo_batch_size=PPO_MINIBATCH_SIZE,
        forward_microbatch_size=1024,
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
