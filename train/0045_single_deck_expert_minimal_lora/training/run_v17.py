"""V17: specialize V15 U57 on focal deck 069 against frozen Champion-G3."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..own_archetype import OwnArchetypeVocabulary
from ..policy_identity import materialize_policy_bundle
from .run_v1 import PROJECT_ROOT, ROOT, RULES, _config, run
from .run_v12 import (
    ACTOR_LEARNING_RATE_SCALE,
    G3_PORTABLE,
    META_RESIDUAL_LEARNING_RATE,
    OPPONENT_POLICY_ID,
    PPO_MINIBATCH_SIZE,
    ROLLOUT_GAMES,
    VALUE_LEARNING_RATE,
)
from .run_v15 import FORWARD_MICROBATCH_SIZE


VERSION = "V17_g4_u57_deck069_champion_g3_rollout_only"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
PARENT_VERSION = "V15_g4_v14_u2_rollout_only_cuda_recovery"
SOURCE_PARENT_UPDATE = 57
REFERENCE_ANCHOR_UPDATE = SOURCE_PARENT_UPDATE
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
    / "checkpoint/update-000057.pt"
)
PARENT_CHECKPOINT_SHA256 = "dff296e81046b0882b0a61f89e926a6a1d401c7fcb04eefd8841ec133200fe50"
FOCAL_DECK_ID = "069"
FOCAL_OWN_ARCHETYPE_ID = 3
OPPONENT_SCHEDULE = "meta_balanced_training_pool"
WANDB_RUN_ID = "0044-v17-g4-u57-deck069-champion-g3-rollout-only"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = registry.validate_all()
    training_deck_ids = tuple(
        row.deck_id for row in registry.decks if "training" in row.roles
    )
    expected_deck_ids = tuple(f"{value:03d}" for value in range(1, 70))
    if training_deck_ids != expected_deck_ids:
        raise RuntimeError("V17 requires the exact expanded training pool 001-069")
    focal_asset = next(row for row in registry.decks if row.deck_id == FOCAL_DECK_ID)
    if focal_asset.roles != ("training",):
        raise RuntimeError("V17 focal deck 069 is not admitted to formal training")
    own = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in own.mappings}
    if own_by_deck.get(FOCAL_DECK_ID) != FOCAL_OWN_ARCHETYPE_ID:
        raise RuntimeError("V17 deck 069 must route through own Meta class 03")
    if (
        registry.latest_champion_policy_id != OPPONENT_POLICY_ID
        or registry.active_policy_ids != (OPPONENT_POLICY_ID,)
    ):
        raise RuntimeError("V17 opponent policy pool must be frozen Champion-G3 only")
    required = (
        PARENT_CHECKPOINT, G3_PORTABLE, RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V17 launch artifacts missing: {missing}")
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V17 formal version path is already used: {version_root}")
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V17 V15-U57 parent checkpoint identity changed")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V17 parent is not exact V15 U57")
    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, OPPONENT_POLICY_ID, purpose="0044_v17_readiness"
    )
    ppo = _config(
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
        actor_learning_rate_scale=ACTOR_LEARNING_RATE_SCALE,
        batch_size=PPO_MINIBATCH_SIZE,
        value_learning_rate=VALUE_LEARNING_RATE,
        prize_learning_rate=VALUE_LEARNING_RATE,
        meta_actor_residual_learning_rate=META_RESIDUAL_LEARNING_RATE,
    )
    return {
        "schema_version": "0044_v17_deck069_launch_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "generation": {
            "source": "V15 G4 candidate U57",
            "target": "deck-069 specialist",
            "champion_mutation": False,
        },
        "parent": {
            "version": PARENT_VERSION,
            "source_update": SOURCE_PARENT_UPDATE,
            "local_start_update": 0,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer": "fresh",
            "pfsp_state": None,
        },
        "focal": {
            "deck_id": FOCAL_DECK_ID,
            "own_archetype_id": FOCAL_OWN_ARCHETYPE_ID,
            "schedule": "fixed_069_v1",
            "registry_role": "training",
        },
        "opponents": {
            "policy_ids": [OPPONENT_POLICY_ID],
            "effective_policy_sha256": opponent_bundle.audit.effective_policy_sha256,
            "deck_ids": list(training_deck_ids),
            "sampling": OPPONENT_SCHEDULE,
            "pfsp": False,
        },
        "evaluation": {
            "enabled": False,
            "reason": "rollout-only specialization; Benchmark V2 runs separately",
        },
        "reference": {
            "source": "exact V15 U57 parent checkpoint",
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "checkpoint_update": REFERENCE_ANCHOR_UPDATE,
        },
        "ppo": asdict(ppo),
        "asset_audit": asdict(asset_audit),
        "wandb": {
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    }


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
        wandb_name="0044 · V17 · V15 U57 → deck 069 · Champion-G3 rollout only",
        focal_deck_ids=(FOCAL_DECK_ID,),
        focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        reference_anchor_update=REFERENCE_ANCHOR_UPDATE,
        reference_anchor_identity=f"{PARENT_VERSION}@update-{SOURCE_PARENT_UPDATE:06d}",
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
