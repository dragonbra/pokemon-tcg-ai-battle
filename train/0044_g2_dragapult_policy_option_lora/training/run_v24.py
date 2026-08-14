"""V24: switch V23 U11 to focal deck 023 and restore standard LR/entropy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..own_archetype import OwnArchetypeVocabulary
from .lr_profiles import STANDARD_LR_PROFILE
from .run_v1 import PROJECT_ROOT, ROOT, run
from .run_v23 import (
    BEHAVIOR_PROBE_BATCH_SIZE,
    FORWARD_MICROBATCH_SIZE,
    G3_PORTABLE,
    LEARNING_RATE_PROFILE as PREVIOUS_LEARNING_RATE_PROFILE,
    OFFLOAD_REFERENCE_AFTER_CACHE,
    OPPONENT_META_WEIGHTS,
    OPPONENT_POLICY_ID,
    OPPONENT_SCHEDULE,
    PPO_MINIBATCH_SIZE,
    ROLLOUT_GAMES,
    VERSION as PARENT_VERSION,
    readiness as v23_readiness,
)


VERSION = "V24_v23_u11_deck023_standard_lr_entropy"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
SOURCE_PARENT_UPDATE = 11
PARENT_CHECKPOINT = (
    ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
    / "checkpoint/update-000011.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "9bcb8d0c948d40fc7756fbf204ce2946150692d29605c5e5b0114c71472738de"
)
PARENT_STATUS = (
    ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
    / "artifact/status.json"
)
FOCAL_DECK_ID = "023"
FOCAL_DECK_DISPLAY_NAME = "Hydrapple ex / Meganium"
FOCAL_OWN_ARCHETYPE_ID = 27
LEARNING_RATE_PROFILE = STANDARD_LR_PROFILE
ENTROPY_COEFFICIENT_BEFORE = 0.0015
ENTROPY_COEFFICIENT = 0.003
WANDB_RUN_ID = "0044-v24-v23-u11-deck023-standard-lr-entropy"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    result = v23_readiness(version_root=version_root)
    if not PARENT_CHECKPOINT.is_file():
        raise FileNotFoundError("V24 exact V23 U11 parent checkpoint is missing")
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V24 V23-U11 parent checkpoint identity changed")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V24 parent is not exact durable V23 U11")
    if not PARENT_STATUS.is_file():
        raise FileNotFoundError("V24 requires the sealed V23 status record")
    parent_status = json.loads(PARENT_STATUS.read_text(encoding="utf-8"))
    if (
        parent_status.get("state") != "stopped"
        or parent_status.get("last_durable_checkpoint_update") != SOURCE_PARENT_UPDATE
        or parent_status.get("last_durable_metrics_update") != SOURCE_PARENT_UPDATE
    ):
        raise RuntimeError("V24 parent is not the sealed durable V23 U11 boundary")

    registry = AssetRegistry.load(PROJECT_ROOT)
    focal_asset = next(
        (row for row in registry.decks if row.deck_id == FOCAL_DECK_ID), None
    )
    if focal_asset is None:
        raise RuntimeError("V24 focal deck 023 is absent from the 0044 registry")
    if focal_asset.archetype != FOCAL_DECK_DISPLAY_NAME:
        raise RuntimeError("V24 focal deck 023 display identity changed")
    if focal_asset.roles != ("training", "evaluation"):
        raise RuntimeError("V24 focal deck 023 must retain training/evaluation roles")
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    if own_by_deck.get(FOCAL_DECK_ID) != FOCAL_OWN_ARCHETYPE_ID:
        raise RuntimeError("V24 deck 023 must route through Own Archetype V2 class 27")

    result["ppo"].update({
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "rollout_games": ROLLOUT_GAMES,
        "batch_size": PPO_MINIBATCH_SIZE,
        "forward_microbatch_size": FORWARD_MICROBATCH_SIZE,
        "behavior_probe_batch_size": BEHAVIOR_PROBE_BATCH_SIZE,
        "offload_reference_after_cache": OFFLOAD_REFERENCE_AFTER_CACHE,
        "learning_rate_profile": LEARNING_RATE_PROFILE.metadata(),
    })
    result.update({
        "schema_version": "0044_v24_v23_u11_deck023_standard_training_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
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
            "display_name": FOCAL_DECK_DISPLAY_NAME,
            "own_archetype_id": FOCAL_OWN_ARCHETYPE_ID,
            "schedule": "fixed_023_v1",
            "registry_roles": list(focal_asset.roles),
        },
        "standard_restore": {
            "on_policy_data": "fresh",
            "learning_rate_before": PREVIOUS_LEARNING_RATE_PROFILE.profile_id,
            "learning_rate_after": LEARNING_RATE_PROFILE.profile_id,
            "entropy_coefficient_before": ENTROPY_COEFFICIENT_BEFORE,
            "entropy_coefficient_after": ENTROPY_COEFFICIENT,
            "semantic_changes": [
                "focal_exact_deck_069_to_023",
                "learning_rate_half_standard_to_standard",
                "entropy_coefficient_0.0015_to_0.003",
            ],
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
        wandb_name="0044 · V24 · V23 U11 → deck 023 · standard LR/entropy",
        focal_deck_ids=(FOCAL_DECK_ID,),
        focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        reference_anchor_update=SOURCE_PARENT_UPDATE,
        reference_anchor_identity=(
            f"{PARENT_VERSION}@update-{SOURCE_PARENT_UPDATE:06d}"
        ),
        opponent_sampling_mode=OPPONENT_SCHEDULE,
        opponent_policy_ids=(OPPONENT_POLICY_ID,),
        latest_champion_policy_id=OPPONENT_POLICY_ID,
        rollout_games=ROLLOUT_GAMES,
        opponent_meta_weights=OPPONENT_META_WEIGHTS,
        ppo_batch_size=PPO_MINIBATCH_SIZE,
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
        behavior_probe_batch_size=BEHAVIOR_PROBE_BATCH_SIZE,
        offload_reference_after_cache=OFFLOAD_REFERENCE_AFTER_CACHE,
        learning_rate_profile=LEARNING_RATE_PROFILE,
        entropy_coefficient=ENTROPY_COEFFICIENT,
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
