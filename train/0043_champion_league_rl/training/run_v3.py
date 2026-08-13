"""V3 generalist continuation with focal decks 001-067.

The default command is a read-only readiness audit. Long training still requires
the user's explicit ``--launch-formal`` authority.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..own_archetype import OwnArchetypeVocabulary
from .run_v1 import PROJECT_ROOT, ROOT, RULES, run


VERSION = "V3_generalist_focal_001_067"
START_UPDATE = 207
FOCAL_DECK_IDS = tuple(f"{index:03d}" for index in range(1, 68))
PARENT_ROOT = ROOT / "rl_runs/0043_champion_league_rl/versions/V2_mixed_focal_cohorts"
PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint/update-000207.pt"
PARENT_PFSP_STATE = PARENT_ROOT / "artifact/pfsp_state.json"
VERSION_ROOT = ROOT / "rl_runs/0043_champion_league_rl/versions" / VERSION
WANDB_RUN_ID = "0043-v3-generalist-focal-001-067"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = registry.validate_all()
    training_decks = tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    )
    if training_decks != FOCAL_DECK_IDS:
        raise RuntimeError("V3 focal pool is not exact 001-067")
    required = (
        PARENT_CHECKPOINT, PARENT_PFSP_STATE, RULES,
        DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V3 launch artifacts missing: {missing}")
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V3 formal version path is already used: {version_root}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if parent.get("schema_version") != "0043_focal_v1_model_only_v1":
        raise RuntimeError("V3 parent checkpoint schema changed")
    if parent.get("update") != START_UPDATE:
        raise RuntimeError("V3 parent checkpoint is not U207")
    state = parent["state_dict"]
    if state["value_head.heads.archetype.1.weight"].shape != (15, 320):
        raise RuntimeError("V3 parent opponent Meta head changed")
    own_keys = (
        "value_adapter.own_embedding.weight",
        "policy_strategy_adapter.own_embedding.weight",
    )
    if any(state[key].shape != (29, 16) for key in own_keys):
        raise RuntimeError("V3 parent own-deck embedding shape changed")
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    if {row.deck_id for row in vocabulary.mappings} != set(FOCAL_DECK_IDS):
        raise RuntimeError("V3 own-archetype mapping does not cover 001-067")
    return {
        "schema_version": "0043_v3_generalist_launch_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "long_training_started": False,
        "launch_token_required": "--launch-formal",
        "project": "0043_champion_league_rl",
        "version": VERSION,
        "parent": {
            "version": "V2_mixed_focal_cohorts",
            "checkpoint_update": START_UPDATE,
            "path": str(PARENT_CHECKPOINT.relative_to(ROOT)),
            "sha256": sha256_file(PARENT_CHECKPOINT),
            "pfsp_state": str(PARENT_PFSP_STATE.relative_to(ROOT)),
            "pfsp_state_sha256": sha256_file(PARENT_PFSP_STATE),
        },
        "focal": {
            "deck_ids": list(FOCAL_DECK_IDS),
            "lanes_per_rollout": 256,
            "per_deck_games": [3, 4],
            "schedule": "seeded_frequency_balanced_random_v1",
            "resident_cohort_key": "opponent_policy_id_only",
        },
        "opponents": {
            "deck_ids": list(FOCAL_DECK_IDS),
            "policy_ids": list(registry.active_policy_ids),
        },
        "optimizer": {"initialization": "fresh", "ppo_contract": "unchanged_from_v2"},
        "reference_anchor_update": 0,
        "own_taxonomy": {"classes": 29, "embedding_width": 16},
        "opponent_meta": {"classes": 15, "trainable": False},
        "asset_audit": asdict(asset_audit),
        "cuda_engine": {
            "version": "2.0",
            "extension_sha256": sha256_file(DEFAULT_BUILD_DIR / "_ptcg_cuda.so"),
        },
        "wandb": {
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument(
        "--updates", type=int, default=None,
        help="absolute checkpoint update to stop at; omitted means continuous training",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--readiness-output", type=Path)
    args = parser.parse_args()
    if not args.launch_formal:
        result = readiness()
        if args.readiness_output:
            args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
            args.readiness_output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    readiness()
    run(
        updates=args.updates,
        wandb_mode=args.wandb_mode,
        launch_formal=True,
        version=VERSION,
        start_update=START_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT,
        parent_pfsp_state=PARENT_PFSP_STATE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0043 · V3 generalist focal 001-067",
        focal_deck_ids=FOCAL_DECK_IDS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
