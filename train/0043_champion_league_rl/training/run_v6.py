"""V6 restart of generalist 001-067 training from the last complete V5 U233."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..own_archetype import OwnArchetypeVocabulary
from .run_v1 import (
    PROJECT_ROOT,
    ROOT,
    RULES,
    _paths,
    _pristine_restart_allowed,
    run,
)


VERSION = "V6_generalist_focal_001_067_u233_restart"
START_UPDATE = 233
FOCAL_DECK_IDS = tuple(f"{index:03d}" for index in range(1, 68))
PARENT_ROOT = (
    ROOT
    / "rl_runs/0043_champion_league_rl/versions"
    / "V5_generalist_focal_001_067_u208_micro512"
)
PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint/update-000233.pt"
PARENT_PFSP_STATE = PARENT_ROOT / "artifact/pfsp_state.json"
VERSION_ROOT = ROOT / "rl_runs/0043_champion_league_rl/versions" / VERSION
WANDB_RUN_ID = "0043-v6-generalist-focal-001-067-u233-restart"
FORWARD_MICROBATCH_SIZE = 512


def _pfsp_max_source_update(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [
        *payload.get("deck_records", {}).values(),
        *payload.get("policy_records", {}).values(),
        *payload.get("joint_records", {}).values(),
    ]
    if not records:
        raise RuntimeError("V6 parent PFSP state has no committed observations")
    return max(int(record["last_updated"]) for record in records)


def readiness(
    *, version_root: Path = VERSION_ROOT, allow_pristine_restart: bool = False,
) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = registry.validate_all()
    training_decks = tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    )
    if training_decks != FOCAL_DECK_IDS:
        raise RuntimeError("V6 focal pool is not exact 001-067")
    required = (
        PARENT_CHECKPOINT,
        PARENT_PFSP_STATE,
        RULES,
        DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V6 launch artifacts missing: {missing}")
    version_used = version_root.exists() and any(version_root.rglob("*"))
    pristine_restart = (
        allow_pristine_restart
        and version_root.resolve() == VERSION_ROOT.resolve()
        and _pristine_restart_allowed(
            _paths(VERSION),
            start_update=START_UPDATE,
            parent_checkpoint=PARENT_CHECKPOINT,
            focal_deck_ids=FOCAL_DECK_IDS,
        )
    )
    if version_used and not pristine_restart:
        raise FileExistsError(f"V6 formal version path is already used: {version_root}")

    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if parent.get("schema_version") != "0043_focal_v1_model_only_v1":
        raise RuntimeError("V6 parent checkpoint schema changed")
    if parent.get("update") != START_UPDATE:
        raise RuntimeError("V6 parent checkpoint is not U233")
    state = parent["state_dict"]
    if state["value_head.heads.archetype.1.weight"].shape != (15, 320):
        raise RuntimeError("V6 parent opponent Meta head changed")
    own_keys = (
        "value_adapter.own_embedding.weight",
        "policy_strategy_adapter.own_embedding.weight",
    )
    if any(state[key].shape != (29, 16) for key in own_keys):
        raise RuntimeError("V6 parent own-deck embedding shape changed")
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    if {row.deck_id for row in vocabulary.mappings} != set(FOCAL_DECK_IDS):
        raise RuntimeError("V6 own-archetype mapping does not cover 001-067")
    pfsp_max_source_update = _pfsp_max_source_update(PARENT_PFSP_STATE)
    if pfsp_max_source_update != START_UPDATE - 1:
        raise RuntimeError(
            "V6 parent PFSP boundary is not the committed source-U232 state"
        )

    return {
        "schema_version": "0043_v6_u233_restart_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "asset_audit": asdict(asset_audit),
        "parent": {
            "version": "V5_generalist_focal_001_067_u208_micro512",
            "checkpoint_update": START_UPDATE,
            "path": str(PARENT_CHECKPOINT.relative_to(ROOT)),
            "sha256": sha256_file(PARENT_CHECKPOINT),
            "pfsp_state": str(PARENT_PFSP_STATE.relative_to(ROOT)),
            "pfsp_state_sha256": sha256_file(PARENT_PFSP_STATE),
            "pfsp_max_source_update": pfsp_max_source_update,
            "pfsp_boundary": (
                "committed through source-U232; interrupted source-U233 rollout "
                "is intentionally re-collected for U234"
            ),
        },
        "focal": {
            "deck_ids": list(FOCAL_DECK_IDS),
            "lanes_per_rollout": 256,
            "schedule": "seeded_frequency_balanced_random_v1",
            "resident_cohort_key": "opponent_policy_id_only",
        },
        "opponents": {
            "deck_ids": list(FOCAL_DECK_IDS),
            "policy_ids": list(registry.active_policy_ids),
        },
        "optimizer": {
            "initialization": "fresh",
            "logical_minibatch_size": 2048,
            "forward_microbatch_size": FORWARD_MICROBATCH_SIZE,
            "epochs": 3,
            "semantic_change": False,
        },
        "reference_anchor_update": 0,
        "own_taxonomy": {"classes": 29, "embedding_width": 16},
        "opponent_meta": {"classes": 15, "trainable": False},
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
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--readiness-output", type=Path)
    args = parser.parse_args()
    result = readiness(allow_pristine_restart=args.launch_formal)
    if not args.launch_formal:
        if args.readiness_output:
            args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
            args.readiness_output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    run(
        updates=args.updates,
        wandb_mode=args.wandb_mode,
        launch_formal=True,
        version=VERSION,
        start_update=START_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT,
        parent_pfsp_state=PARENT_PFSP_STATE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0043 · V6 generalist 001-067 · U233 restart",
        focal_deck_ids=FOCAL_DECK_IDS,
        allow_pristine_restart=True,
        forward_microbatch_size=FORWARD_MICROBATCH_SIZE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
