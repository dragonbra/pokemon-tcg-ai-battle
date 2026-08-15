"""0044 V4: immutable G2 seed, uniform G2 opponents, 0042-V1 actor LR."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..own_archetype import OwnArchetypeVocabulary
from ..policy import load_actor_critic
from .run_v1 import FOCAL_DECK_ID, PROJECT_ROOT, ROOT, RULES, _cards, _config, run


VERSION = "V4_g2_uniform_0042_v1_lr_benchmark_v2"
START_UPDATE = 0
FOCAL_DECK_IDS = (FOCAL_DECK_ID,)
OPPONENT_DECK_IDS = tuple(f"{index:03d}" for index in range(1, 68))
OPPONENT_POLICY_IDS = ("Champion-G2",)
OPPONENT_SAMPLING_MODE = "uniform_001_067"
ACTOR_LEARNING_RATE_SCALE = 0.5
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
G2_PORTABLE = PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin"
G2_SOURCE = PROJECT_ROOT / "assets/policies/definitions/champion_g002/source_update_000407.pt"
WANDB_RUN_ID = "0044-v4-g2-uniform-0042-v1-lr-benchmark-v2"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = registry.validate_all()
    training_decks = tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    )
    if training_decks != OPPONENT_DECK_IDS:
        raise RuntimeError("V4 opponent deck pool is not exact 001-067")
    required = (G2_PORTABLE, G2_SOURCE, RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V4 launch artifacts missing: {missing}")
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V4 formal version path is already used: {version_root}")

    source = torch.load(G2_SOURCE, map_location="cpu", weights_only=True)
    if source.get("schema_version") != "0043_focal_v1_model_only_v1" or source.get("update") != 407:
        raise RuntimeError("V4 source is not immutable Champion-G2 U407")
    model, identity = load_actor_critic(
        deck=_cards(registry, FOCAL_DECK_ID), deck_id=FOCAL_DECK_ID
    )
    if not model.policy_option_lora.is_zero_delta():
        raise RuntimeError("V4 G2 seed does not have zero-delta Option LoRA")
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    if {row.deck_id for row in vocabulary.mappings} != set(OPPONENT_DECK_IDS):
        raise RuntimeError("V4 own-archetype mapping does not cover 001-067")
    ppo = asdict(_config(actor_learning_rate_scale=ACTOR_LEARNING_RATE_SCALE))
    expected = {
        "decoder_learning_rate": 5e-6,
        "policy_adapter_learning_rate": 5e-6,
        "allocation_learning_rate": 5e-6,
        "option_lora_learning_rate": 1e-5,
        "value_learning_rate": 1e-4,
        "prize_learning_rate": 1e-4,
    }
    if any(ppo[key] != value for key, value in expected.items()):
        raise RuntimeError("V4 optimizer does not match the approved low-LR contract")
    return {
        "schema_version": "0044_v4_launch_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "long_training_started": False,
        "project": "0044_g2_dragapult_policy_option_lora",
        "version": VERSION,
        "initialization": {
            "policy_id": "Champion-G2", "checkpoint_update": 407,
            "portable_sha256": sha256_file(G2_PORTABLE),
            "source_sha256": sha256_file(G2_SOURCE),
            "effective_source_sha256": identity.checkpoint_sha256,
            "option_lora": "rank4_alpha8_zero_delta",
            "optimizer": "fresh",
            "parent_training_checkpoint": None,
            "parent_pfsp_state": None,
        },
        "focal": {"deck_ids": [FOCAL_DECK_ID]},
        "opponent": {
            "policy_ids": list(OPPONENT_POLICY_IDS),
            "deck_ids": ["001", "067"], "deck_count": 67,
            "sampling": {"mode": OPPONENT_SAMPLING_MODE, "pfsp": 0, "uniform": 256},
        },
        "optimizer": expected,
        "benchmark_v2": {
            "baseline_checkpoint_update": 0,
            "interval_updates": 10, "games": 2048,
            "opponent_policy_id": "Policy-0809",
            "common_random_numbers": True,
        },
        "asset_audit": asdict(asset_audit),
        "cuda_engine": {
            "version": "2.0",
            "extension_sha256": sha256_file(DEFAULT_BUILD_DIR / "_ptcg_cuda.so"),
        },
        "wandb": {
            "entity": "dragon_bra", "project": "pokemon-tcg-policy-learning",
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
    result = readiness()
    if not args.launch_formal:
        if args.readiness_output:
            args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
            args.readiness_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    run(
        updates=args.updates, wandb_mode=args.wandb_mode, launch_formal=True,
        version=VERSION, start_update=START_UPDATE,
        parent_checkpoint=None, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=START_UPDATE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V4 G2 uniform · 0042 V1 LR · Benchmark V2",
        focal_deck_ids=FOCAL_DECK_IDS,
        opponent_sampling_mode=OPPONENT_SAMPLING_MODE,
        actor_learning_rate_scale=ACTOR_LEARNING_RATE_SCALE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
