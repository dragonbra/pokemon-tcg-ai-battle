"""V9: restart V8 U20 as local U0 with focal exact deck 066."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from . import run_v4 as contract
from .run_v1 import PROJECT_ROOT, ROOT, RULES, _config, run


VERSION = "V9_u20_deck066_core16_benchmark_v2"
LOCAL_START_UPDATE = 0
SOURCE_PARENT_UPDATE = 20
PARENT_VERSION = "V8_u33_deck003_core16_benchmark_v2"
PARENT_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint/update-000020.pt"
PARENT_CHECKPOINT_SHA256 = "1ebe9f66df8adbcf38d922c39cd8a3971ca6459f823551c54695270a5086e7ac"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
WANDB_RUN_ID = "0044-v9-u20-deck066-core16-benchmark-v2"
FOCAL_DECK_ID = "066"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    audit = registry.validate_all()
    required = (PARENT_CHECKPOINT, RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V9 launch artifacts missing: {missing}")
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V9 formal version path is already used: {version_root}")
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V9 parent checkpoint hash changed")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V9 parent is not exact V8 U20")
    ppo = asdict(_config(actor_learning_rate_scale=contract.ACTOR_LEARNING_RATE_SCALE))
    if (
        ppo["decoder_learning_rate"] != 5e-6
        or ppo["policy_adapter_learning_rate"] != 5e-6
        or ppo["allocation_learning_rate"] != 5e-6
        or ppo["option_lora_learning_rate"] != 1e-5
    ):
        raise RuntimeError("V9 low-LR contract changed")
    return {
        "schema_version": "0044_v9_launch_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH", "long_training_started": False,
        "version": VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "source_checkpoint_update": SOURCE_PARENT_UPDATE,
            "local_start_update": LOCAL_START_UPDATE,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer": "fresh", "pfsp_state": None,
        },
        "training": {
            "focal_deck_ids": [FOCAL_DECK_ID],
            "opponent_policy_ids": ["Champion-G2"],
            "opponent_deck_ids": ["001", "067"], "opponent_deck_count": 67,
            "sampling": {"mode": "uniform_001_067", "pfsp": 0, "uniform": 256},
            "ppo": ppo,
        },
        "benchmark_v2": {
            "contract_id": "0044_benchmark_v2_core16_meta_balanced_policy0809_common_seeds_cuda2048_v2",
            "baseline_checkpoint_update": LOCAL_START_UPDATE,
            "focal_deck_id": FOCAL_DECK_ID,
            "selected_meta_ids": list(range(14)) + [17, 27],
            "games_per_meta": 128, "games": 2048,
            "opponent_policy_id": "Policy-0809", "common_random_numbers": True,
        },
        "asset_audit": asdict(audit),
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
            args.readiness_output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    run(
        updates=args.updates, wandb_mode=args.wandb_mode, launch_formal=True,
        version=VERSION, start_update=LOCAL_START_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=LOCAL_START_UPDATE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V9 V8-U20 → local-U0 · Deck 066 · Core-16",
        focal_deck_ids=(FOCAL_DECK_ID,), focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        opponent_sampling_mode="uniform_001_067",
        actor_learning_rate_scale=contract.ACTOR_LEARNING_RATE_SCALE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
