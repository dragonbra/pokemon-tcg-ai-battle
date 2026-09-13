"""V6 continuation from U4 under the final Benchmark V2 Core-16 contract."""

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


VERSION = "V6_u4_core16_benchmark_v2"
START_UPDATE = 4
PARENT_VERSION = "V5_g2_uniform_0042_v1_lr_benchmark_v2"
PARENT_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint/update-000004.pt"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
WANDB_RUN_ID = "0044-v6-u4-core16-benchmark-v2"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    audit = registry.validate_all()
    required = (PARENT_CHECKPOINT, RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V6 launch artifacts missing: {missing}")
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V6 formal version path is already used: {version_root}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if parent.get("schema_version") != "0044_focal_v1_model_only_v1" or parent.get("update") != START_UPDATE:
        raise RuntimeError("V6 parent is not the exact V5 U4 model-only checkpoint")
    ppo = asdict(_config(actor_learning_rate_scale=contract.ACTOR_LEARNING_RATE_SCALE))
    if (
        ppo["decoder_learning_rate"] != 5e-6
        or ppo["policy_adapter_learning_rate"] != 5e-6
        or ppo["allocation_learning_rate"] != 5e-6
        or ppo["option_lora_learning_rate"] != 1e-5
    ):
        raise RuntimeError("V6 low-LR contract changed")
    return {
        "schema_version": "0044_v6_launch_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH", "long_training_started": False,
        "version": VERSION,
        "parent": {
            "version": PARENT_VERSION, "checkpoint_update": START_UPDATE,
            "checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
            "optimizer": "fresh", "pfsp_state": None,
        },
        "training": {
            "focal_deck_ids": ["007"], "opponent_policy_ids": ["Champion-G2"],
            "opponent_deck_ids": ["001", "067"], "opponent_deck_count": 67,
            "sampling": {"mode": "uniform_001_067", "pfsp": 0, "uniform": 256},
            "ppo": ppo,
        },
        "benchmark_v2": {
            "contract_id": "0044_benchmark_v2_core16_meta_balanced_policy0809_common_seeds_cuda2048_v2",
            "baseline_checkpoint_update": START_UPDATE,
            "selected_meta_ids": list(range(14)) + [17, 27],
            "games_per_meta": 128, "games": 2048,
            "opponent_policy_id": "Policy-0809", "common_random_numbers": True,
        },
        "asset_audit": asdict(audit),
        "cuda_engine": {"version": "2.0", "extension_sha256": sha256_file(DEFAULT_BUILD_DIR / "_ptcg_cuda.so")},
        "wandb": {"entity": "dragon_bra", "project": "pokemon-tcg-policy-learning", "run_id": WANDB_RUN_ID},
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
        parent_checkpoint=PARENT_CHECKPOINT, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=START_UPDATE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V6 U4 · Core-16 Benchmark V2",
        focal_deck_ids=("007",), opponent_sampling_mode="uniform_001_067",
        actor_learning_rate_scale=contract.ACTOR_LEARNING_RATE_SCALE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
