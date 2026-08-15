"""V10/G3: V9 U50 to an indefinite 001-067 focal generalist run."""

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


VERSION = "V10_g3_v9_u50_generalist_001_067_core16_benchmark_v2"
LOCAL_START_UPDATE = 0
SOURCE_PARENT_UPDATE = 50
PARENT_VERSION = "V9_u20_deck066_core16_benchmark_v2"
PARENT_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / PARENT_VERSION
PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint/update-000050.pt"
HANDOFF_MANIFEST = PARENT_ROOT / "artifact/g3_handoff_u50.json"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
WANDB_RUN_ID = "0044-v10-g3-v9-u50-generalist-001-067-core16-benchmark-v2"
FOCAL_DECK_IDS = tuple(f"{index:03d}" for index in range(1, 68))
INITIALIZATION_DECK_ID = "066"
EVALUATION_SENTINEL_DECK_ID = "066"
FOCAL_SCHEDULE = "seeded_frequency_balanced_random_v1"


def readiness(
    *, version_root: Path = VERSION_ROOT,
    parent_checkpoint: Path = PARENT_CHECKPOINT,
    handoff_manifest: Path = HANDOFF_MANIFEST,
) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    audit = registry.validate_all()
    training_decks = tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    )
    if training_decks != FOCAL_DECK_IDS:
        raise RuntimeError("V10 focal pool is not exact ordered 001-067")
    required = (parent_checkpoint, handoff_manifest, RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V10 launch artifacts missing: {missing}")
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V10 formal version path is already used: {version_root}")

    handoff = json.loads(handoff_manifest.read_text(encoding="utf-8"))
    parent_sha256 = sha256_file(parent_checkpoint)
    benchmark_report = Path(handoff.get("benchmark_v2", {}).get("report", ""))
    if (
        handoff.get("schema_version") != "0044_v9_u50_g3_handoff_v1"
        or handoff.get("status") != "PASS"
        or handoff.get("parent_version") != PARENT_VERSION
        or handoff.get("parent_checkpoint_update") != SOURCE_PARENT_UPDATE
        or handoff.get("parent_checkpoint_sha256") != parent_sha256
        or handoff.get("benchmark_v2", {}).get("status") != "PASS"
        or handoff.get("benchmark_v2", {}).get("checkpoint_update") != SOURCE_PARENT_UPDATE
        or not benchmark_report.is_file()
        or handoff.get("benchmark_v2", {}).get("report_sha256") != sha256_file(benchmark_report)
    ):
        raise RuntimeError("V10 handoff manifest does not bind exact V9 U50 evidence")
    parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != SOURCE_PARENT_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V10 parent is not exact V9 U50")

    ppo = asdict(_config(actor_learning_rate_scale=contract.ACTOR_LEARNING_RATE_SCALE))
    return {
        "schema_version": "0044_v10_g3_launch_readiness_v1",
        "status": "READY_AWAITING_AUTOMATIC_HANDOFF",
        "version": VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "source_checkpoint_update": SOURCE_PARENT_UPDATE,
            "local_start_update": LOCAL_START_UPDATE,
            "checkpoint_sha256": parent_sha256,
            "optimizer": "fresh",
            "pfsp_state": None,
        },
        "training": {
            "duration": "indefinite",
            "focal_deck_ids": list(FOCAL_DECK_IDS),
            "focal_schedule": FOCAL_SCHEDULE,
            "focal_initialization_deck_id": INITIALIZATION_DECK_ID,
            "opponent_policy_ids": ["Champion-G2"],
            "opponent_deck_ids": ["001", "067"],
            "opponent_deck_count": 67,
            "sampling": {"mode": "uniform_001_067", "pfsp": 0, "uniform": 256},
            "ppo": ppo,
        },
        "benchmark_v2": {
            "baseline_checkpoint_update": 0,
            "focal_deck_id": EVALUATION_SENTINEL_DECK_ID,
            "role": "longitudinal_sentinel_not_all_deck_aggregate",
            "games": 2048,
        },
        "asset_audit": asdict(audit),
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


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    readiness()
    run(
        updates=updates,
        wandb_mode=wandb_mode,
        launch_formal=True,
        version=VERSION,
        start_update=LOCAL_START_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT,
        parent_pfsp_state=None,
        baseline_evaluation_checkpoint=LOCAL_START_UPDATE,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0044 · V10 G3 · V9-U50 → 001–067 focal generalist · Core-16",
        focal_deck_ids=FOCAL_DECK_IDS,
        focal_deck_id=INITIALIZATION_DECK_ID,
        focal_schedule_mode=FOCAL_SCHEDULE,
        evaluation_focal_deck_id=EVALUATION_SENTINEL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=SOURCE_PARENT_UPDATE,
        opponent_sampling_mode="uniform_001_067",
        actor_learning_rate_scale=contract.ACTOR_LEARNING_RATE_SCALE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int, default=None)
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
