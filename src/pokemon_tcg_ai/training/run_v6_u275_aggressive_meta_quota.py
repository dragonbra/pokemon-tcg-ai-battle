"""0045 V6: U275 aggressive 02/03/05 quotas against Policy-0809."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..evaluation.run_benchmark_tiny_v2_three_pool import validate_report
from ..league.aggressive_meta_quota import aggressive_meta_quota_schedule
from ..own_archetype import OwnArchetypeVocabulary
from .lr_profiles import EXPERT_COLD_START_LR_PROFILE
from .run_v1 import PROJECT_ROOT, ROOT, run


VERSION = "V6_dragapult_007_u275_aggressive_meta_quota_policy0809"
VERSION_ROOT = ROOT / "runs/versions" / VERSION
START_UPDATE = 275
PARENT_VERSION = "V4_dragapult_007_u200_policy0809_three_pool"
PARENT_CHECKPOINT = (
    ROOT / "runs/versions"
    / PARENT_VERSION / "checkpoint/update-000275.pt"
)
PARENT_CHECKPOINT_SHA256 = "ed063401e16ff96cd7d3cf3e744876e97368bef843e4de00b4c13fe41d09ecdb"
BASELINE_REPORT = (
    ROOT / "runs/versions"
    / PARENT_VERSION / "artifact/periodic_evaluation/update-000275/report.json"
)
REFERENCE_CHECKPOINT = (
    ROOT / "runs/versions"
    / "V1_minimal_lora_dragapult_007/checkpoint/update-000000.pt"
)
REFERENCE_CHECKPOINT_SHA256 = "3c13e085b4587a8112148b393c4b80c4c875efbc13820dcf44acb6a0a1991bc7"
FOCAL_DECK_ID = "007"
ROLLOUT_GAMES = 512
OPPONENT_POLICY_IDS = ("Policy-0809",)
OPPONENT_SAMPLING_MODE = "aggressive_meta_quota_training_pool"
OPPONENT_META_QUOTAS = {2: 50, 3: 200, 5: 200}
PERIODIC_EVALUATION_PROFILE = "policy0809_three_pool_cuda512"
PERIODIC_EVALUATION_INTERVAL_UPDATES = 1
WANDB_RUN_ID = "0045-v6-dragapult-007-u275-aggressive-meta-quota-policy0809"


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V6 formal version path is already used: {version_root}")
    for path, digest in (
        (PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256),
        (REFERENCE_CHECKPOINT, REFERENCE_CHECKPOINT_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"V6 checkpoint identity failed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    reference = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        parent.get("schema_version") != "0045_minimal_lora_model_only_v1"
        or parent.get("update") != START_UPDATE
        or parent.get("metadata", {}).get("version") != PARENT_VERSION
    ):
        raise RuntimeError("V6 parent is not exact V4 U275")
    if reference.get("update") != 0:
        raise RuntimeError("V6 reference is not Frozen-0045-Init")
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    validate_report(baseline)
    if (
        baseline.get("focal_checkpoint_update") != START_UPDATE
        or baseline["focal_policy_identity_audit"].get("source_checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("V6 baseline report is not exact V4 U275")

    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    schedule = aggressive_meta_quota_schedule(
        quota_seed=440_120_000 + START_UPDATE,
        shuffle_seed=440_120_200 + START_UPDATE,
        mappings=vocabulary.mappings,
        lanes=ROLLOUT_GAMES,
        fixed_meta_quotas=OPPONENT_META_QUOTAS,
    )
    counts = Counter(row.archetype_id for row in schedule)
    remaining = {
        meta_id: games for meta_id, games in sorted(counts.items())
        if meta_id not in OPPONENT_META_QUOTAS
    }
    return {
        "schema_version": "0045_v6_u275_aggressive_meta_quota_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION,
        "parent": {
            "version": PARENT_VERSION,
            "update": START_UPDATE,
            "sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer": "fresh",
            "on_policy_data": "fresh",
        },
        "reference": {
            "policy_id": "Frozen-0045-Init",
            "update": 0,
            "sha256": REFERENCE_CHECKPOINT_SHA256,
        },
        "rollout": {
            "games": ROLLOUT_GAMES,
            "opponent_policy_ids": list(OPPONENT_POLICY_IDS),
            "sampling_mode": OPPONENT_SAMPLING_MODE,
            "fixed_meta_quotas": OPPONENT_META_QUOTAS,
            "remaining_meta_total": sum(remaining.values()),
            "remaining_meta_quotas": remaining,
        },
        "periodic_evaluation": {
            "profile": PERIODIC_EVALUATION_PROFILE,
            "interval_updates": PERIODIC_EVALUATION_INTERVAL_UPDATES,
            "opponent_policy_id": "Policy-0809",
            "games_per_pool": 512,
            "pool_count": 3,
            "games_per_update": 1536,
            "baseline_report": str(BASELINE_REPORT),
            "baseline_report_sha256": sha256_file(BASELINE_REPORT),
            "baseline_deployment_effective_sha256": baseline[
                "focal_deployment_effective_sha256"
            ],
        },
        "learning_rate_profile": EXPERT_COLD_START_LR_PROFILE.metadata(),
        "update_limit": None,
        "checkpoint_retention": "all",
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
        start_update=START_UPDATE,
        parent_checkpoint=PARENT_CHECKPOINT,
        reference_checkpoint=REFERENCE_CHECKPOINT,
        baseline_evaluation_checkpoint=None,
        baseline_evaluation_provenance=BASELINE_REPORT,
        periodic_evaluation_enabled=True,
        periodic_evaluation_profile=PERIODIC_EVALUATION_PROFILE,
        periodic_evaluation_interval_updates=PERIODIC_EVALUATION_INTERVAL_UPDATES,
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0045 · V6 · U275 aggressive 03/05/02 · Policy-0809",
        focal_deck_ids=(FOCAL_DECK_ID,),
        focal_deck_id=FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        evaluation_focal_deck_id=FOCAL_DECK_ID,
        source_parent_version=PARENT_VERSION,
        source_parent_update=START_UPDATE,
        reference_anchor_update=0,
        reference_anchor_identity="Frozen-0045-Init",
        opponent_sampling_mode=OPPONENT_SAMPLING_MODE,
        opponent_policy_ids=OPPONENT_POLICY_IDS,
        latest_champion_policy_id="Policy-0809",
        rollout_games=ROLLOUT_GAMES,
        opponent_meta_quotas=OPPONENT_META_QUOTAS,
        learning_rate_profile=EXPERT_COLD_START_LR_PROFILE,
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
    payload = readiness()
    if args.readiness_output:
        args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
        args.readiness_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
