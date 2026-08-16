"""V14: clean restart of V13 after admitting Policy-0814 at the PPO identity gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..policy.actor_critic import DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT
from ..policy_identity import materialize_policy_bundle
from . import run_v13_policy0814_shared_encoder as contract
from .run_v1 import PROJECT_ROOT, ROOT, RULES, _atomic_torch, _checkpoint, run


VERSION = "V14_policy0814_identity_gate_fix"
VERSION_ROOT = ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions" / VERSION
U0_CHECKPOINT = VERSION_ROOT / "checkpoint/update-000000.pt"
WANDB_RUN_ID = "0045-v14-policy0814-identity-gate-fix"
FAILED_VERSION = contract.VERSION


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def readiness() -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = registry.validate_all()
    required = (
        RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
        DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V14 launch artifacts missing: {missing}")
    if VERSION_ROOT.exists() and any(VERSION_ROOT.rglob("*")):
        raise FileExistsError(f"V14 formal version path is already used: {VERSION_ROOT}")
    failed_status = (
        ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions"
        / FAILED_VERSION / "artifact/status.json"
    )
    if not failed_status.is_file() or json.loads(
        failed_status.read_text(encoding="utf-8")
    ).get("state") != "failed":
        raise RuntimeError("V14 requires preserved failed V13 provenance")
    bundle = materialize_policy_bundle(
        PROJECT_ROOT, contract.OPPONENT_POLICY_ID, purpose="0045_v14_readiness"
    )
    model, source, audit = contract._model_and_audit()
    del model
    return {
        "schema_version": "0045_v14_policy0814_identity_gate_fix_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH", "version": VERSION,
        "failed_predecessor": {
            "version": FAILED_VERSION,
            "failure": "PPO identity consumer omitted registered Policy-0814",
            "checkpoint_update": 0, "optimizer_updates": 0,
        },
        "source": asdict(source), "trainable": audit,
        "opponent": {
            "policy_id": contract.OPPONENT_POLICY_ID,
            "effective_policy_sha256": bundle.audit.effective_policy_sha256,
        },
        "rollout": {
            "games": 512, "lanes": 512,
            "fixed_deck_quotas": contract.FIXED_DECK_QUOTAS,
            "random_remainder_deck_ids": list(contract.RANDOM_DECK_IDS),
        },
        "eval": {"every_updates": 5, "games": 512, "outputs": ["summary", "per_deck"]},
        "asset_audit": asdict(asset_audit),
        "source_hashes": {
            "actor": sha256_file(DEFAULT_0814_ACTOR_CHECKPOINT),
            "value": sha256_file(DEFAULT_0814_VALUE_CHECKPOINT),
        },
        "wandb": {
            "entity": "dragon_bra", "project": "pokemon-tcg-policy-learning",
            "run_id": WANDB_RUN_ID,
        },
    }


def prepare_u0() -> None:
    evidence = readiness()
    model, _, audit = contract._model_and_audit()
    for name in ("artifact", "checkpoint", "tensorboard", "wandb"):
        (VERSION_ROOT / name).mkdir(parents=True, exist_ok=True)
    _atomic_torch(U0_CHECKPOINT, _checkpoint(
        model, 0, version=VERSION, focal_deck_id=contract.FOCAL_DECK_ID,
        focal_deck_ids=(contract.FOCAL_DECK_ID,), focal_schedule_mode="fixed",
        source_parent_version="Policy-0814-pretrained",
        source_parent_update=None, parent_policy_id=contract.OPPONENT_POLICY_ID,
        parent_policy_update=None,
    ))
    _atomic_json(VERSION_ROOT / "artifact/trainable_tensor_audit.json", audit)
    _atomic_json(VERSION_ROOT / "artifact/readiness.json", evidence)


def launch(*, wandb_mode: str = "online", updates: int | None = None) -> None:
    prepare_u0()
    run(
        updates=updates, wandb_mode=wandb_mode, launch_formal=True,
        version=VERSION, start_update=0, parent_checkpoint=U0_CHECKPOINT,
        reference_checkpoint=U0_CHECKPOINT, parent_pfsp_state=None,
        baseline_evaluation_checkpoint=None, periodic_evaluation_enabled=True,
        periodic_evaluation_interval_updates=5,
        periodic_evaluation_profile="policy0814_exact_deck_cuda512",
        wandb_run_id=WANDB_RUN_ID,
        wandb_name="0045 · V14 · Policy-0814 identity gate fix",
        focal_deck_ids=(contract.FOCAL_DECK_ID,),
        focal_deck_id=contract.FOCAL_DECK_ID,
        evaluation_focal_deck_id=contract.FOCAL_DECK_ID,
        source_parent_version="Policy-0814-pretrained", source_parent_update=None,
        reference_anchor_update=None, reference_anchor_identity="Policy-0814-V14-U0",
        opponent_sampling_mode="exact_deck_quota_training_pool",
        opponent_policy_ids=(contract.OPPONENT_POLICY_ID,),
        latest_champion_policy_id=contract.OPPONENT_POLICY_ID,
        rollout_games=contract.ROLLOUT_GAMES,
        opponent_deck_quotas=contract.FIXED_DECK_QUOTAS,
        opponent_random_deck_ids=contract.RANDOM_DECK_IDS,
        ppo_batch_size=contract.PPO_BATCH_SIZE,
        forward_microbatch_size=contract.FORWARD_MICROBATCH_SIZE,
        behavior_probe_batch_size=contract.BEHAVIOR_PROBE_BATCH_SIZE,
        offload_reference_after_cache=True,
        focal_base_checkpoint=DEFAULT_0814_ACTOR_CHECKPOINT,
        focal_value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT,
        adaptation_config=contract.ADAPTATION,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    if args.launch_formal:
        launch(wandb_mode=args.wandb_mode, updates=args.updates)
    else:
        print(json.dumps(readiness(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
