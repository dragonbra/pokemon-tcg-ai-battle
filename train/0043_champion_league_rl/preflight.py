"""CPU-first hard gate and optional small GPU tensor-isolation smoke."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .assets import AssetRegistry
from .league.pfsp import PFSPConfig, PFSPState
from .league.sampler import BRANCH_COUNTS, build_schedule
from .policy_identity import assert_storage_isolation, materialize_policy_bundle
from .runtime import audit_runtime_tree, forward_parity
from .evaluation.schedule import materialize as materialize_frozen_schedule
from .training.config import audit_training_config


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
CONFIG = REPOSITORY_ROOT / "experiments/0043_champion_league_rl/active_training_config.json"
LEAGUE_CONFIG = PROJECT_ROOT / "league/config.json"


def _pool_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def run_preflight(*, gpu_smoke: bool = False) -> dict[str, Any]:
    assets = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = assets.validate_all()
    training = audit_training_config(CONFIG)
    anchor = materialize_policy_bundle(PROJECT_ROOT, "Policy-0809", purpose="0043_preflight")
    champion = materialize_policy_bundle(PROJECT_ROOT, "Champion-G1", purpose="0043_preflight")
    assert_storage_isolation(anchor, champion)
    deck_ids = [deck.deck_id for deck in assets.decks if "training" in deck.roles]
    policy_ids = [policy.policy_id for policy in assets.policies]
    state = PFSPState()
    curriculum = state.curriculum_for(
        0, deck_ids=deck_ids, policy_ids=policy_ids,
        config=PFSPConfig.load(LEAGUE_CONFIG), training_deck_pool_hash=_pool_hash(deck_ids),
        active_policy_pool_hash=_pool_hash(policy_ids),
    )
    schedule = build_schedule(
        update=0, seed=430043001, deck_ids=deck_ids, policy_ids=policy_ids,
        latest_champion_policy_id=assets.latest_champion_policy_id,
        curriculum=curriculum,
    )
    gpu: dict[str, Any] = {"requested": gpu_smoke, "status": "NOT_RUN"}
    runtime_tree_sha256 = audit_runtime_tree()
    forward = [
        asdict(forward_parity("Policy-0809", deck_id="001", gpu=gpu_smoke)),
        asdict(forward_parity("Champion-G1", deck_id="048", gpu=gpu_smoke)),
    ]
    if gpu_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU smoke requested but CUDA is unavailable")
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        # Move representative tensors from both identities, including the
        # full G1 portable bundle. This is an identity/storage smoke, not an
        # official-engine or model-forward strength result.
        anchor_sample = {
            name: tensor for index, (name, tensor) in enumerate(anchor.tensors.items())
            if index < 16
        }
        anchor_sample_bundle = type(anchor)(anchor.policy_id, anchor_sample, anchor.audit)
        gpu_anchor = anchor_sample_bundle.clone_to("cuda:0", runtime_dtype=torch.float32)
        gpu_champion = champion.clone_to("cuda:0", runtime_dtype=torch.float32)
        assert_storage_isolation(gpu_anchor, gpu_champion)
        floating = [
            tensor for bundle in (gpu_anchor, gpu_champion)
            for tensor in bundle.tensors.values() if torch.is_floating_point(tensor)
        ]
        checksum = sum(float(tensor.reshape(-1)[:1].float().sum().item()) for tensor in floating)
        torch.cuda.synchronize()
        gpu = {
            "requested": True,
            "status": "PASS",
            "device": torch.cuda.get_device_name(0),
            "runtime_dtype": "fp32",
            "anchor_sample_tensors": len(gpu_anchor.tensors),
            "champion_tensors": len(gpu_champion.tensors),
            "storage_isolation": "PASS",
            "finite_checksum": bool(torch.isfinite(torch.tensor(checksum))),
            "allocated_bytes_delta": torch.cuda.memory_allocated() - before,
            "scope": "tensor materialization plus synthetic forward parity; no official games or strength claim",
        }
    frozen_cpu = materialize_frozen_schedule(
        PROJECT_ROOT, focal_deck_id="048",
        focal_deployment_identity=champion.audit.effective_policy_sha256, replicas=1,
    )
    frozen_cuda = materialize_frozen_schedule(
        PROJECT_ROOT, focal_deck_id="048",
        focal_deployment_identity=champion.audit.effective_policy_sha256, replicas=8,
    )
    return {
        "schema_version": "0043_pretraining_preflight_v1",
        "status": "PASS",
        "formal_training_authorized": False,
        "asset_audit": asdict(asset_audit),
        "training_regression_audit": training.to_manifest(),
        "policy_audits": [asdict(anchor.audit), asdict(champion.audit)],
        "policy_storage_isolation": "PASS",
        "semantic_runtime": {"status": "PASS", "tree_sha256": runtime_tree_sha256},
        "forward_parity": forward,
        "curriculum": {"version": curriculum.curriculum_version, "hash": curriculum.content_hash},
        "schedule": {
            "games": len(schedule),
            "branches": dict(Counter(lane.branch for lane in schedule)),
            "first": sum(lane.focal_goes_first for lane in schedule),
            "second": sum(not lane.focal_goes_first for lane in schedule),
        },
        "gpu_smoke": gpu,
        "frozen_schedules": {
            "cpu_games": len(frozen_cpu["jobs"]),
            "cuda_games": len(frozen_cuda["jobs"]),
            "base_schedule_sha256": frozen_cpu["base_schedule_sha256"],
            "cpu_schedule_sha256": frozen_cpu["schedule_sha256"],
            "cuda_schedule_sha256": frozen_cuda["schedule_sha256"],
            "frequency_unit_parity": frozen_cpu["jobs"] == frozen_cuda["jobs"][:256],
        },
        "remaining_hard_gates": [
            "official CPU engine smoke",
            "official observation CPU/CUDA first-divergence parity",
            "formal PPO version allocation and W&B preflight",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_preflight(gpu_smoke=args.gpu_smoke)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
