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
from .cuda_engine_2.build import DEFAULT_BUILD_DIR, smoke_extension
from .cuda_engine_2.identity import CudaEngineIdentity
from .cuda_engine_2.routing import materialize_lane_requests


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
CONFIG = REPOSITORY_ROOT / "experiments/0043_champion_league_rl/active_training_config.json"
LEAGUE_CONFIG = PROJECT_ROOT / "league/config.json"
CUDA_RULE_PACK = REPOSITORY_ROOT / ".tmp/cuda_0032_rules/official_rules.bin"


def _pool_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def run_preflight(*, gpu_smoke: bool = False) -> dict[str, Any]:
    assets = AssetRegistry.load(PROJECT_ROOT)
    asset_audit = assets.validate_all()
    training = audit_training_config(CONFIG)
    bundles = [
        materialize_policy_bundle(PROJECT_ROOT, policy_id, purpose="0043_preflight")
        for policy_id in assets.active_policy_ids
    ]
    assert_storage_isolation(*bundles)
    deck_ids = [deck.deck_id for deck in assets.decks if "training" in deck.roles]
    policy_ids = [
        policy.policy_id for policy in assets.policies
        if policy.frozen and policy.role in {"historical_anchor", "champion", "latest_champion"}
    ]
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
    cuda_requests, cuda_schedule_sha256 = materialize_lane_requests(PROJECT_ROOT, schedule)
    gpu: dict[str, Any] = {"requested": gpu_smoke, "status": "NOT_RUN"}
    runtime_tree_sha256 = audit_runtime_tree()
    forward = [asdict(forward_parity(policy_id, deck_id="048", gpu=gpu_smoke))
               for policy_id in assets.active_policy_ids]
    if gpu_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU smoke requested but CUDA is unavailable")
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        # Move representative tensors from both identities, including the
        # full G1 portable bundle. This is an identity/storage smoke, not an
        # official-engine or model-forward strength result.
        anchor = bundles[0]
        anchor_sample = {
            name: tensor for index, (name, tensor) in enumerate(anchor.tensors.items())
            if index < 16
        }
        anchor_sample_bundle = type(anchor)(anchor.policy_id, anchor_sample, anchor.audit)
        gpu_anchor = anchor_sample_bundle.clone_to("cuda:0", runtime_dtype=torch.float32)
        gpu_champions = [
            bundle.clone_to("cuda:0", runtime_dtype=torch.float32)
            for bundle in bundles[1:]
        ]
        assert_storage_isolation(gpu_anchor, *gpu_champions)
        floating = [
            tensor for bundle in (gpu_anchor, *gpu_champions)
            for tensor in bundle.tensors.values() if torch.is_floating_point(tensor)
        ]
        checksum = sum(float(tensor.reshape(-1)[:1].float().sum().item()) for tensor in floating)
        torch.cuda.synchronize()
        native_binary = DEFAULT_BUILD_DIR / "ptcg_cuda_smoke"
        extension = DEFAULT_BUILD_DIR / "_ptcg_cuda.so"
        engine_identity = CudaEngineIdentity.resolve(
            REPOSITORY_ROOT, rule_pack=CUDA_RULE_PACK,
            binary=native_binary, extension=extension,
            require_gpu=True, require_extension=True,
        )
        engine_smoke = smoke_extension(CUDA_RULE_PACK, DEFAULT_BUILD_DIR)
        gpu = {
            "requested": True,
            "status": "PASS",
            "device": torch.cuda.get_device_name(0),
            "runtime_dtype": "fp32",
            "anchor_sample_tensors": len(gpu_anchor.tensors),
            "champion_tensors": sum(len(bundle.tensors) for bundle in gpu_champions),
            "storage_isolation": "PASS",
            "finite_checksum": bool(torch.isfinite(torch.tensor(checksum))),
            "allocated_bytes_delta": torch.cuda.memory_allocated() - before,
            "cuda_engine_2_identity": engine_identity.to_manifest(),
            "cuda_engine_2_smoke": engine_smoke,
            "scope": "tensor materialization, synthetic forward parity and official CUDA runtime reset/classify; no complete official games or strength claim",
        }
    frozen_cpu = materialize_frozen_schedule(
        PROJECT_ROOT, focal_deck_id="048",
        focal_deployment_identity=bundles[-1].audit.effective_policy_sha256, replicas=1,
    )
    frozen_cuda = materialize_frozen_schedule(
        PROJECT_ROOT, focal_deck_id="048",
        focal_deployment_identity=bundles[-1].audit.effective_policy_sha256, replicas=8,
    )
    return {
        "schema_version": "0043_pretraining_preflight_v1",
        "status": "PASS",
        "readiness": "READY_AWAITING_USER_LAUNCH",
        "formal_training_authorized": False,
        "asset_audit": asdict(asset_audit),
        "training_regression_audit": training.to_manifest(),
        "policy_audits": [asdict(bundle.audit) for bundle in bundles],
        "policy_storage_isolation": "PASS",
        "semantic_runtime": {"status": "PASS", "tree_sha256": runtime_tree_sha256},
        "forward_parity": forward,
        "curriculum": {"version": curriculum.curriculum_version, "hash": curriculum.content_hash},
        "schedule": {
            "games": len(schedule),
            "branches": dict(Counter(lane.branch for lane in schedule)),
            "first": sum(lane.focal_goes_first for lane in schedule),
            "second": sum(not lane.focal_goes_first for lane in schedule),
            "cuda_request_schedule_sha256": cuda_schedule_sha256,
            "cuda_request_count": len(cuda_requests),
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
        "remaining_hard_gates": [],
        "launch_latch": "explicit user authorization via --launch-formal",
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
