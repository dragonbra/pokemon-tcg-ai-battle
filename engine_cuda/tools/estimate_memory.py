from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.memory import estimate_memory  # noqa: E402
from ptcg_cuda_engine.policy_pool import PolicyPoolManifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate CUDA engine and resident policy-pool VRAM.")
    parser.add_argument("--envs", type=int, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort-capacity", type=int)
    parser.add_argument("--learner-rollout-kib-per-env", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = PolicyPoolManifest.load(args.manifest)
    estimates = [
        estimate_memory(
            envs,
            manifest,
            workspace_root=WORKSPACE_ROOT,
            cohort_capacity=args.cohort_capacity,
            learner_rollout_bytes_per_env=int(args.learner_rollout_kib_per_env * 1024),
        ).to_dict()
        for envs in args.envs
    ]
    print(json.dumps({"manifest": manifest.name, "estimates": estimates}, indent=2))


if __name__ == "__main__":
    main()
