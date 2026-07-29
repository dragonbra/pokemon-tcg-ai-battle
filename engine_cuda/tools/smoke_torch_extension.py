from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.native import create_engine, pack_reset_specs  # noqa: E402
from ptcg_cuda_engine.reference import ReferenceResetSpec  # noqa: E402
from ptcg_cuda_engine.schema import RulePack  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the optional PyTorch CUDA binding.")
    parser.add_argument("--extension-dir", type=Path, default=CUDA_ENGINE_ROOT / "build" / "torch")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--policies", type=int, default=12)
    return parser.parse_args()


def reset_rows(batch: int, policies: int) -> list[ReferenceResetSpec]:
    return [
        ReferenceResetSpec(
            seed=12345 + env * 17,
            active_card_ids=(100 + env % 1000, 200 + env % 1000),
            active_hp=(120, 120),
            policy_ids=(env % policies, (env + 1) % policies),
        )
        for env in range(batch)
    ]


def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.steps <= 0 or not 1 <= args.policies <= 32:
        raise SystemExit("invalid batch, steps, or policies")
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required")
    pack = RulePack.load(CUDA_ENGINE_ROOT / "rules" / "smoke_rules.json")
    engine = create_engine(
        pack,
        batch_size=args.batch,
        policy_count=args.policies,
        route_capacity=args.batch,
    )
    reset_bytes = bytearray(pack_reset_specs(reset_rows(args.batch, args.policies)))
    reset_tensor = torch.frombuffer(reset_bytes, dtype=torch.uint8)
    engine.reset(reset_tensor)
    actions = torch.zeros(args.batch, dtype=torch.int32, device="cuda")

    for _ in range(5):
        engine.step(actions)
        engine.encode_policy_v1()
        engine.route_ready()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps):
        engine.step(actions)
        encoded = engine.encode_policy_v1()
        routes, route_counts = engine.route_ready()
    stop.record()
    stop.synchronize()
    elapsed_ms = start.elapsed_time(stop)
    digest = engine.digest()
    torch.cuda.synchronize()

    normal_error = int(encoded["global_cat"][0, 6].item())
    routed = int(route_counts.to(torch.int64).sum().item())
    first_digest = int(digest[0].item()) & ((1 << 64) - 1)
    if routed != args.batch or normal_error != 0:
        raise RuntimeError("normal PyTorch binding path failed validation")
    if tuple(routes.shape) != (args.policies, args.batch):
        raise RuntimeError("route tensor shape is wrong")

    engine.reset(reset_tensor)
    actions.fill_(4)
    engine.step(actions)
    known = engine.encode_policy_v1()
    known_error = int(known["global_cat"][0, 6].item())
    if known_error != 6601207:
        raise RuntimeError(f"known divergence returned {known_error}, expected 6601207")

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "batch": args.batch,
                "policies": args.policies,
                "steps": args.steps,
                "elapsed_ms": round(elapsed_ms, 3),
                "env_steps_per_second": round(args.batch * args.steps * 1000.0 / elapsed_ms, 3),
                "allocated_mib": round(engine.allocated_bytes / 1024**2, 3),
                "last_route_count": routed,
                "first_digest": first_digest,
                "known_divergence_error": known_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
