from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.native import create_engine, pack_reset_specs  # noqa: E402
from ptcg_cuda_engine.policy_pool import PolicyPoolManifest, PolicySpec  # noqa: E402
from ptcg_cuda_engine.reference import ReferenceResetSpec  # noqa: E402
from ptcg_cuda_engine.schema import RulePack  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the actual heterogeneous policy pool and probe CUDA residency."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CUDA_ENGINE_ROOT / "configs" / "policy_pool.example.json",
    )
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=CUDA_ENGINE_ROOT / "build" / "torch",
    )
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--engine-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--no-learner-training-reserve",
        action="store_true",
        help="Do not allocate gradient, FP32 master, and two Adam moment mirrors.",
    )
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source_dir = str(path.parent.resolve())
    sibling_names = {
        sibling.stem
        for sibling in path.parent.glob("*.py")
        if sibling.stem != path.stem
    }
    saved_siblings = {
        sibling_name: sys.modules.pop(sibling_name)
        for sibling_name in sibling_names
        if sibling_name in sys.modules
    }
    sys.path.insert(0, source_dir)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        try:
            sys.path.remove(source_dir)
        except ValueError:
            pass
        for sibling_name in sibling_names:
            sys.modules.pop(sibling_name, None)
        sys.modules.update(saved_siblings)
    return module


def checkpoint_state(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("state_dict", "model_state_dict", "policy_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError(f"checkpoint has no supported state dict key: {sorted(payload)}")


def load_idonly_model(policy: PolicySpec, checkpoint: Path, torch):
    source = checkpoint.parent / "idonly_policy.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing bundled model source: {source}")
    module_name = f"_cuda_residency_{policy.policy_id}_{checkpoint.parent.name}"
    module = load_module(source, module_name)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"{checkpoint} has no model_config")
    config = module.ModelConfig(**raw_config)
    model = module.IDOnlyPointerPolicy(config)
    model.load_state_dict(checkpoint_state(payload), strict=True)
    return model


def load_alakazam_model(policy: PolicySpec, checkpoint: Path):
    strategy_dir = checkpoint.parent
    agent_dir = strategy_dir.parent
    package_name = f"_cuda_residency_alakazam_{policy.policy_id}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(strategy_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    inference = importlib.import_module(f"{package_name}.r2_inference")
    deck = [
        int(line)
        for line in (agent_dir / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    aliases = {1264: 1266}
    model_deck = [aliases.get(card_id, card_id) for card_id in deck]
    runtime = inference.R2Policy.from_checkpoint(
        checkpoint,
        strategy_dir / "card_ontology.json",
        model_deck,
    )
    return runtime.model


def load_model(policy: PolicySpec, torch):
    checkpoint = Path(policy.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = WORKSPACE_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if policy.adapter == "alakazam_independent_v1":
        return load_alakazam_model(policy, checkpoint)
    return load_idonly_model(policy, checkpoint, torch)


def tensor_bytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def model_bytes(model) -> tuple[int, int]:
    parameters = sum(tensor_bytes(parameter) for parameter in model.parameters())
    buffers = sum(tensor_bytes(buffer) for buffer in model.buffers())
    return parameters, buffers


def memory_snapshot(torch, device) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "torch_allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 3),
        "torch_reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 3),
        "device_free_mib": round(free_bytes / 1024**2, 3),
        "device_used_mib": round((total_bytes - free_bytes) / 1024**2, 3),
        "device_total_mib": round(total_bytes / 1024**2, 3),
    }


def learner_training_reserve(model, torch, device) -> list[Any]:
    reserve: list[Any] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        reserve.append(torch.empty_like(parameter, device=device))
        for _ in range(3):
            reserve.append(
                torch.empty(parameter.shape, dtype=torch.float32, device=device)
            )
    return reserve


def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.engine_steps <= 0:
        raise SystemExit("batch and engine steps must be positive")
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required")
    device = torch.device(args.device)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    manifest = PolicyPoolManifest.load(args.manifest)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = memory_snapshot(torch, device)

    resident_models: list[Any] = []
    rows: list[dict[str, Any]] = []
    learner = None
    for policy in manifest.policies:
        model = load_model(policy, torch)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(not policy.frozen)
        model.to(device=device, dtype=dtype)
        torch.cuda.synchronize(device)
        parameter_bytes, buffer_bytes = model_bytes(model)
        rows.append(
            {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "adapter": policy.adapter,
                "frozen": policy.frozen,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "device_parameter_mib": round(parameter_bytes / 1024**2, 3),
                "device_buffer_mib": round(buffer_bytes / 1024**2, 3),
                "device_used_mib_after_load": memory_snapshot(torch, device)["device_used_mib"],
            }
        )
        resident_models.append(model)
        if not policy.frozen:
            learner = model

    if manifest.frozen_policy_count < 10 or learner is None:
        raise RuntimeError("manifest did not load 10+ frozen policies and one learner")
    reserve: list[Any] = []
    if not args.no_learner_training_reserve:
        reserve = learner_training_reserve(learner, torch, device)
        torch.cuda.synchronize(device)
    after_models = memory_snapshot(torch, device)

    pack = RulePack.load(CUDA_ENGINE_ROOT / "rules" / "smoke_rules.json")
    engine = create_engine(
        pack,
        batch_size=args.batch,
        policy_count=len(manifest.policies),
        route_capacity=args.batch,
        device_index=device.index or 0,
    )
    specs = [
        ReferenceResetSpec(
            seed=1000 + env,
            active_card_ids=(100 + env % 1000, 200 + env % 1000),
            active_hp=(120, 120),
            policy_ids=(env % len(manifest.policies), (env + 1) % len(manifest.policies)),
        )
        for env in range(args.batch)
    ]
    reset_storage = bytearray(pack_reset_specs(specs))
    engine.reset(torch.frombuffer(reset_storage, dtype=torch.uint8))
    actions = torch.zeros(args.batch, dtype=torch.int32, device=device)
    torch.cuda.nvtx.range_push("decision_hot_path_engine_codec_route")
    try:
        for _ in range(args.engine_steps):
            engine.step(actions)
            engine.encode_policy_v1()
            _, route_counts = engine.route_ready()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    routed = int(route_counts.to(torch.int64).sum().item())
    if routed != args.batch:
        raise RuntimeError(f"engine routed {routed} of {args.batch} environments")
    final = memory_snapshot(torch, device)

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "dtype": args.dtype,
                "frozen_policies": manifest.frozen_policy_count,
                "learner_policies": sum(not policy.frozen for policy in manifest.policies),
                "learner_training_reserve": not args.no_learner_training_reserve,
                "learner_reserve_mib": round(
                    sum(tensor_bytes(tensor) for tensor in reserve) / 1024**2, 3
                ),
                "engine_batch": args.batch,
                "engine_allocated_mib": round(engine.allocated_bytes / 1024**2, 3),
                "engine_steps": args.engine_steps,
                "last_route_count": routed,
                "baseline": baseline,
                "after_models": after_models,
                "final": final,
                "torch_peak_allocated_mib": round(
                    torch.cuda.max_memory_allocated(device) / 1024**2, 3
                ),
                "models": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
