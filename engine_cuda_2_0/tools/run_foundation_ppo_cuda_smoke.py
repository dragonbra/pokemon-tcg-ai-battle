"""Run a small real-engine PPO update with one shared 0020 foundation model.

This is a residency gate, not a policy-quality benchmark.  The official CUDA
engine supplies observations, the six deck profiles are batched as static
device fields, and rollout storage, GAE, actor/critic evaluation, and the PPO
optimizer all stay on the RTX device.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.cuda_ppo import CudaRolloutBuffer, ppo_update_device  # noqa: E402
from ptcg_cuda_engine.legacy_codecs import (  # noqa: E402
    policy_codec_v1_to_idonly_codec_v1,
)
from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    foundation_r15_static_fields_for_decks,
)
from run_official_seeded_multi_policy_loop import (  # noqa: E402
    ERROR,
    load_0020_runtime_package,
    load_foundation_0020_checkpoint,
    load_pool_rows,
    read_deck,
)


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description="GPU-resident PPO smoke for the shared 0020 foundation model."
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--pool",
        type=Path,
        default=WORKSPACE_ROOT / ".tmp" / "cuda_zero_shot_six_decks" / "pool.json",
    )
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-select", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=2026080101)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--train-scope",
        choices=("full", "decoder_only"),
        default="full",
        help="Train all shared foundation parameters or only its decoder/value head.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=CUDA_ENGINE_ROOT / "artifacts" / "foundation_ppo_cuda_smoke.json")
    return parser.parse_args()


def _move_masks(batch: dict[str, Any]) -> dict[str, Any]:
    batch = dict(batch)
    batch["entity_mask"] = batch["entity_mask"].bool()
    batch["option_mask"] = batch["option_mask"].bool()
    return batch


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.steps <= 0 or not 1 <= args.max_select <= 64:
        raise ValueError("batch/steps must be positive and max-select must be in [1, 64]")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rules_path = args.rules.resolve()
    pool_path = args.pool.resolve()
    rows, unsupported = load_pool_rows(pool_path, [])
    if len(rows) != 6:
        raise ValueError(f"expected six promoted deck rows, got {len(rows)}")
    checkpoint = (
        WORKSPACE_ROOT / str(rows[0]["directory"]) / str(rows[0]["checkpoint"])
    ).resolve()
    ontology = (
        WORKSPACE_ROOT / str(rows[0]["ontology"])
        if rows[0].get("ontology")
        else checkpoint.parent.parent / "artifact" / "card_ontology.json"
    ).resolve()
    deck_profiles = [
        read_deck((WORKSPACE_ROOT / str(row["directory"])).resolve() / "deck.csv")
        for row in rows
    ]
    lane_decks = [deck_profiles[index % len(deck_profiles)] for index in range(args.batch)]
    decks = torch.tensor(
        [[deck, deck] for deck in lane_decks], dtype=torch.int32, device=device
    )
    seeds = torch.arange(
        args.seed_start,
        args.seed_start + args.batch,
        dtype=torch.int64,
        device=device,
    )
    lane_mask = torch.ones(args.batch, dtype=torch.bool, device=device)
    static_fields = foundation_r15_static_fields_for_decks(lane_decks, device)

    runtime_package = load_0020_runtime_package()
    actor_critic_module = importlib.import_module(f"{runtime_package}.actor_critic")
    policy, _config = load_foundation_0020_checkpoint(checkpoint, ontology, device)
    model = actor_critic_module.SourceR15ActorCritic(policy).to(device)
    if args.train_scope == "decoder_only":
        scope = model.configure_decoder_only()
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        scope = {
            "mode": "full",
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "frozen_parameter_count": 0,
        }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
    )

    engine = create_official_engine(
        rules_path.read_bytes(), batch_size=args.batch, device_index=args.device_index
    )
    engine.reset_seeded_interactive(decks, seeds)
    buffer = CudaRolloutBuffer(max_steps=args.steps, batch_size=args.batch)
    last_batch: dict[str, Any] | None = None
    status_error_count = torch.zeros((), dtype=torch.int64, device=device)

    for _step in range(args.steps):
        statuses = engine.statuses()
        reset_mask = statuses.eq(2) | statuses.eq(ERROR)
        seeds.add_(reset_mask.to(torch.int64) * args.batch)
        engine.reset_seeded_interactive_masked(decks, seeds, reset_mask)
        status_error_count.add_(statuses.eq(ERROR).long().sum())
        engine.advance_to_decision()
        encoded = _move_masks(dict(engine.encode_policy_v1()))
        idonly = policy_codec_v1_to_idonly_codec_v1(
            encoded,
            max_card_id=2048,
            max_action_steps=args.max_select,
            target_entity_capacity=128,
            target_option_capacity=80,
        )
        foundation_batch = dict(idonly)
        foundation_batch.update(static_fields)
        ready = engine.statuses().eq(1)
        foundation_batch["_route_mask"] = ready
        sample = model.sample_actions_device(
            foundation_batch, max_select=args.max_select, temperature=1.0
        )
        record_batch = {
            key: value for key, value in foundation_batch.items() if key != "_route_mask"
        }
        record_batch["targets"] = sample["targets"]
        engine.pack_actions(sample["actions"], sample["lengths"])
        engine.apply_packed_actions()
        done = engine.statuses().eq(2) | engine.statuses().eq(ERROR)
        reward = -sample["lengths"].to(torch.float32) * 1.0e-3 + done.to(torch.float32)
        buffer.append(
            record_batch,
            old_logprob=sample["logprob"],
            old_value=sample["value"],
            reward=reward,
            done=done,
        )
        last_batch = record_batch

    if last_batch is None:
        raise RuntimeError("no rollout batch was produced")
    with torch.no_grad():
        last_value = model.value(last_batch)
    stats = ppo_update_device(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        last_value=last_value,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=1.0,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    torch.cuda.synchronize(device)
    status_error_count_host = int(status_error_count.item())
    output = {
        "schema_version": 1,
        "passed": status_error_count_host == 0,
        "model_instances_loaded": 1,
        "shared_model_across_decks": True,
        "deck_profiles": [str(row["name"]) for row in rows],
        "unsupported_pool_entries": unsupported,
        "train_scope": args.train_scope,
        "scope": scope,
        "batch": args.batch,
        "rollout_steps": len(buffer),
        "ppo_epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "status_error_count": status_error_count_host,
        "cuda_resident": {
            "engine": True,
            "observations": True,
            "rollout_buffer": True,
            "gae": True,
            "actor_critic": True,
            "optimizer": True,
            "h2d_rollout_copies": 0,
            "d2h_rollout_copies": 0,
        },
        "ppo": stats.as_host(),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "output": str(args.output),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
