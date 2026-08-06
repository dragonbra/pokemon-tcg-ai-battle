"""One-update official-engine PPO canary without allocating a formal version."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .ppo import PPOTrainer
from .rollout import ResidentRolloutCollector
from .runtime import build_runtime


def _state_hash(model: torch.nn.Module, *, trainable: bool) -> str:
    digest = hashlib.sha256()
    parameters = dict(model.named_parameters())
    for name, value in sorted(model.state_dict().items()):
        parameter = parameters.get(name)
        is_trainable = parameter is not None and (
            name.startswith("action_decoder.") or name.startswith("value_head.")
        )
        if is_trainable != trainable:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=152)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    shared = args.foundation_checkpoint is not None
    runtime = build_runtime(
        rules=args.rules,
        extension_dir=args.extension_dir,
        source_checkpoint=args.source_checkpoint,
        support_report=args.support_report,
        frozen_root=args.frozen_root,
        checkpoint_root=args.checkpoint_root,
        shared_foundation_checkpoint=args.foundation_checkpoint,
        copies_per_opponent=args.lanes // 38 if shared else args.lanes,
        opponents_per_cohort=38 if shared else 1,
    )
    frozen_before = _state_hash(runtime.learner, trainable=False)
    trainable_before = _state_hash(runtime.learner, trainable=True)
    collector = ResidentRolloutCollector(
        engine=runtime.engine,
        policy=runtime.policy,
        decks=runtime.decks,
        seeds=runtime.seeds,
        opponent_head_indices=runtime.opponent_head_indices,
        opponent_deck_ids=runtime.opponent_deck_ids,
        steps=args.steps,
        actor_execution="cuda_graph",
    )
    trainer = PPOTrainer(runtime.learner, device=torch.device("cuda"))
    update_rows = []
    for update in range(args.updates):
        rollout = collector.collect(source_policy_update=update)
        ppo_started = time.perf_counter()
        ppo = trainer.update(rollout.batch)
        torch.cuda.synchronize()
        ppo_seconds = time.perf_counter() - ppo_started
        update_rows.append(
            {
                "update": update + 1,
                "rollout": rollout.metrics,
                "ppo": ppo,
                "ppo_seconds": ppo_seconds,
                "end_to_end_update_decisions_per_second": (
                    rollout.metrics["rollout/engine_decisions"]
                    / (rollout.metrics["rollout/wall_seconds"] + ppo_seconds)
                ),
                "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_reserved_bytes": torch.cuda.memory_reserved(),
            }
        )
    total_seconds = time.perf_counter() - started
    frozen_after = _state_hash(runtime.learner, trainable=False)
    trainable_after = _state_hash(runtime.learner, trainable=True)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen encoder changed during PPO canary")
    if trainable_before == trainable_after:
        raise RuntimeError("decoder/value parameters did not change during PPO canary")
    payload = {
        "schema": "0034_official_cuda_ppo_canary_v1",
        "passed": True,
        "opponents": list(runtime.base_opponent_deck_ids),
        "lanes": args.lanes,
        "steps": args.steps,
        "updates": update_rows,
        "total_seconds_including_initialization": total_seconds,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "frozen_state_unchanged": True,
        "trainable_state_sha256_before": trainable_before,
        "trainable_state_sha256_after": trainable_after,
        "trainable_state_changed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
