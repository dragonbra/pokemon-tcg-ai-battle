"""Warm-start only the Phantom allocation head from Zero-Shot sequential labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import Tensor

from ..action_boundary.dragapult import StableTargetIdentity, enumerate_allocations
from ..action_boundary.macro_planner import MacroPlanner
from ..initialization import build_update0_model
from ..policy.batching import collate_feature_batches, move_batch
from .run_full_semantic import focal_deck
from .storage_full_semantic import save_model_only

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "rl_runs/0038_action_boundary_rl/datasets/allocation_zero_shot_v1/allocation_samples.pt"
VERSION = "V2_zero_shot_action_boundary_update0"
VERSION_ROOT = ROOT / "rl_runs/0038_action_boundary_rl/versions" / VERSION


def _repository_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _encoded_examples(model, samples: list[dict], device: torch.device, batch_size: int = 64):
    examples = []
    planner = MacroPlanner(model.allocation_head)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            rows = samples[start:start + batch_size]
            features = move_batch(
                collate_feature_batches([item["pre_action_features"] for item in rows]), device
            )
            validated, state, options = model.actor.encode(features)
            for row, sample in enumerate(rows):
                identities = tuple(StableTargetIdentity(**item) for item in sample["target_ids"])
                allocations = enumerate_allocations(identities)
                label = next(index for index, item in enumerate(allocations)
                             if item.counters == tuple(sample["counters"]))
                embeddings, visible = [], []
                for target in identities:
                    match = (
                        validated.card_mask[row]
                        & validated.card_cat[row, :, 1].eq(target.serial + 1)
                        & validated.card_cat[row, :, 2].eq(2)
                        & validated.card_cat[row, :, 3].eq(6)
                    ).nonzero(as_tuple=False).flatten()
                    if match.numel() != 1:
                        raise ValueError(f"BC target serial {target.serial} is not uniquely visible")
                    position = int(match[0])
                    cat, numeric = validated.card_cat[row, position], validated.card_num[row, position]
                    embeddings.append(state.cards[row, position].cpu())
                    visible.append({
                        "serial": target.serial, "id": int(cat[0]),
                        "benchSlot": target.initial_bench_slot,
                        "hp": float(numeric[0]), "maxHp": float(numeric[1]),
                        "energyCards": [None] * int(numeric[2]),
                        "preEvolution": [None] * int(numeric[5]), "statusBits": int(cat[6]),
                    })
                allocation_features = planner.visible_features(visible, allocations)
                examples.append({
                    "split": sample["split"], "n": len(identities),
                    "state": state.summary[row].cpu(),
                    "root": options[row, int(sample["root_index"])].cpu(),
                    "targets": torch.stack(embeddings), "features": allocation_features,
                    "label": label, "counters": tuple(sample["counters"]),
                    "allocations": tuple(item.counters for item in allocations),
                })
    return examples


def _batch_logits(model, rows: list[dict], device: torch.device) -> Tensor:
    state = torch.stack([item["state"] for item in rows]).to(device)
    root = torch.stack([item["root"] for item in rows]).to(device)
    targets = torch.stack([item["targets"] for item in rows]).to(device)
    features = torch.stack([item["features"] for item in rows]).to(device)
    batch, allocations, target_count = features.shape[:3]
    expanded = targets[:, None].expand(-1, allocations, -1, -1)
    return model.allocation_head(
        state, root, expanded, features,
        torch.ones((batch, allocations, target_count), dtype=torch.bool, device=device),
        torch.ones((batch, allocations), dtype=torch.bool, device=device),
    )


def _evaluate(model, rows: list[dict], device: torch.device, batch_size: int,
              seen_allocations: set[tuple[int, tuple[int, ...]]] | None = None) -> dict[str, float]:
    losses = correct = count = absolute = entropy = prize_teacher = prize_chosen = 0.0
    unseen = 0
    model.eval()
    with torch.inference_mode():
        by_n = defaultdict(list)
        for row in rows:
            by_n[row["n"]].append(row)
        for group in by_n.values():
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
                logits = _batch_logits(model, batch, device)
                labels = torch.tensor([item["label"] for item in batch], device=device)
                distribution = torch.distributions.Categorical(logits=logits.float())
                chosen = logits.argmax(-1)
                losses += float((-distribution.log_prob(labels)).sum())
                entropy += float(distribution.entropy().sum())
                correct += int(chosen.eq(labels).sum())
                for index, item in enumerate(batch):
                    if (seen_allocations is not None
                            and (item["n"], item["counters"]) not in seen_allocations):
                        unseen += 1
                    predicted = item["allocations"][int(chosen[index])]
                    absolute += sum(abs(a - b) for a, b in zip(predicted, item["counters"])) / item["n"]
                    teacher_features = item["features"][item["label"]]
                    chosen_features = item["features"][int(chosen[index])]
                    prize_teacher += float((teacher_features[:, 2] * teacher_features[:, 3]).sum())
                    prize_chosen += float((chosen_features[:, 2] * chosen_features[:, 3]).sum())
                count += len(batch)
    return {"nll": losses / count, "top1_accuracy": correct / count,
            "allocation_count_mae": absolute / count, "entropy": entropy / count,
            "teacher_immediate_prizes": prize_teacher / count,
            "chosen_immediate_prizes": prize_chosen / count,
            "samples": count, "unseen_allocation_count": unseen,
            "unseen_allocation_rate": unseen / count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if VERSION_ROOT.exists() and any(VERSION_ROOT.rglob("*")):
        raise FileExistsError(f"update-0 version already exists: {VERSION_ROOT}")
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
    samples = payload["samples"]
    device = torch.device(args.device)
    model, identity = build_update0_model(focal_deck(), device=device)
    examples = _encoded_examples(model, samples, device)
    train = [item for item in examples if item["split"] == "train" and item["n"] > 1]
    validation = [item for item in examples if item["split"] == "validation" and item["n"] > 1]
    if not train or not validation:
        raise RuntimeError("allocation BC requires non-forced train and validation battles")
    optimizer = torch.optim.AdamW(model.allocation_head.parameters(), lr=args.learning_rate)
    best_state = None
    best_nll = float("inf")
    stale = 0
    history = []
    rng = random.Random(380_038_101)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.allocation_head.train()
        groups = defaultdict(list)
        for row in train:
            groups[row["n"]].append(row)
        batches = []
        for group in groups.values():
            rng.shuffle(group)
            batches.extend(group[start:start + args.batch_size]
                           for start in range(0, len(group), args.batch_size))
        rng.shuffle(batches)
        for batch in batches:
            logits = _batch_logits(model, batch, device)
            labels = torch.tensor([item["label"] for item in batch], device=device)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.allocation_head.parameters(), 1.0)
            optimizer.step()
        metrics = _evaluate(model, validation, device, args.batch_size)
        history.append({"epoch": epoch, **metrics})
        if metrics["nll"] < best_nll - 1e-5:
            best_nll = metrics["nll"]
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.allocation_head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.allocation_head.load_state_dict(best_state, strict=True)
    seen_allocations = {(item["n"], item["counters"]) for item in train}
    train_metrics = _evaluate(model, train, device, args.batch_size)
    validation_metrics = _evaluate(
        model, validation, device, args.batch_size, seen_allocations
    )
    checkpoint = VERSION_ROOT / "checkpoint/update-000000.pt"
    digest = save_model_only(model, checkpoint, update=0, metadata={
        "version": VERSION, "initialization": "zero_shot_pre_rl_no_v5",
        "allocation_bc_dataset_sha256": dataset_hash,
        "allocation_bc_best_validation_nll": best_nll,
        "ppo_updates": 0,
    })
    artifact = VERSION_ROOT / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "0038_allocation_bc_training_v1", "version": VERSION,
        "source_identity": identity.__dict__ if hasattr(identity, "__dict__") else {
            name: getattr(identity, name) for name in identity.__slots__},
        "dataset": _repository_relative(args.dataset), "dataset_sha256": dataset_hash,
        "checkpoint": _repository_relative(checkpoint), "checkpoint_sha256": digest,
        "epochs_completed": len(history), "best_validation_nll": best_nll,
        "train": train_metrics, "validation": validation_metrics, "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "contract": "allocation-only BC; no PPO; optimizer/scheduler/RNG/buffer not saved",
    }
    (artifact / "allocation_bc_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (artifact / "status.json").write_text(json.dumps({
        "status": "update0_ready", "ppo_updates": 0, "wandb": "not_applicable_data_preparation",
        "checkpoint_sha256": digest,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint": str(checkpoint), "sha256": digest,
                      "validation": validation_metrics, "epochs": len(history)}, indent=2))


if __name__ == "__main__":
    main()
