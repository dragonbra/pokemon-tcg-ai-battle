from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from rl_environment.streaming import chunks

from .config import load_config
from .data import collate_records, iter_split_records, model_inputs
from .train import _device, _load_source, _move, _sha256


Decision = dict[str, Any]
Predicate = Callable[[Decision], bool]


def _positive(field: str) -> Predicate:
    return lambda decision: float(decision.get(field, 0.0)) > 0.0


def _negative(field: str) -> Predicate:
    return lambda decision: float(decision.get(field, 0.0)) < 0.0


SLICE_PREDICATES: dict[str, Predicate] = {
    "opening_active_abra_positive": _positive("opening_active_abra_credit"),
    "opening_active_abra_negative": _negative("opening_active_abra_credit"),
    "second_turn_setup_positive": _positive("second_turn_setup_credit"),
    "second_turn_setup_negative": _negative("second_turn_setup_credit"),
    "bridge_first_turn_positive": _positive("dunsparce_first_turn_setup_credit"),
    "bridge_first_turn_negative": _negative("dunsparce_first_turn_setup_credit"),
    "bridge_execution_positive": _positive("dunsparce_second_turn_execution_credit"),
    "bridge_execution_negative": _negative("dunsparce_second_turn_execution_credit"),
    "attack_prize_positive": _positive("attack_prize_delta"),
    "non_prize_attack": _positive("non_prize_attack"),
    "powerful_hand_non_prize_attack": _positive("powerful_hand_non_prize_attack"),
    "post_ko_relay_positive": _positive("post_ko_relay_credit"),
    "post_ko_relay_negative": _negative("post_ko_relay_credit"),
    "recoverable_discard_miss": _positive("recoverable_discard_miss"),
}


@torch.no_grad()
def evaluate_slices(
    dataset: Path,
    reward_sidecar: Path,
    checkpoint: Path,
    config_path: Path,
    *,
    device_name: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    device = _device(device_name)
    model, _ = _load_source(checkpoint, config, device)
    model.eval()
    totals = {name: {"records": 0, "exact": 0} for name in SLICE_PREDICATES}
    overall_records = 0
    overall_exact = 0
    records = iter_split_records(dataset, "validation", reward_sidecar=reward_sidecar)
    for chunk in chunks(records, config.batch_size):
        batch = _move(collate_records(chunk), device)
        _, logits, count_logits = model.forward_with_count(**model_inputs(batch))
        positions = torch.arange(count_logits.shape[1], device=device).unsqueeze(0)
        maximum = batch["selection_max_count"].clamp_max(count_logits.shape[1] - 1)
        valid = (positions >= batch["selection_min_count"].unsqueeze(1)) & (
            positions <= maximum.unsqueeze(1)
        )
        predicted_counts = count_logits.masked_fill(
            ~valid, torch.finfo(count_logits.dtype).min
        ).argmax(dim=1)
        predicted_counts = torch.minimum(
            predicted_counts, batch["action_mask"].sum(dim=1)
        )
        for row, record in enumerate(chunk):
            count = int(predicted_counts[row].item())
            prediction = (
                set(int(index) for index in torch.topk(logits[row], k=count).indices.tolist())
                if count
                else set()
            )
            exact = prediction == set(int(target) for target in record["targets"])
            overall_records += 1
            overall_exact += int(exact)
            decision = record["reward_metrics"]["decision"]
            for name, predicate in SLICE_PREDICATES.items():
                if predicate(decision):
                    totals[name]["records"] += 1
                    totals[name]["exact"] += int(exact)
    slices = {
        name: {
            **values,
            "exact_action_rate": (
                values["exact"] / values["records"] if values["records"] else None
            ),
        }
        for name, values in totals.items()
    }
    return {
        "schema_version": "alakazam_validation_reward_slices_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset_sha256": _sha256(dataset),
        "reward_sidecar_sha256": _sha256(reward_sidecar),
        "validation_records": overall_records,
        "overall_exact_action_rate": overall_exact / overall_records,
        "slices": slices,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reward-event validation slices")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reward-sidecar", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_slices(
        args.dataset,
        args.reward_sidecar,
        args.checkpoint,
        args.config,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    _main()
