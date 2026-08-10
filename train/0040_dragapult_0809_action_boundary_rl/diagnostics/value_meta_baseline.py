"""Post-hoc local validation baseline for the deployed paired-0809 Value artifact."""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


PROJECT = "train.0040_dragapult_0809_action_boundary_rl"
ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / (
    "rl_runs/0036_dedicated_action_value_network/datasets/"
    "dragapult_focal_all_dates_weighted_exact10k_tensor_v1"
)
SOURCE_DATASET = ROOT / (
    "rl_runs/0036_dedicated_action_value_network/datasets/"
    "dragapult_focal_all_dates_weighted_exact10k"
)
OUTPUT = ROOT / ".tmp/strategy_adapter_v2_audit/value_meta_baseline.json"
ARCHETYPES = ROOT / "train/0036_dedicated_action_value_network/assets/archetypes_v1.json"


def _weighted_mean(value: Tensor, weight: Tensor) -> float:
    return float((value.double() * weight.double()).sum() / weight.double().sum())


def _weighted_variance(value: Tensor, weight: Tensor) -> Tensor:
    mean = (value.double() * weight.double()).sum() / weight.double().sum()
    return ((value.double() - mean).square() * weight.double()).sum() / weight.double().sum()


def _weighted_correlation(left: Tensor, right: Tensor, weight: Tensor) -> float:
    left = left.double()
    right = right.double()
    weight = weight.double()
    total = weight.sum()
    left_mean = (left * weight).sum() / total
    right_mean = (right * weight).sum() / total
    covariance = ((left - left_mean) * (right - right_mean) * weight).sum() / total
    denominator = (
        _weighted_variance(left, weight).sqrt()
        * _weighted_variance(right, weight).sqrt()
    ).clamp_min(1e-12)
    return float(covariance / denominator)


def _weighted_auc(probability: Tensor, target: Tensor, weight: Tensor) -> float:
    order = probability.argsort()
    probability = probability[order]
    target = target[order].long()
    weight = weight[order].double()
    positive = weight[target == 1].sum()
    negative = weight[target == 0].sum()
    if positive == 0 or negative == 0:
        return 0.5
    rank_negative = torch.zeros((), dtype=torch.float64)
    area = torch.zeros((), dtype=torch.float64)
    index = 0
    while index < probability.numel():
        end = index + 1
        while end < probability.numel() and probability[end] == probability[index]:
            end += 1
        group_target = target[index:end]
        group_weight = weight[index:end]
        group_negative = group_weight[group_target == 0].sum()
        group_positive = group_weight[group_target == 1].sum()
        area += group_positive * (rank_negative + 0.5 * group_negative)
        rank_negative += group_negative
        index = end
    return float(area / (positive * negative))


def _value_metrics(probability: Tensor, target: Tensor, weight: Tensor) -> dict[str, Any]:
    probability = probability.float()
    target = target.float()
    weight = weight.float()
    log_loss = torch.nn.functional.binary_cross_entropy(
        probability.clamp(1e-7, 1 - 1e-7), target, reduction="none"
    )
    signed_prediction = 2.0 * probability - 1.0
    signed_target = 2.0 * target - 1.0
    residual_variance = _weighted_variance(signed_target - signed_prediction, weight)
    target_variance = _weighted_variance(signed_target, weight)
    bins = []
    ece = 0.0
    indices = (probability * 10).long().clamp_max(9)
    for index in range(10):
        selected = indices == index
        if not selected.any():
            continue
        bin_weight = weight[selected].sum()
        predicted = _weighted_mean(probability[selected], weight[selected])
        empirical = _weighted_mean(target[selected], weight[selected])
        fraction = float(bin_weight / weight.sum())
        ece += fraction * abs(predicted - empirical)
        bins.append({
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "decision_count": int(selected.sum()),
            "episode_weight": float(bin_weight),
            "predicted_win_probability": predicted,
            "empirical_win_rate": empirical,
        })
    return {
        "decision_count": probability.numel(),
        "episode_weight": float(weight.sum()),
        "bce": _weighted_mean(log_loss, weight),
        "brier": _weighted_mean((probability - target).square(), weight),
        "accuracy_at_0_5": _weighted_mean(
            (probability.ge(0.5) == target.bool()).float(), weight
        ),
        "auroc": _weighted_auc(probability, target, weight),
        "explained_variance_signed_outcome": float(
            1.0 - residual_variance / target_variance.clamp_min(1e-12)
        ),
        "pearson_signed_outcome": _weighted_correlation(
            signed_prediction, signed_target, weight
        ),
        "ece_10": ece,
        "mean_predicted_win_probability": _weighted_mean(probability, weight),
        "empirical_win_rate": _weighted_mean(target, weight),
        "calibration_bins": bins,
    }


def _meta_metrics(
    probabilities: Tensor,
    target: Tensor,
    weight: Tensor,
    names: list[str],
) -> dict[str, Any]:
    prediction = probabilities.argmax(dim=-1)
    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)
    raw_confusion = torch.zeros(15, 15, dtype=torch.long)
    weighted_confusion = torch.zeros(15, 15, dtype=torch.float64)
    for true_class, predicted_class, sample_weight in zip(target, prediction, weight):
        raw_confusion[int(true_class), int(predicted_class)] += 1
        weighted_confusion[int(true_class), int(predicted_class)] += float(sample_weight)
    per_class = []
    accuracies = []
    for class_id, name in enumerate(names):
        selected = target == class_id
        class_weight = weight[selected].sum()
        accuracy = (
            _weighted_mean(prediction[selected].eq(class_id).float(), weight[selected])
            if selected.any() else None
        )
        if accuracy is not None:
            accuracies.append(accuracy)
        per_class.append({
            "class_id": class_id,
            "name": name,
            "decision_count": int(selected.sum()),
            "episode_weight": float(class_weight),
            "accuracy": accuracy,
            "mean_entropy_nats": (
                _weighted_mean(entropy[selected], weight[selected]) if selected.any() else None
            ),
        })
    return {
        "weighted_accuracy": _weighted_mean(prediction.eq(target).float(), weight),
        "raw_decision_accuracy": float(prediction.eq(target).float().mean()),
        "macro_weighted_class_accuracy": sum(accuracies) / len(accuracies),
        "mean_entropy_nats": _weighted_mean(entropy, weight),
        "normalized_mean_entropy": _weighted_mean(entropy, weight) / math.log(15),
        "per_class": per_class,
        "raw_confusion_matrix_rows_true_cols_predicted": raw_confusion.tolist(),
        "episode_weighted_confusion_matrix_rows_true_cols_predicted": weighted_confusion.tolist(),
    }


def _first_decision_probe(
    model: Any,
    *,
    device: str,
    names: list[str],
    taxonomy: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    dataset_module = importlib.import_module(
        "train.0036_dedicated_action_value_network.training.dataset"
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(SOURCE_DATASET.glob("validation-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = (row.get("audit") or {}).get("identity") or {}
                if identity.get("actor_decision_index") == 0:
                    rows.append(row)
    triggers = {
        int(item["class_id"]): {int(card_id) for card_id in item["trigger_card_ids"]}
        for item in taxonomy["classes"]
    }
    probabilities: list[Tensor] = []
    targets: list[Tensor] = []
    trigger_visible: list[Tensor] = []
    turns: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = dataset_module.ValueDataset._collate(rows[start : start + batch_size]).to(
                device
            )
            validated, state, options = model.actor.encode(batch.features)
            memory = torch.cat((state.tokens, options), dim=1)
            memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
            outputs = model.value_head(memory, memory_mask)
            local_visible = []
            card_ids = batch.features["card_cat"][..., 0]
            owners = batch.features["card_cat"][..., 2]
            card_mask = batch.features["card_mask"]
            for row_index, target in enumerate(batch.archetype_target.tolist()):
                visible_ids = set(
                    card_ids[row_index][(owners[row_index] == 2) & card_mask[row_index]].tolist()
                )
                local_visible.append(bool(triggers[target] & visible_ids))
            probabilities.append(outputs.archetype_logits.softmax(dim=-1).cpu())
            targets.append(batch.archetype_target.cpu())
            trigger_visible.append(torch.tensor(local_visible, dtype=torch.bool))
            turns.append(batch.features["global_num"][:, 0].cpu())
    probability = torch.cat(probabilities)
    target = torch.cat(targets)
    visible = torch.cat(trigger_visible)
    turn = torch.cat(turns)
    weight = torch.ones(target.shape[0])
    result = {
        "trajectory_count": len(rows),
        "raw_turn_distribution": {
            str(int(value)): int((turn == value).sum()) for value in turn.unique(sorted=True)
        },
        "overall": _meta_metrics(probability, target, weight, names),
        "trigger_card_publicly_visible": {
            "trajectory_count": int(visible.sum()),
            "metrics": _meta_metrics(
                probability[visible], target[visible], weight[visible], names
            ) if visible.any() else None,
        },
        "trigger_card_not_publicly_visible": {
            "trajectory_count": int((~visible).sum()),
            "metrics": _meta_metrics(
                probability[~visible], target[~visible], weight[~visible], names
            ) if (~visible).any() else None,
        },
    }
    return result


def run_baseline(
    output: Path = OUTPUT,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, Any]:
    initialization = importlib.import_module(f"{PROJECT}.initialization")
    presets = importlib.import_module(f"{PROJECT}.integrated.presets")
    runner = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
    materialized = importlib.import_module(
        "train.0036_dedicated_action_value_network.training.materialized"
    )
    dataset = materialized.MaterializedValueDataset(
        DATASET,
        source_root=SOURCE_DATASET,
        verify_hashes=False,
        pin_memory=True,
        prefetch_depth=2,
    )
    model, identity = initialization.build_update0_model(
        runner.focal_deck(),
        device=device,
        integrated_flags=presets.preset("PRIZE"),
    )
    model.eval()
    collected: dict[str, list[Tensor]] = {
        name: [] for name in (
            "probability", "target", "weight", "meta_probability", "meta_target",
            "turn", "is_exact_007",
        )
    }
    with torch.inference_mode():
        for batch in dataset.batches("validation", batch_size, shuffle=False):
            batch = batch.to(device, non_blocking=True)
            validated, state, options = model.actor.encode(batch.features)
            memory = torch.cat((state.tokens, options), dim=1)
            memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
            outputs = model.value_head(memory, memory_mask)
            values = {
                "probability": outputs.value_logit.sigmoid(),
                "target": batch.value_target,
                "weight": batch.episode_weight,
                "meta_probability": outputs.archetype_logits.softmax(dim=-1),
                "meta_target": batch.archetype_target,
                "turn": batch.features["global_num"][:, 0],
                "is_exact_007": batch.is_exact_007,
            }
            for name, value in values.items():
                collected[name].append(value.detach().cpu())
    values = {name: torch.cat(parts) for name, parts in collected.items()}
    taxonomy = json.loads(ARCHETYPES.read_text(encoding="utf-8"))
    names = [item["name"] for item in taxonomy["classes"]]
    first_decision = _first_decision_probe(
        model,
        device=device,
        names=names,
        taxonomy=taxonomy,
        batch_size=min(batch_size, 128),
    )

    turn_buckets = []
    for name, low, high in (
        ("early_raw_turn_0_5", 0, 6),
        ("middle_raw_turn_6_11", 6, 12),
        ("late_raw_turn_12_plus", 12, None),
    ):
        selected = values["turn"].ge(low)
        if high is not None:
            selected &= values["turn"].lt(high)
        turn_buckets.append({
            "name": name,
            "raw_turn_min": low,
            "raw_turn_max_exclusive": high,
            "value": _value_metrics(
                values["probability"][selected], values["target"][selected],
                values["weight"][selected],
            ),
            "meta": _meta_metrics(
                values["meta_probability"][selected], values["meta_target"][selected],
                values["weight"][selected], names,
            ),
        })

    by_archetype = []
    for class_id, name in enumerate(names):
        selected = values["meta_target"] == class_id
        by_archetype.append({
            "class_id": class_id,
            "name": name,
            "value": _value_metrics(
                values["probability"][selected], values["target"][selected],
                values["weight"][selected],
            ),
        })
    by_own_deck = []
    for exact in (False, True):
        selected = values["is_exact_007"] == exact
        if selected.any():
            by_own_deck.append({
                "group": "exact_007" if exact else "other_dragapult_decks",
                "value": _value_metrics(
                    values["probability"][selected], values["target"][selected],
                    values["weight"][selected],
                ),
            })

    report = {
        "schema_version": "0040_strategy_adapter_v2_value_meta_baseline_v1",
        "evidence_boundary": {
            "kind": "post_hoc_local_validation",
            "checkpoint_original_catalog_sha256": (
                "f04233b1da6b86ca2f2122fc5a450fdc1202da1430a6db60848a06ceb4fc0868"
            ),
            "original_catalog_present": False,
            "local_materialized_dataset": str(DATASET.relative_to(ROOT)),
            "local_source_catalog_sha256": dataset.reference["source_catalog_sha256"],
            "local_source_checkpoint_required_sha256": dataset.reference[
                "source_checkpoint_required_sha256"
            ],
            "deployed_actor_checkpoint_sha256": identity.checkpoint_sha256,
            "warning": (
                "The local validation feature split predates the paired-0809 V9 run and does "
                "not reproduce the checkpoint's unavailable original catalog."
            ),
        },
        "turn_bucket_contract": (
            "Actor-visible raw engine turn from global_num[:,0]: early [0,6), "
            "middle [6,12), late [12,infinity)."
        ),
        "value_overall": _value_metrics(
            values["probability"], values["target"], values["weight"]
        ),
        "meta_overall": _meta_metrics(
            values["meta_probability"], values["meta_target"], values["weight"], names
        ),
        "first_actor_decision_meta_probe": first_decision,
        "turn_buckets": turn_buckets,
        "value_by_opponent_archetype": by_archetype,
        "value_by_own_deck_group": by_own_deck,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = run_baseline(args.output, batch_size=args.batch_size, device=args.device)
    print(json.dumps({
        "output": str(args.output),
        "value_overall": report["value_overall"],
        "meta_summary": {
            key: value for key, value in report["meta_overall"].items()
            if key not in {"per_class", "raw_confusion_matrix_rows_true_cols_predicted",
                           "episode_weighted_confusion_matrix_rows_true_cols_predicted"}
        },
        "turn_buckets": [
            {
                "name": row["name"],
                "meta_accuracy": row["meta"]["weighted_accuracy"],
                "meta_entropy": row["meta"]["mean_entropy_nats"],
                "value_brier": row["value"]["brier"],
            }
            for row in report["turn_buckets"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
