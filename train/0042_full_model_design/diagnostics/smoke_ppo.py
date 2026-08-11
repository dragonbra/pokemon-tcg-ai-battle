"""Fail-closed 0042 architecture preflight and isolated real PPO smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from ..initialization import build_preset_from_common_update0
from ..integrated.presets import preset
from ..policy.batching import move_batch
from ..policy.opponent_archetype import (
    OpponentArchetypeTarget,
    OpponentArchetypeTaxonomy,
)
from ..policy.own_archetype import OwnArchetypeId, OwnArchetypeVocabulary
from ..training.batch_full_semantic import prepare_episodes
from ..training.ppo_full_semantic import PPOConfig, PPOTrainer, index_feature_batch
from ..training.run_full_semantic import (
    RunConfig,
    TRAINING_OPPONENT_POLICY_ID,
    _episode_metrics,
    _replace_chance_boundary_episodes,
    build_collector,
    build_jobs,
    focal_deck,
    load_frozen_opponent,
)
from ..training.storage_full_semantic import save_model_only
from .strategy_adapter_v2_audit import DEFAULT_REPLAY, _public_feature_batch, run_audit
from .strategy_sensitivity import run as run_sensitivity
from .value_meta_baseline import run_baseline


ROOT = Path(__file__).resolve().parents[3]
SMOKE_ROOT = ROOT / ".tmp/0042_smoke_ppo"
POLICY_REGISTRY = ROOT / "train/0042_full_model_design/policy_registry.json"
FORMAL_RUN_ROOT = ROOT / "rl_runs/0042_full_model_design"
FORMAL_EVAL_ROOT = ROOT / "experiments/0042_full_model_design/evaluation"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _has_actor_visible_meta_target(actor_batch_keys: Iterable[str]) -> bool:
    """Action-sequence `targets` are actor inputs; Meta labels are not."""
    return bool(
        {"opponent_meta_label", "archetype_target"}.intersection(actor_batch_keys)
    )


def _module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _frozen_hashes(model: nn.Module) -> dict[str, str]:
    return {
        "prototype_encoder": _module_sha256(model.actor.prototype_encoder),
        "state_encoder": _module_sha256(model.actor.state_encoder),
        "option_encoder": _module_sha256(model.actor.option_encoder),
        "pretrained_meta_head": _module_sha256(model.value_head.heads.archetype),
    }


def _optimizer_inventory(model: nn.Module, trainer: PPOTrainer) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    seen: set[int] = set()
    frozen: list[str] = []
    duplicates: list[str] = []
    for group in trainer.optimizer.param_groups:
        prefixes = sorted({
            names[id(parameter)].split(".")[0]
            + ("." + names[id(parameter)].split(".")[1]
               if names[id(parameter)].startswith("actor.action_decoder.") else "")
            for parameter in group["params"]
        })
        for parameter in group["params"]:
            name = names[id(parameter)]
            if not parameter.requires_grad:
                frozen.append(name)
            if id(parameter) in seen:
                duplicates.append(name)
            seen.add(id(parameter))
        groups.append({
            "name": str(group["name"]),
            "parameter_prefixes": prefixes,
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "tensor_count": len(group["params"]),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        })
    return {
        "groups": groups,
        "frozen_parameters_in_optimizer": frozen,
        "duplicate_parameters_in_optimizer": duplicates,
        "optimizer_parameter_count": sum(row["parameter_count"] for row in groups),
    }


def _base_equivalence(model: nn.Module, features: dict[str, Tensor]) -> dict[str, bool]:
    model.eval()
    with torch.inference_mode():
        validated, state, options = model.actor.encode(features)
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = model.value_head.decode(memory, memory_mask)
        base_value = 2.0 * model.value_head.heads.value(queries[:, 0]).squeeze(-1).sigmoid() - 1.0
        base_meta = model.value_head.heads.archetype(queries[:, 1])
        value, auxiliary = model.value_and_aux_from_encoded(validated, state, options)
        context = model.strategy_context(validated, value, auxiliary)
        decoder_state = model.actor.action_decoder.initialize(validated, state.summary)
        base_logits = model.actor.action_decoder.logits(validated, options, decoder_state)
        adapted_logits = model.head.logits(validated, options, decoder_state, context)
    return {
        "g_v_exact_zero": bool(model.value_adapter.gate.eq(0)),
        "g_pi_exact_zero": bool(model.policy_strategy_adapter.gate.eq(0)),
        "value_exact": torch.equal(value, base_value),
        "meta_exact": torch.equal(auxiliary["meta_logits"], base_meta),
        "policy_logits_exact": torch.equal(adapted_logits, base_logits),
        "base_has_no_option_lora": not any(
            "lora" in name.lower() or ".parametrizations." in name
            for name, _ in model.actor.option_encoder.named_parameters()
        ),
    }


def _preflight(output: Path) -> tuple[dict[str, Any], nn.Module, PPOTrainer]:
    audit = run_audit(output / "architecture_audit.json", DEFAULT_REPLAY)
    flags = preset("FULL_MODEL")
    model, identity = build_preset_from_common_update0(
        focal_deck(), flags, device="cuda:0"
    )
    ppo_config = PPOConfig(meta_anchor_coef=flags.meta_anchor_coef)
    trainer = PPOTrainer(model, device=torch.device("cuda:0"), config=ppo_config)
    features, deck, _ = _public_feature_batch(DEFAULT_REPLAY)
    features = move_batch(features, torch.device("cuda:0"))
    vocabulary = OwnArchetypeVocabulary.load()
    own = vocabulary.classify_own_deck(focal_deck())
    target = OpponentArchetypeTaxonomy.load().classify_target(deck)
    inventory = _optimizer_inventory(model, trainer)
    cases = audit["gradient_cases"]

    def nonzero(case: str, module: str) -> bool:
        return cases[case]["modules"][module]["gradient_l2"] > 0.0

    gradient_contract = {
        "policy": {
            name: nonzero("policy", name) for name in (
                "prototype_encoder", "state_encoder", "option_encoder", "value_trunk",
                "value_adapter", "pretrained_meta_head", "action_decoder",
                "policy_strategy_adapter",
            )
        },
        "value": {
            name: nonzero("value", name) for name in (
                "prototype_encoder", "state_encoder", "option_encoder", "value_trunk",
                "value_adapter", "pretrained_meta_head", "action_decoder",
                "policy_strategy_adapter",
            )
        },
        "meta": {
            name: nonzero("pretrained_meta", name) for name in (
                "prototype_encoder", "state_encoder", "option_encoder", "value_trunk",
                "value_adapter", "pretrained_meta_head", "action_decoder",
                "policy_strategy_adapter",
            )
        },
    }
    expected = {
        "policy": {"action_decoder", "policy_strategy_adapter"},
        "value": {"value_trunk", "value_adapter"},
        "meta": {"value_trunk"},
    }
    gradients_pass = all(
        {name for name, value in row.items() if value} == expected[case]
        for case, row in gradient_contract.items()
    )
    base = _base_equivalence(model, features)
    own_embedding_ids = {
        id(model.value_adapter.own_embedding.weight),
        id(model.policy_strategy_adapter.own_embedding.weight),
    }
    own_semantics = {
        "vocabulary_size": len(vocabulary.classes),
        "embedding_dim": model.value_adapter.own_embedding.embedding_dim,
        "runtime_index_source": "focal exact deck -> OwnArchetypeVocabulary.classify_own_deck",
        "runtime_index": int(model.default_own_archetype_id),
        "focal_deck_classified_index": own.value,
        "others_id": vocabulary.others_id,
        "own_type": type(own).__name__,
        "opponent_target_type": type(target).__name__,
        "types_distinct": isinstance(own, OwnArchetypeId)
        and isinstance(target, OpponentArchetypeTarget)
        and type(own) is not type(target),
        "embedding_parameters_independent": len(own_embedding_ids) == 2,
        "taxonomy_apis_independent": (
            OwnArchetypeVocabulary is not OpponentArchetypeTaxonomy
        ),
        "value_num_embeddings": model.value_adapter.own_embedding.num_embeddings,
        "policy_num_embeddings": model.policy_strategy_adapter.own_embedding.num_embeddings,
    }
    meta_plumbing = {
        "preset_coef": flags.meta_anchor_coef,
        "ppo_coef": trainer.config.meta_anchor_coef,
        "meta_head_trainable_parameters": sum(
            parameter.numel() for parameter in model.value_head.heads.archetype.parameters()
            if parameter.requires_grad
        ),
        "target_actor_visible": _has_actor_visible_meta_target(
            model.actor.expected_batch_keys
        ),
    }
    option_lora_installed = any(
        "lora" in name.lower() or ".parametrizations." in name
        for name, _ in model.actor.option_encoder.named_parameters(remove_duplicate=False)
    )
    optimizer_boundary = {
        **inventory,
        "option_encoder_trainable_parameters": sum(
            parameter.numel() for parameter in model.actor.option_encoder.parameters()
            if parameter.requires_grad
        ),
        "option_lora_installed": option_lora_installed,
        "option_lora_optimizer_group_present": any(
            "lora" in str(group["name"]).lower() or "option" in str(group["name"]).lower()
            for group in trainer.optimizer.param_groups
        ),
    }
    gates = {
        "own_archetype_semantics": (
            own_semantics["vocabulary_size"] == 15
            and own_semantics["value_num_embeddings"] == 15
            and own_semantics["policy_num_embeddings"] == 15
            and own_semantics["runtime_index"] == own_semantics["focal_deck_classified_index"]
            and own_semantics["types_distinct"]
            and own_semantics["embedding_parameters_independent"]
            and own_semantics["taxonomy_apis_independent"]
        ),
        "meta_anchor_plumbing": (
            meta_plumbing["preset_coef"] == 0.10
            and meta_plumbing["ppo_coef"] == 0.10
            and meta_plumbing["meta_head_trainable_parameters"] == 0
            and not meta_plumbing["target_actor_visible"]
            and gradient_contract["meta"]["value_trunk"]
        ),
        "zero_gate_base": all(base.values()),
        "optimizer_boundary": (
            not inventory["frozen_parameters_in_optimizer"]
            and not inventory["duplicate_parameters_in_optimizer"]
            and optimizer_boundary["option_encoder_trainable_parameters"] == 0
            and not option_lora_installed
            and not optimizer_boundary["option_lora_optimizer_group_present"]
            and len(inventory["groups"]) == 6
        ),
        "gradient_boundary": gradients_pass,
    }
    report = {
        "schema_version": "0042_final_preflight_v1",
        "passed": all(gates.values()),
        "gates": gates,
        "source_identity": asdict(identity),
        "own_archetype": own_semantics,
        "meta_anchor": meta_plumbing,
        "zero_gate_0042_base": base,
        "optimizer": optimizer_boundary,
        "gradient_matrix_nonzero": gradient_contract,
    }
    _atomic_json(output / "preflight.json", report)
    if not report["passed"]:
        raise RuntimeError(f"0042 architecture preflight blocker: {gates}")
    return report, model, trainer


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    return all(
        not isinstance(value, float) or math.isfinite(value)
        for value in metrics.values()
    )


def _benchmark_feature_roundtrip(features: dict[str, Tensor]) -> dict[str, float]:
    first = next(iter(features.values()), None)
    count = min(256, 0 if first is None else int(first.shape[0]))
    sample = {name: value[:count] for name, value in features.items()}
    if not sample or next(iter(sample.values())).device.type != "cuda":
        raise RuntimeError("CUDA smoke did not retain device-resident trajectory features")
    torch.cuda.synchronize()
    started = time.perf_counter()
    host_batch = {name: value.cpu() for name, value in sample.items()}
    move_batch(host_batch, torch.device("cuda:0"))
    torch.cuda.synchronize()
    roundtrip = time.perf_counter() - started
    del host_batch
    torch.cuda.synchronize()
    started = time.perf_counter()
    index_feature_batch(
        features, torch.arange(count), torch.device("cuda:0")
    )
    torch.cuda.synchronize()
    resident = time.perf_counter() - started
    bytes_one_way = sum(
        value.numel() * value.element_size() for value in sample.values()
    )
    return {
        "sample_decisions": float(count),
        "feature_bytes_one_way": float(bytes_one_way),
        "legacy_gpu_cpu_gpu_seconds": roundtrip,
        "device_resident_collate_seconds": resident,
        "measured_speedup": roundtrip / max(resident, 1.0e-12),
        "avoided_transfer_bytes_per_sample": float(2 * bytes_one_way),
    }


def _fixed_summary(report: dict[str, Any]) -> dict[str, Any]:
    buckets = {row["name"]: row for row in report["turn_buckets"]}
    return {
        "value": {
            key: report["value_overall"][key] for key in (
                "auroc", "brier", "ece_10", "explained_variance_signed_outcome",
                "mean_predicted_win_probability", "empirical_win_rate",
            )
        },
        "meta": {
            key: report["meta_overall"][key] for key in (
                "weighted_accuracy", "macro_weighted_class_accuracy", "mean_entropy_nats",
            )
        },
        "raw_turn_0_accuracy": buckets["raw_turn_0"]["meta"]["weighted_accuracy"],
        "turn_buckets": {
            name: {
                "meta_accuracy": row["meta"]["weighted_accuracy"],
                "meta_entropy": row["meta"]["mean_entropy_nats"],
                "value_explained_variance": row["value"]["explained_variance_signed_outcome"],
            }
            for name, row in buckets.items() if name != "raw_turn_0"
        },
    }


def run(output: Path, *, seed: int = 420_042_911, updates: int = 16) -> dict[str, Any]:
    output = output.resolve()
    if SMOKE_ROOT.resolve() not in output.parents or output.exists():
        raise ValueError("Smoke output must be a fresh child of .tmp/0042_smoke_ppo")
    if updates < 1 or updates > 20:
        raise ValueError("Smoke PPO updates must be in [1, 20]")
    output.mkdir(parents=True)
    _atomic_json(output / "status.json", {
        "state": "preflight", "artifact_role": "NON-CANDIDATE_SMOKE_ONLY"
    })
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    registry_before = POLICY_REGISTRY.read_bytes()
    formal_run_existed = FORMAL_RUN_ROOT.exists()
    formal_eval_listing = sorted(str(path) for path in FORMAL_EVAL_ROOT.glob("**/*"))

    preflight, model, trainer = _preflight(output)
    frozen_before = _frozen_hashes(model)
    before_checkpoint = output / "checkpoint-before-smoke.pt"
    save_model_only(model, before_checkpoint, update=0, metadata={
        "artifact_role": "NON-CANDIDATE_SMOKE_ONLY",
        "candidate_eligible": False,
        "smoke_seed": seed,
    })
    before_fixed = run_baseline(
        output / "fixed-validation-before.json", checkpoint=before_checkpoint,
        batch_size=512, device="cuda:0",
    )
    torch.cuda.empty_cache()
    before_sensitivity = run_sensitivity(before_checkpoint)
    _atomic_json(output / "strategy-sensitivity-before.json", before_sensitivity)

    opponent = load_frozen_opponent(torch.device("cuda:0"))
    flags = preset("FULL_MODEL")
    ppo_config = trainer.config
    rollout_config = RunConfig(
        version="V999_smoke_only",
        updates=updates,
        games_per_update=256,
        trajectory_games_per_update=256,
        rollout_batch_size=256,
        cuda_lane_count=256,
        ppo_minibatch_size=ppo_config.batch_size,
        ppo_gradient_accumulation=ppo_config.gradient_accumulation,
        ppo_epochs=ppo_config.epochs,
        eval_every=999,
        preset_name="FULL_MODEL",
        wandb_mode="offline",
        launch_formal=False,
        ppo=ppo_config,
    )
    rollout_config.validate()
    config_report = {
        "seed": seed,
        "updates": updates,
        "games_per_update": 256,
        "frequency_units_per_update": 1,
        "trajectory_games_per_update": 256,
        "rollout_batch_size": 256,
        "cuda_lane_count": 256,
        "ppo": asdict(ppo_config),
        "device": "cuda:0",
        "training_precision": "fp32",
        "base_actor": preflight["source_identity"],
        "paired_value": "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/value_head.pt",
        "opponent_policy_id": TRAINING_OPPONENT_POLICY_ID,
        "artifact_role": "NON-CANDIDATE_SMOKE_ONLY",
        "wandb": "disabled_for_diagnostic_smoke",
        "formal_evaluation": False,
    }
    _atomic_json(output / "smoke-config.json", config_report)
    metrics_path = output / "updates.jsonl"
    update_rows = []
    transfer_benchmark: dict[str, float] | None = None
    started = time.perf_counter()
    for update in range(1, updates + 1):
        source_update = update - 1
        collector = build_collector(model, opponent, rollout_config, mode="sample")
        jobs = build_jobs(source_policy_update=source_update, seed=seed, count=256)
        rollout_started = time.perf_counter()
        episodes = collector.collect(jobs)

        replacement_metrics: list[dict[str, float]] = []
        def collect_replacements(replacement_jobs):
            replacement = build_collector(model, opponent, rollout_config, mode="sample")
            values = replacement.collect(replacement_jobs)
            replacement_metrics.append(replacement.metrics())
            return values

        episodes, exclusions = _replace_chance_boundary_episodes(
            episodes, collect_replacements
        )
        rollout_seconds = time.perf_counter() - rollout_started
        batch = prepare_episodes(
            episodes,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            credit_clock=ppo_config.credit_clock,
            loss_weighting=ppo_config.loss_weighting,
            prize_mode=flags.prize_aux_mode,
            prize_scale=flags.prize_aux_scale,
            require_policy_identity=True,
        )
        rollout_metrics = _episode_metrics(episodes, "rollout")
        del episodes
        torch.cuda.empty_cache()
        if transfer_benchmark is None:
            transfer_benchmark = _benchmark_feature_roundtrip(batch.features)
            _atomic_json(output / "cuda-feature-transfer-benchmark.json", transfer_benchmark)
        if not all(
            value.device.type == "cuda"
            for value in batch.features.values()
        ):
            raise RuntimeError("trajectory feature left CUDA before PPO")
        gradient_probe = trainer.sparse_gradient_diagnostics(batch, samples=64)
        ppo_started = time.perf_counter()
        ppo_metrics = trainer.update(batch, update=update)
        ppo_seconds = time.perf_counter() - ppo_started
        collector_metrics = collector.metrics()
        if (
            collector_metrics.get("rollout/cuda_features_device_resident") != 1.0
            or collector_metrics.get("rollout/cuda_feature_d2h_bytes") != 0.0
            or collector_metrics.get("rollout/lane_routing_audit_pass") != 1.0
            or collector_metrics.get("rollout/lane_routing_audit_failures") != 0.0
        ):
            raise RuntimeError(
                f"Smoke CUDA residency/routing contract failed: {collector_metrics}"
            )
        row = {
            "trainer/update": update,
            "rollout/source_policy_update": source_update,
            "rollout/wall_seconds": rollout_seconds,
            "ppo/wall_seconds": ppo_seconds,
            "rollout/chance_boundary_exclusions": len(exclusions),
            **rollout_metrics,
            **collector_metrics,
            **gradient_probe,
            **ppo_metrics,
        }
        if not _finite_metrics(row):
            raise FloatingPointError(f"nonfinite Smoke metric at update {update}")
        update_rows.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        save_model_only(
            model,
            output / f"checkpoint-update-{update:06d}.pt",
            update=update,
            metadata={
                "artifact_role": "NON-CANDIDATE_SMOKE_ONLY",
                "candidate_eligible": False,
                "smoke_seed": seed,
                "smoke_updates_completed": update,
            },
        )
        print(
            f"[0042 SMOKE] update {update}/{updates} "
            f"win_rate={row['rollout/win_rate']:.3f} "
            f"gV={row['ppo/value_adapter_gate_raw']:.6g} "
            f"gPi={row['ppo/policy_adapter_gate_raw']:.6g} "
            f"KL={row['ppo/behavior_kl']:.6g}",
            flush=True,
        )
        del batch, collector
        torch.cuda.empty_cache()

    final_checkpoint = output / "checkpoint-after-smoke.pt"
    save_model_only(model, final_checkpoint, update=updates, metadata={
        "artifact_role": "NON-CANDIDATE_SMOKE_ONLY",
        "candidate_eligible": False,
        "smoke_seed": seed,
        "smoke_updates": updates,
    })
    frozen_after = _frozen_hashes(model)
    del opponent
    torch.cuda.empty_cache()
    after_fixed = run_baseline(
        output / "fixed-validation-after.json", checkpoint=final_checkpoint,
        batch_size=512, device="cuda:0",
    )
    torch.cuda.empty_cache()
    after_sensitivity = run_sensitivity(final_checkpoint)
    _atomic_json(output / "strategy-sensitivity-after.json", after_sensitivity)
    frozen_integrity = {
        name: {
            "before": frozen_before[name],
            "after": frozen_after[name],
            "unchanged": frozen_before[name] == frozen_after[name],
        }
        for name in frozen_before
    }
    contamination = {
        "policy_registry_unchanged": POLICY_REGISTRY.read_bytes() == registry_before,
        "formal_run_root_preexisted": formal_run_existed,
        "formal_run_root_created": not formal_run_existed and FORMAL_RUN_ROOT.exists(),
        "formal_evaluation_listing_unchanged": (
            sorted(str(path) for path in FORMAL_EVAL_ROOT.glob("**/*")) == formal_eval_listing
        ),
        "candidate_identity_created": False,
        "promotion_performed": False,
        "wandb_run_created": False,
    }
    if not all(row["unchanged"] for row in frozen_integrity.values()):
        raise RuntimeError("Smoke PPO mutated a frozen representation module")
    if not contamination["policy_registry_unchanged"] or contamination["formal_run_root_created"]:
        raise RuntimeError("Smoke PPO contaminated formal policy/run state")
    report = {
        "schema_version": "0042_final_preflight_smoke_ppo_v1",
        "artifact_role": "NON-CANDIDATE_SMOKE_ONLY",
        "preflight": preflight,
        "configuration": config_report,
        "updates": update_rows,
        "cuda_feature_transfer": transfer_benchmark,
        "fixed_validation": {
            "before": _fixed_summary(before_fixed),
            "after": _fixed_summary(after_fixed),
        },
        "strategy_sensitivity": {
            "before": before_sensitivity,
            "after": after_sensitivity,
        },
        "frozen_parameter_integrity": frozen_integrity,
        "formal_contamination": contamination,
        "elapsed_seconds": time.perf_counter() - started,
        "nan_or_inf": False,
    }
    _atomic_json(output / "report.json", report)
    _atomic_json(output / "status.json", {
        "state": "complete",
        "artifact_role": "NON-CANDIDATE_SMOKE_ONLY",
        "updates": updates,
        "report": "report.json",
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=420_042_911)
    parser.add_argument("--updates", type=int, default=16)
    args = parser.parse_args()
    report = run(args.output, seed=args.seed, updates=args.updates)
    print(json.dumps({
        "output": str(args.output),
        "preflight_passed": report["preflight"]["passed"],
        "updates": len(report["updates"]),
        "frozen_unchanged": all(
            row["unchanged"] for row in report["frozen_parameter_integrity"].values()
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
