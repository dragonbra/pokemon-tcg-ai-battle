"""Evaluate causal feature interventions on one immutable R1 checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig
from .base_model import IDOnlyConfig
from .config import MODEL_READY_DATASET
from .r1_model import R1ModelConfig, R1StructuredCausalPolicy
from .training.a0_trainer import evaluate
from .training.ac_data import iter_ac_batches


AC_GATES = (
    "entity_semantic_gate",
    "global_zone_gate",
    "state_aux_gate",
    "option_semantic_gate",
    "option_aux_gate",
)


def _model_config(raw: dict[str, object]) -> R1ModelConfig:
    ac_raw = raw["ac"]
    if not isinstance(ac_raw, dict) or not isinstance(ac_raw.get("base"), dict):
        raise ValueError("R1 checkpoint metadata is missing model.ac.base")
    ac = ACModelConfig(
        base=IDOnlyConfig(**ac_raw["base"]),
        event_layers=int(ac_raw["event_layers"]),
        auxiliary_ffn_multiplier=int(ac_raw["auxiliary_ffn_multiplier"]),
        goal_roles=int(ac_raw["goal_roles"]),
    )
    return R1ModelConfig(
        ac=ac,
        conditioning_layers=int(raw["conditioning_layers"]),
        conditioning_ffn_multiplier=int(raw["conditioning_ffn_multiplier"]),
        initial_conditioning_strength=float(raw["initial_conditioning_strength"]),
    )


@contextmanager
def _parameter_value(parameter: nn.Parameter, value: float) -> Iterator[None]:
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.fill_(value)
    try:
        yield
    finally:
        with torch.no_grad():
            parameter.copy_(original)


@contextmanager
def _zero_method(model: nn.Module, name: str) -> Iterator[None]:
    original = getattr(model, name)

    def zeroed(*args: object, **kwargs: object) -> Tensor:
        value = original(*args, **kwargs)
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} did not return a tensor")
        return torch.zeros_like(value)

    setattr(model, name, zeroed)
    try:
        yield
    finally:
        delattr(model, name)


@contextmanager
def _zero_module_output(module: nn.Module) -> Iterator[None]:
    handle = module.register_forward_hook(
        lambda _module, _inputs, output: torch.zeros_like(output)
    )
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _intervention(model: R1StructuredCausalPolicy, name: str) -> Iterator[None]:
    with ExitStack() as stack:
        if name in {"legacy_trunk_only", "structured_r1_only"}:
            for gate_name in AC_GATES:
                stack.enter_context(_parameter_value(getattr(model, gate_name), 0.0))
        if name in {"legacy_trunk_only", "inherited_ac_only", "no_r1_state"}:
            stack.enter_context(_parameter_value(model.state_conditioning_logit, -30.0))
        if name in {"legacy_trunk_only", "inherited_ac_only", "no_r1_option"}:
            stack.enter_context(_parameter_value(model.option_conditioning_logit, -30.0))
        if name == "no_card_capability":
            stack.enter_context(_zero_method(model, "_card_features"))
        if name == "no_public_zones":
            stack.enter_context(_zero_method(model, "_zone_context"))
            stack.enter_context(_zero_method(model, "_zone_tokens"))
        if name == "no_resource_ledger":
            stack.enter_context(_zero_method(model, "_resource_tokens"))
        if name == "no_event_memory":
            stack.enter_context(_zero_method(model, "_event_context"))
        if name == "no_opponent_hand_memory":
            stack.enter_context(_zero_method(model, "_hand_context"))
        if name == "no_goal_routing":
            stack.enter_context(_zero_module_output(model.goal_qkv))
        yield


def _gate_diagnostics(model: R1StructuredCausalPolicy) -> dict[str, float]:
    result = {
        "state_conditioning_scale": float(
            torch.sigmoid(model.state_conditioning_logit.detach()).cpu()
        ),
        "option_conditioning_scale": float(
            torch.sigmoid(model.option_conditioning_logit.detach()).cpu()
        ),
    }
    for name in AC_GATES:
        gate = torch.tanh(getattr(model, name).detach()).abs().mean()
        result[f"{name}_mean_abs_scale"] = float(gate.cpu())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    payload = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    raw_config = metadata.get("model")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint metadata has no R1 model config")
    config = _model_config(raw_config)
    model = R1StructuredCausalPolicy(
        config, ontology_path=MODEL_READY_DATASET / "card_ontology.json"
    )
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(arguments.device)
    model.to(device).eval()

    variants = (
        "full_r1",
        "legacy_trunk_only",
        "inherited_ac_only",
        "structured_r1_only",
        "no_r1_state",
        "no_r1_option",
        "no_card_capability",
        "no_public_zones",
        "no_resource_ledger",
        "no_event_memory",
        "no_opponent_hand_memory",
        "no_goal_routing",
    )
    results: list[dict[str, float | str]] = []
    for name in variants:
        started = time.perf_counter()
        with _intervention(model, name):
            metrics = evaluate(
                model,
                iter_ac_batches(
                    MODEL_READY_DATASET,
                    "validation",
                    batch_size=arguments.batch_size,
                    shuffle=False,
                    seed=20260723,
                ),
                device=device,
                amp=device.type == "cuda",
            )
        row: dict[str, float | str] = {
            "variant": name,
            **metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    baseline = results[0]
    for row in results:
        row["delta_exact_action"] = float(row["bc/validation/exact_action"]) - float(
            baseline["bc/validation/exact_action"]
        )
        row["delta_loss"] = float(row["bc/validation/loss"]) - float(
            baseline["bc/validation/loss"]
        )
    report = {
        "schema_version": "0014_r1_checkpoint_ablation_v1",
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "dataset_content_sha256": metadata.get("dataset_content_sha256"),
        "gate_diagnostics": _gate_diagnostics(model),
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
