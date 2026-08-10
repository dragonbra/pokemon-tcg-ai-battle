"""Empirical 0042 Strategy Adapter architecture audit.

This module is deliberately outside every training and deployment import path. It
loads the real paired-0809 model, compiles one public replay observation through
the production feature compiler, and records parameter ownership and independent
backward probes without stepping the production optimizer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn


PROJECT = "train.0042_full_model_design"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLAY = ROOT / (
    "data/replays/0016_alakazam_multideck_bc/goonew/"
    "submission-54960905/episode-88310066-replay.json"
)
DEFAULT_OUTPUT = ROOT / ".tmp/strategy_adapter_v2_audit/runtime_audit.json"


class ToyZeroGatedResidual(nn.Module):
    """Minimal form of the proposed normally initialized zero-gated adapter."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(()))
        self.residual = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.gate.tanh() * self.residual(hidden)


def zero_gate_startup_probe(seed: int = 400_040_211) -> dict[str, float | bool]:
    torch.manual_seed(seed)
    module = ToyZeroGatedResidual(8)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    hidden = torch.randn(4, 8)
    target = torch.randn(4, 8)

    optimizer.zero_grad(set_to_none=True)
    first_loss = (module(hidden) - target).square().mean()
    first_loss.backward()
    first_gate_grad = float(module.gate.grad)
    first_residual_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in module.residual.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    gate_after_step = float(module.gate.detach())

    optimizer.zero_grad(set_to_none=True)
    second_loss = (module(hidden) - target).square().mean()
    second_loss.backward()
    second_residual_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in module.residual.parameters()
        if parameter.grad is not None
    )
    return {
        "first_gate_grad": first_gate_grad,
        "first_residual_grad_l1": first_residual_grad,
        "gate_after_step": gate_after_step,
        "second_residual_grad_l1": second_residual_grad,
        "passes": (
            abs(first_gate_grad) > 0.0
            and first_residual_grad == 0.0
            and abs(gate_after_step) > 0.0
            and second_residual_grad > 0.0
        ),
    }


def _public_feature_batch(
    replay_path: Path,
) -> tuple[dict[str, Tensor], tuple[int, ...], dict[str, object]]:
    compiler_module = importlib.import_module(f"{PROJECT}.rollout.worker_compiler")
    collate_module = importlib.import_module(f"{PROJECT}.semantic_policy.features.collate")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    observation = next(
        agent["observation"]
        for step in replay["steps"]
        for agent in step
        if isinstance((agent.get("observation") or {}).get("current"), dict)
        and agent["observation"]["current"].get("yourIndex") == 0
        and (agent["observation"].get("select") or {}).get("option")
    )
    deck = tuple(replay["steps"][0][0]["visualize"][0]["action"][0])
    compiler = compiler_module.WorkerLocalCompiler(0, deck)
    batch = collate_module.collate_canonical_records([compiler.compile(observation)])
    return batch, deck, {
        "replay": str(replay_path.relative_to(ROOT)),
        "episode_id": replay.get("id"),
        "actor": 0,
        "deck_cards": len(deck),
        "context": observation.get("select", {}).get("type"),
    }


def _parameter_count(module: nn.Module) -> dict[str, int | bool]:
    parameters = list(module.parameters())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "tensor_count": len(parameters),
        "trainable_tensor_count": sum(parameter.requires_grad for parameter in parameters),
        "all_require_grad": bool(parameters) and all(p.requires_grad for p in parameters),
        "any_require_grad": any(p.requires_grad for p in parameters),
    }


def _nonzero_gradient(module: nn.Module) -> dict[str, int | float]:
    gradients = [
        parameter.grad.detach()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    nonzero = [gradient for gradient in gradients if torch.count_nonzero(gradient).item()]
    return {
        "gradient_tensors": len(gradients),
        "nonzero_gradient_tensors": len(nonzero),
        "nonzero_gradient_elements": sum(
            int(torch.count_nonzero(gradient).item()) for gradient in nonzero
        ),
        "gradient_l2": float(
            sum(gradient.double().square().sum() for gradient in gradients).sqrt()
        ) if gradients else 0.0,
    }


def _clear_gradients(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _audited_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {
        "prototype_encoder": model.actor.prototype_encoder,
        "state_encoder": model.actor.state_encoder,
        "option_encoder": model.actor.option_encoder,
        "action_decoder": model.actor.action_decoder,
        "value_trunk": nn.ModuleList([
            model.value_head.blocks,
            model.value_head.final_norm,
        ]),
        "value_queries": _ParameterModule(model.value_head.queries),
        "value_head": model.value_head.heads.value,
        "pretrained_meta_head": model.value_head.heads.archetype,
        "pretrained_final_diff_head": model.value_head.heads.final_diff,
        "value_adapter": model.value_adapter,
        "policy_strategy_adapter": model.policy_strategy_adapter,
        "allocation_head": model.allocation_head,
    }


class _ParameterModule(nn.Module):
    """Non-owning parameter view used only for uniform diagnostic summaries."""

    def __init__(self, parameter: nn.Parameter) -> None:
        super().__init__()
        object.__setattr__(self, "_view", parameter)

    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]:
        del recurse
        return iter((self._view,))


def _tensor_shapes(model: nn.Module, features: dict[str, Tensor]) -> dict[str, list[int]]:
    with torch.no_grad():
        validated, state, options = model.actor.encode(features)
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = model.value_head.decode(memory, memory_mask)
        value, auxiliary = model.value_and_aux_from_encoded(validated, state, options)
        meta_logits = auxiliary["meta_logits"]
        decoder_state = model.actor.action_decoder.initialize(
            validated, model.actor_summary(state)
        )
        context = model.strategy_context(validated, value, auxiliary)
        logits = model.head.logits(validated, options, decoder_state, context)
    return {
        "state_tokens": list(state.tokens.shape),
        "state_summary": list(state.summary.shape),
        "option_tokens": list(options.shape),
        "value_memory": list(memory.shape),
        "value_queries": list(queries.shape),
        "h_V_query0": list(queries[:, 0].shape),
        "pretrained_meta_query1": list(queries[:, 1].shape),
        "decoder_hidden_h_pi": list(decoder_state.hidden.shape),
        "value": list(value.shape),
        "pretrained_meta_logits": list(meta_logits.shape),
        "policy_logits": list(logits.shape),
    }


def _gradient_case(
    model: nn.Module,
    features: dict[str, Tensor],
    case: str,
    modules: dict[str, nn.Module],
) -> dict[str, object]:
    _clear_gradients(model)
    validated, state, options = model.actor.encode(features)
    if case == "policy":
        value, auxiliary = model.value_and_aux_from_encoded(validated, state, options)
        context = model.strategy_context(validated, value, auxiliary)
        decoder_state = model.actor.action_decoder.initialize(
            validated, model.actor_summary(state)
        )
        logits = model.head.logits(validated, options, decoder_state, context)
        target = validated.option_mask.float().argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, target)
    else:
        value, auxiliary = model.value_and_aux_from_encoded(validated, state, options)
        if case == "value":
            loss = value.square().mean()
        elif case == "pretrained_meta":
            meta_logits = auxiliary["meta_logits"]
            loss = torch.nn.functional.cross_entropy(
                meta_logits, torch.zeros(meta_logits.shape[0], dtype=torch.long)
            )
        else:
            raise ValueError(f"unknown gradient case: {case}")
    loss.backward()
    result = {
        "loss": float(loss.detach()),
        "modules": {name: _nonzero_gradient(module) for name, module in modules.items()},
    }
    _clear_gradients(model)
    return result


def _shared_parameter_names(model: nn.Module) -> list[dict[str, object]]:
    by_identity: dict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        by_identity[id(parameter)].append(name)
    return [
        {"names": names, "numel": dict(model.named_parameters())[names[0]].numel()}
        for names in by_identity.values()
        if len(names) > 1
    ]


def run_audit(output: Path = DEFAULT_OUTPUT, replay: Path = DEFAULT_REPLAY) -> dict[str, object]:
    initialization = importlib.import_module(f"{PROJECT}.initialization")
    presets = importlib.import_module(f"{PROJECT}.integrated.presets")
    ppo_module = importlib.import_module(f"{PROJECT}.training.ppo_full_semantic")
    features, deck, fixture = _public_feature_batch(replay)
    model, identity = initialization.build_preset_from_common_update0(
        deck, presets.preset("FULL_MODEL"), device="cpu"
    )
    trainer = ppo_module.PPOTrainer(model, device=torch.device("cpu"))
    modules = _audited_modules(model)
    optimizer_by_id: dict[int, list[str]] = defaultdict(list)
    frozen_optimizer_parameters: list[str] = []
    duplicate_optimizer_parameters: list[dict[str, object]] = []
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            optimizer_by_id[id(parameter)].append(str(group["name"]))
            if not parameter.requires_grad:
                frozen_optimizer_parameters.append(parameter_names.get(id(parameter), "<unnamed>"))
    for identity_key, groups in optimizer_by_id.items():
        if len(groups) > 1:
            duplicate_optimizer_parameters.append({
                "name": parameter_names.get(identity_key, "<unnamed>"),
                "groups": groups,
            })

    report = {
        "schema_version": "0042_strategy_adapter_runtime_audit_v1",
        "fixture": fixture,
        "source_identity": {
            "checkpoint_sha256": identity.checkpoint_sha256,
            "project_id": identity.project_id,
            "version": identity.version,
            "epoch": identity.epoch,
        },
        "formal_preset": "FULL_MODEL",
        "integrated_flags": model.integrated_flags.metadata(),
        "tensor_shapes": _tensor_shapes(model, features),
        "modules": {name: _parameter_count(module) for name, module in modules.items()},
        "optimizer_groups": trainer.optimizer_group_manifest(),
        "frozen_optimizer_parameters": frozen_optimizer_parameters,
        "duplicate_optimizer_parameters": duplicate_optimizer_parameters,
        "shared_parameter_objects": _shared_parameter_names(model),
        "gradient_cases": {
            case: _gradient_case(model, features, case, modules)
            for case in ("policy", "value", "pretrained_meta")
        },
        "zero_gate_startup": zero_gate_startup_probe(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    args = parser.parse_args()
    report = run_audit(args.output, args.replay)
    print(json.dumps({
        "output": str(args.output),
        "tensor_shapes": report["tensor_shapes"],
        "optimizer_groups": report["optimizer_groups"],
        "zero_gate_startup": report["zero_gate_startup"],
    }, indent=2))


if __name__ == "__main__":
    main()
