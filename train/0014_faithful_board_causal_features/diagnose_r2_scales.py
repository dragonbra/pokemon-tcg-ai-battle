"""Measure learned R2 state/option ScaleGate distributions on validation data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from .ac_model import ACModelConfig
from .base_model import IDOnlyConfig
from .config import MODEL_READY_DATASET
from .r2_model import R2ModelConfig, R2StrongScenarioPolicy
from .training.ac_data import iter_ac_batches


def _config(raw: dict[str, object]) -> R2ModelConfig:
    ac_raw = raw["ac"]
    if not isinstance(ac_raw, dict) or not isinstance(ac_raw.get("base"), dict):
        raise ValueError("R2 checkpoint metadata is missing model.ac.base")
    return R2ModelConfig(
        ac=ACModelConfig(
            base=IDOnlyConfig(**ac_raw["base"]),
            event_layers=int(ac_raw["event_layers"]),
            auxiliary_ffn_multiplier=int(ac_raw["auxiliary_ffn_multiplier"]),
            goal_roles=int(ac_raw["goal_roles"]),
        ),
        scenario_layers=int(raw["scenario_layers"]),
        scenario_ffn_multiplier=int(raw["scenario_ffn_multiplier"]),
        scale_gate_ffn_multiplier=int(raw["scale_gate_ffn_multiplier"]),
    )


def _summary(values: Tensor) -> dict[str, float]:
    values = values.float().flatten().cpu()
    maximum_quantile_values = 1_000_000
    if values.numel() > maximum_quantile_values:
        indices = torch.linspace(
            0, values.numel() - 1, maximum_quantile_values, dtype=torch.float64
        ).long()
        quantile_values = values[indices]
    else:
        quantile_values = values
    quantiles = torch.quantile(
        quantile_values, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
    )
    return {
        "count": float(values.numel()),
        "quantile_sample_count": float(quantile_values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "minimum": float(values.min()),
        "p01": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    payload = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    raw = metadata.get("model")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint metadata has no R2 model config")
    model = R2StrongScenarioPolicy(
        _config(raw), ontology_path=MODEL_READY_DATASET / "card_ontology.json"
    )
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(arguments.device)
    model.to(device).eval()

    captured: dict[str, Tensor] = {}
    state_handle = model.state_scale_gate.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("state", output.detach())
    )
    option_handle = model.option_scale_gate.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("option", output.detach())
    )
    states: list[Tensor] = []
    options: list[Tensor] = []
    try:
        with torch.inference_mode():
            for source in iter_ac_batches(
                MODEL_READY_DATASET,
                "validation",
                batch_size=arguments.batch_size,
                shuffle=False,
                seed=20260723,
            ):
                batch = {name: value.to(device) for name, value in source.items()}
                with torch.amp.autocast(
                    "cuda", enabled=device.type == "cuda", dtype=torch.bfloat16
                ):
                    model.encode(batch)
                states.append(captured["state"].float().cpu())
                option_values = captured["option"]
                options.append(option_values[batch["option_mask"]].float().cpu())
    finally:
        state_handle.remove()
        option_handle.remove()

    report = {
        "schema_version": "0014_r2_scale_diagnostic_v1",
        "checkpoint": str(arguments.checkpoint),
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "state_scale": _summary(torch.cat(states, dim=0)),
        "legal_option_scale": _summary(torch.cat(options, dim=0)),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
