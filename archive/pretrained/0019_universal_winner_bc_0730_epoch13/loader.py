"""Strict loader for the frozen 0019 universal winner BC checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from model_source.ac_model import ACModelConfig
from model_source.base_model import IDOnlyConfig
from model_source.r15_model import R15ModelConfig
from model_source.source_model import SourceModelConfig
from model_source.source_r15_model import SourceConditionedR15Policy


ARCHIVE_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ARCHIVE_ROOT / "model.pt"
ONTOLOGY_PATH = ARCHIVE_ROOT / "contracts/card_ontology.json"


def _model_config() -> tuple[R15ModelConfig, SourceModelConfig]:
    base = IDOnlyConfig(
        max_card_id=2048,
        d_model=320,
        heads=8,
        encoder_layers=4,
        ffn_multiplier=3,
        max_entities=192,
        max_options=128,
        max_action_steps=64,
        dropout=0.1,
    )
    ac = ACModelConfig(
        base=base,
        event_layers=1,
        auxiliary_ffn_multiplier=2,
        goal_roles=4,
    )
    return (
        R15ModelConfig(
            ac=ac,
            scenario_layers=2,
            scenario_ffn_multiplier=3,
            scale_gate_ffn_multiplier=2,
            option_initial_scale=0.35,
        ),
        SourceModelConfig(vocabulary_size=509, initial_scale=0.05),
    )


def load_model(
    device: str | torch.device = "cpu",
    *,
    eval_mode: bool = True,
) -> SourceConditionedR15Policy:
    """Instantiate the bound architecture and strictly load Epoch 13 weights."""
    model_config, source_config = _model_config()
    model = SourceConditionedR15Policy(
        model_config,
        source_config,
        ontology_path=ONTOLOGY_PATH,
    )
    payload: dict[str, Any] = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("epoch") != 13:
        raise ValueError("archive checkpoint is not 0019 Epoch 13")
    model.load_state_dict(payload["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 17_756_162:
        raise ValueError("archive model parameter count does not match its contract")
    model.to(device)
    if eval_mode:
        model.eval()
    return model


def neutral_source_id(batch_size: int, device: str | torch.device) -> torch.Tensor:
    """Return source_id=0, the audited neutral persona for unseen policies."""
    return torch.zeros(batch_size, dtype=torch.long, device=device)


__all__ = ["load_model", "neutral_source_id"]
