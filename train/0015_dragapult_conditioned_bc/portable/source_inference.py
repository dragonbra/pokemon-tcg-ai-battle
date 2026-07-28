"""CPU inference for a portable source-conditioned 0015 R2/R15 candidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .ablation import remove_initial_deck_information
from .ac_model import ACModelConfig
from .base_model import IDOnlyConfig
from .online_runtime import OnlineCausalEncoder
from .r2_model import R2ModelConfig
from .r15_model import R15ModelConfig
from .source_model import (
    SourceConditionedR2Config,
    SourceConditionedR2Policy,
    SourceConditionedR15Config,
    SourceConditionedR15Policy,
)


class SourcePolicy:
    def __init__(
        self,
        model: nn.Module,
        deck: Sequence[int],
        config: IDOnlyConfig,
        *,
        source_id: int,
        no_deck: bool,
    ) -> None:
        torch.set_num_threads(1)
        self.model = model.eval()
        self.deck = tuple(deck)
        self.config = config
        self.source_id = source_id
        self.no_deck = no_deck
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        ontology_path: str | Path,
        deck: Sequence[int],
        *,
        source_id: int,
        no_deck: bool,
    ) -> "SourcePolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        raw = metadata.get("model")
        if not isinstance(raw, dict):
            raise ValueError("0015 checkpoint metadata is missing model config")
        model_family = metadata.get("model_family", "r15")
        family_key = "r2" if model_family == "r2" else "r15"
        family_raw = raw.get(family_key)
        if not isinstance(family_raw, dict):
            raise ValueError(f"0015 checkpoint metadata is missing model.{family_key}")
        ac_raw = family_raw["ac"]
        base = IDOnlyConfig(**ac_raw["base"])
        ac = ACModelConfig(
            base=base,
            event_layers=int(ac_raw["event_layers"]),
            auxiliary_ffn_multiplier=int(ac_raw["auxiliary_ffn_multiplier"]),
            goal_roles=int(ac_raw["goal_roles"]),
        )
        if model_family == "r2":
            r2 = R2ModelConfig(
                ac=ac,
                scenario_layers=int(family_raw["scenario_layers"]),
                scenario_ffn_multiplier=int(family_raw["scenario_ffn_multiplier"]),
                scale_gate_ffn_multiplier=int(family_raw["scale_gate_ffn_multiplier"]),
            )
            config = SourceConditionedR2Config(
                r2=r2,
                source_vocabulary_size=int(raw["source_vocabulary_size"]),
                source_initial_scale=float(raw["source_initial_scale"]),
            )
            model = SourceConditionedR2Policy(config, ontology_path=ontology_path)
        elif model_family == "r15":
            r15 = R15ModelConfig(
                ac=ac,
                scenario_layers=int(family_raw["scenario_layers"]),
                scenario_ffn_multiplier=int(family_raw["scenario_ffn_multiplier"]),
                scale_gate_ffn_multiplier=int(family_raw["scale_gate_ffn_multiplier"]),
                option_initial_scale=float(family_raw["option_initial_scale"]),
            )
            config = SourceConditionedR15Config(
                r15=r15,
                source_vocabulary_size=int(raw["source_vocabulary_size"]),
                source_initial_scale=float(raw["source_initial_scale"]),
            )
            model = SourceConditionedR15Policy(config, ontology_path=ontology_path)
        elif model_family == "r15_rule_contract":
            r15 = R15ModelConfig(
                ac=ac,
                scenario_layers=int(family_raw["scenario_layers"]),
                scenario_ffn_multiplier=int(family_raw["scenario_ffn_multiplier"]),
                scale_gate_ffn_multiplier=int(family_raw["scale_gate_ffn_multiplier"]),
                option_initial_scale=float(family_raw["option_initial_scale"]),
            )
            from .rule_contract_model import (
                SourceConditionedR15RuleConfig,
                SourceConditionedR15RulePolicy,
            )

            config = SourceConditionedR15RuleConfig(
                r15=r15,
                source_vocabulary_size=int(raw["source_vocabulary_size"]),
                source_initial_scale=float(raw["source_initial_scale"]),
                rule_layers=int(raw["rule_layers"]),
                rule_ffn_multiplier=int(raw["rule_ffn_multiplier"]),
                rule_initial_scale=float(raw["rule_initial_scale"]),
                rule_model_width=int(raw.get("rule_model_width", 320)),
                rule_heads=int(raw.get("rule_heads", 8)),
            )
            model = SourceConditionedR15RulePolicy(config, ontology_path=ontology_path)
        else:
            raise ValueError(f"unsupported 0015 model family: {model_family}")
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, deck, base, source_id=source_id, no_deck=no_deck)

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: dict[str, Any]) -> list[int]:
        current = observation.get("current") or {}
        actor = current.get("yourIndex")
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor")
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = OnlineCausalEncoder(actor, self.deck, self.config)
        batch = self.encoder.encode(observation)
        batch["source_id"] = torch.tensor([self.source_id], dtype=torch.long)
        if self.no_deck:
            batch = remove_initial_deck_information(batch)
        with torch.inference_mode():
            return self.model.greedy_action(batch)


SourceR15Policy = SourcePolicy

__all__ = ["SourcePolicy", "SourceR15Policy"]
