"""CPU inference for the source-conditioned 0016 R15 policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .ac_model import ACModelConfig
from .base_model import IDOnlyConfig
from .online_runtime import OnlineCausalEncoder
from .r15_model import R15ModelConfig
from .source_model import SourceModelConfig
from .source_r15_model import SourceConditionedR15Policy


def legal_fallback(observation: dict[str, Any]) -> list[int]:
    """Return a count-valid action without assuming any model capacity."""
    select = observation.get("select")
    if not isinstance(select, dict):
        raise ValueError("observation has no select payload")
    options = select.get("option")
    if not isinstance(options, list):
        raise ValueError("select.option is not a list")
    minimum = select.get("minCount", 0)
    maximum = select.get("maxCount", len(options))
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (minimum, maximum)
    ):
        raise ValueError("select bounds are not exact integers")
    if not 0 <= minimum <= maximum <= len(options):
        raise ValueError("select bounds are inconsistent with options")
    return list(range(minimum))


class SourceR15Policy:
    def __init__(
        self,
        model: SourceConditionedR15Policy,
        deck: Sequence[int],
        config: IDOnlyConfig,
        source_id: int,
    ) -> None:
        torch.set_num_threads(1)
        self.model = model.eval()
        self.deck = tuple(deck)
        self.config = config
        self.source_id = source_id
        self.encoder: OnlineCausalEncoder | None = None
        self.fallback_count = 0
        self.last_fallback_reason: str | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        ontology_path: str | Path,
        deck: Sequence[int],
    ) -> "SourceR15Policy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        raw = metadata.get("model")
        source_raw = metadata.get("source_model")
        if not isinstance(raw, dict) or not isinstance(raw.get("ac"), dict):
            raise ValueError("checkpoint metadata is missing model.ac")
        if not isinstance(source_raw, dict):
            raise ValueError("checkpoint metadata is missing source_model")
        ac_raw = raw["ac"]
        base = IDOnlyConfig(**ac_raw["base"])
        ac = ACModelConfig(
            base=base,
            event_layers=int(ac_raw["event_layers"]),
            auxiliary_ffn_multiplier=int(ac_raw["auxiliary_ffn_multiplier"]),
            goal_roles=int(ac_raw["goal_roles"]),
        )
        config = R15ModelConfig(
            ac=ac,
            scenario_layers=int(raw["scenario_layers"]),
            scenario_ffn_multiplier=int(raw["scenario_ffn_multiplier"]),
            scale_gate_ffn_multiplier=int(raw["scale_gate_ffn_multiplier"]),
            option_initial_scale=float(raw["option_initial_scale"]),
        )
        source_config = SourceModelConfig(**source_raw)
        model = SourceConditionedR15Policy(
            config,
            source_config,
            ontology_path=ontology_path,
        )
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, deck, base, source_id=int(metadata["deployment_source_id"]))

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: dict[str, Any]) -> list[int]:
        current = observation.get("current") or {}
        actor = current.get("yourIndex")
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor")
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = OnlineCausalEncoder(actor, self.deck, self.config)
        select = observation.get("select") or {}
        options = select.get("option") or []
        minimum = select.get("minCount", 0)
        if len(options) > self.config.max_options or (
            isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and minimum > self.config.max_action_steps
        ):
            self.fallback_count += 1
            self.last_fallback_reason = "model_capacity"
            return legal_fallback(observation)
        try:
            batch = self.encoder.encode(observation)
            batch["source_id"] = torch.tensor([self.source_id], dtype=torch.long)
            with torch.inference_mode():
                return self.model.greedy_action(batch)
        except (IndexError, RuntimeError, ValueError) as error:
            self.fallback_count += 1
            self.last_fallback_reason = f"{type(error).__name__}: {error}"
            self.encoder = None
            return legal_fallback(observation)


__all__ = ["SourceR15Policy", "legal_fallback"]
