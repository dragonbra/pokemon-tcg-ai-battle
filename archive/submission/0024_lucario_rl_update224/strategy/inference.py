"""Fail-closed neutral inference for the frozen 0019 epoch-13 foundation."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from .ac_model import ACModelConfig
from .base_model import IDOnlyConfig
from .online_runtime import OnlineCausalEncoder
from .r15_model import R15ModelConfig
from .source_model import SourceModelConfig
from .source_r15_model import SourceConditionedR15Policy


EXPECTED_CHECKPOINT_SHA256 = (
    "5ef30a31c880e3658ad3ffa38dc92e3925fddfc3b0a26051f610f82a1c38ceac"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "8144c63e512a2a00fabaaf2b19cd002c5b59c25fb0763a112256848460113a4d"
)
DEPLOYMENT_SOURCE_ID = 0


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_deck(deck: Sequence[int]) -> tuple[int, ...]:
    if len(deck) != 60:
        raise ValueError(f"registered deck must contain exactly 60 cards, got {len(deck)}")
    normalized = tuple(deck)
    if any(
        isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0
        for card_id in normalized
    ):
        raise ValueError("registered deck must contain positive integer card IDs")
    return normalized


def registered_deck_tensors(deck: Sequence[int]) -> dict[str, torch.Tensor]:
    """Expose the exact tensor contract used by the online causal encoder."""
    counts = Counter(validate_deck(deck))
    card_ids = sorted(counts)
    return {
        "registered_card_ids": torch.tensor([card_ids], dtype=torch.long),
        "registered_multiplicity": torch.tensor(
            [[counts[card_id] for card_id in card_ids]], dtype=torch.float32
        ),
        "registered_mask": torch.ones((1, len(card_ids)), dtype=torch.bool),
    }


def legal_fallback(observation: dict[str, Any]) -> list[int]:
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


def _model_config(metadata: dict[str, Any]) -> tuple[R15ModelConfig, SourceModelConfig]:
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
    model = R15ModelConfig(
        ac=ac,
        scenario_layers=int(raw["scenario_layers"]),
        scenario_ffn_multiplier=int(raw["scenario_ffn_multiplier"]),
        scale_gate_ffn_multiplier=int(raw["scale_gate_ffn_multiplier"]),
        option_initial_scale=float(raw["option_initial_scale"]),
    )
    return model, SourceModelConfig(**source_raw)


class FrozenNeutralPolicy:
    def __init__(
        self,
        model: SourceConditionedR15Policy,
        deck: Sequence[int],
        config: IDOnlyConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        torch.set_num_threads(1)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda is not available")
        self.model = model.to(self.device).eval()
        self.deck = validate_deck(deck)
        self.config = config
        self.source_id = DEPLOYMENT_SOURCE_ID
        self.encoder: OnlineCausalEncoder | None = None
        self.fallback_count = 0
        self.last_fallback_reason: str | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        ontology_path: str | Path,
        deck: Sequence[int],
        *,
        device: str | torch.device = "cpu",
    ) -> "FrozenNeutralPolicy":
        checkpoint_path = Path(checkpoint)
        ontology = Path(ontology_path)
        if _sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("checkpoint hash does not match the frozen 0020 foundation")
        if _sha256(ontology) != EXPECTED_ONTOLOGY_SHA256:
            raise ValueError("ontology hash does not match the frozen 0020 foundation")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if set(payload) != {"model", "epoch", "global_step", "metadata"}:
            raise ValueError("checkpoint is not the audited model-only payload")
        if payload["epoch"] != 13 or payload["global_step"] != 337194:
            raise ValueError("checkpoint epoch/global step does not match the foundation")
        metadata = payload["metadata"]
        if metadata.get("deployment_source_id") != DEPLOYMENT_SOURCE_ID:
            raise ValueError("checkpoint deployment source is not neutral")
        config, source_config = _model_config(metadata)
        model = SourceConditionedR15Policy(
            config, source_config, ontology_path=ontology
        )
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, deck, config.ac.base, device=device)

    def reset(self) -> None:
        self.encoder = None
        self.last_fallback_reason = None

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
            batch["source_id"] = torch.tensor([DEPLOYMENT_SOURCE_ID], dtype=torch.long)
            batch = _move_batch(batch, self.device)
            with torch.inference_mode():
                return self.model.greedy_action(batch)
        except (IndexError, RuntimeError, ValueError) as error:
            self.fallback_count += 1
            self.last_fallback_reason = f"{type(error).__name__}: {error}"
            self.encoder = None
            return legal_fallback(observation)


__all__ = [
    "DEPLOYMENT_SOURCE_ID",
    "EXPECTED_CHECKPOINT_SHA256",
    "FrozenNeutralPolicy",
    "legal_fallback",
    "registered_deck_tensors",
    "validate_deck",
]
