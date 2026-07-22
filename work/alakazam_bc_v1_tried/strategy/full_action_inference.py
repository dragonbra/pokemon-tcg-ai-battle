"""Inference bridge for the pure full-action BC candidate."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from .core_model import ModelConfig

from .features import PTCGFeatureConfig, encode_observation
from .full_action_model import FullActionPolicyValueNet


class FullActionPolicy:
    """Stateful candidate policy with no teacher or rules fallback."""

    def __init__(
        self,
        model: FullActionPolicyValueNet,
        *,
        feature_config: PTCGFeatureConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.feature_config = feature_config
        self.device = torch.device(device)
        self.history: list[dict[str, int]] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "FullActionPolicy":
        try:
            payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        except TypeError:  # Compatibility with PyTorch releases before weights_only.
            payload = torch.load(checkpoint, map_location=map_location)
        metadata = payload.get("metadata") or {}
        model_config = ModelConfig(**metadata["model_config"])
        feature_config = PTCGFeatureConfig(**metadata["feature_config"])
        max_count = int(metadata.get("max_selection_count", model_config.max_candidates))
        model = FullActionPolicyValueNet(model_config, max_selection_count=max_count)
        model.load_state_dict(payload["model"])
        return cls(model, feature_config=feature_config, device=map_location)

    def reset(self) -> None:
        self.history.clear()

    @staticmethod
    def _int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def select(self, observation: dict[str, Any]) -> list[int]:
        if observation.get("select") is None or observation.get("current") is None:
            self.reset()
            return []

        model_observation = copy.deepcopy(observation)
        model_observation["rl_history"] = list(self.history)
        encoded = encode_observation(model_observation, self.feature_config)
        batch = {
            "state_numeric": torch.tensor([encoded["state_numeric"]], dtype=torch.float32),
            "state_card_ids": torch.tensor([encoded["state_card_ids"]], dtype=torch.long),
            "action_type_ids": torch.tensor([encoded["action_type_ids"]], dtype=torch.long),
            "action_card_ids": torch.tensor([encoded["action_card_ids"]], dtype=torch.long),
            "action_target_ids": torch.tensor([encoded["action_target_ids"]], dtype=torch.long),
            "action_numeric": torch.tensor([encoded["action_numeric"]], dtype=torch.float32),
            "action_mask": torch.tensor([encoded["action_mask"]], dtype=torch.bool),
        }
        batch = {key: value.to(self.device) for key, value in batch.items()}
        select = observation.get("select") or {}
        minimum = max(0, self._int(select.get("minCount"), 0))
        maximum = min(
            self.model.max_selection_count,
            self._int(select.get("maxCount"), minimum),
        )
        if maximum < minimum:
            raise ValueError("simulator supplied an invalid minCount/maxCount range")

        with torch.no_grad():
            _, logits, count_logits = self.model.forward_with_count(**batch)
            count_mask = torch.zeros_like(count_logits, dtype=torch.bool)
            count_mask[:, minimum : maximum + 1] = True
            masked_counts = count_logits.masked_fill(
                ~count_mask, torch.finfo(count_logits.dtype).min
            )
            count = int(masked_counts.argmax())
            candidate_count = sum(encoded["action_mask"])
            count = min(count, candidate_count)
            if count:
                indices = torch.topk(logits[0], k=count).indices.tolist()
                action = sorted(int(index) for index in indices)
            else:
                action = []

        if self._int(select.get("type"), -1) == 0 and self._int(select.get("context"), -1) == 0:
            options = select.get("option") or []
            if len(action) == 1 and 0 <= action[0] < len(options):
                option = options[action[0]]
                if isinstance(option, dict):
                    self.history.append(
                        {
                            "type": self._int(option.get("type"), 0),
                            "cardId": self._int(option.get("cardId"), 0),
                            "attackId": self._int(option.get("attackId"), 0),
                        }
                    )
                    del self.history[:-32]
        return action
