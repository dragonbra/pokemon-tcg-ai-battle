from __future__ import annotations

import torch

from rl.core.batch import collate_encoded
from rl.core.model import CandidatePolicyValueNet, ModelConfig

from .features import PTCGFeatureConfig, encode_observation


class PTCGCandidatePolicy:
    """Inference bridge from an official observation to one legal option index."""

    def __init__(
        self,
        model: CandidatePolicyValueNet,
        *,
        feature_config: PTCGFeatureConfig = PTCGFeatureConfig(),
    ) -> None:
        model_config = model.config
        expected = {
            "state_numeric_dim": feature_config.state_numeric_dim,
            "state_token_count": feature_config.state_token_count,
            "candidate_numeric_dim": feature_config.candidate_numeric_dim,
            "max_candidates": feature_config.max_candidates,
            "card_vocab_size": feature_config.card_vocab_size,
            "action_type_vocab_size": feature_config.action_type_vocab_size,
        }
        for key, value in expected.items():
            if getattr(model_config, key) != value:
                raise ValueError(f"model/feature schema mismatch for {key}")
        self.model = model.eval()
        self.feature_config = feature_config

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        model_config: ModelConfig | None = None,
        *,
        feature_config: PTCGFeatureConfig = PTCGFeatureConfig(),
        map_location: str = "cpu",
    ) -> "PTCGCandidatePolicy":
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        if model_config is None:
            metadata = payload.get("metadata") or {}
            saved_config = metadata.get("model_config")
            if not isinstance(saved_config, dict):
                raise ValueError(
                    "checkpoint lacks metadata.model_config; "
                    "pass model_config explicitly"
                )
            model_config = ModelConfig(**saved_config)
        model = CandidatePolicyValueNet(model_config)
        model.load_state_dict(payload["model"])
        return cls(model, feature_config=feature_config)

    @torch.no_grad()
    def select(self, observation: dict) -> tuple[int, float]:
        """Return ``(option_index, value_estimate)`` for the current observation."""
        encoded = encode_observation(observation, self.feature_config)
        batch = collate_encoded([encoded])
        value, logits = self.model(**batch)
        index = int(logits.argmax(dim=-1).item())
        if not bool(batch["action_mask"][0, index]):
            raise RuntimeError("model selected an option outside the legal action mask")
        return index, float(value.item())
