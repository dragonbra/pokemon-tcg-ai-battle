from __future__ import annotations

from pathlib import Path

import torch

from rl_environment.model import ModelConfig
from archive.train_legacy.alakazam_bc_rl.features import PTCGFeatureConfig
from archive.train_legacy.alakazam_bc_rl.full_action_inference import FullActionPolicy
from archive.train_legacy.alakazam_bc_rl.full_action_model import FullActionPolicyValueNet

from .model import CategoryAugmentedFullActionPolicyValueNet


class RewardWeightedFullActionPolicy(FullActionPolicy):
    """Inference loader owned by the independent stage-two project."""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "RewardWeightedFullActionPolicy":
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        metadata = payload.get("metadata") or {}
        model_config = ModelConfig(**metadata["model_config"])
        feature_config = PTCGFeatureConfig(**metadata["feature_config"])
        maximum = int(metadata.get("max_selection_count", model_config.max_candidates))
        category = metadata.get("card_category_embedding")
        if isinstance(category, dict) and category.get("enabled"):
            model: FullActionPolicyValueNet = CategoryAugmentedFullActionPolicyValueNet(
                model_config,
                card_category_lookup=torch.zeros(
                    model_config.card_vocab_size + 1, dtype=torch.long
                ),
                category_vocab_size=int(category["category_vocab_size"]),
                category_scale=float(category.get("scale", 1.0)),
                max_selection_count=maximum,
            )
        else:
            model = FullActionPolicyValueNet(
                model_config, max_selection_count=maximum
            )
        model.load_state_dict(payload["model"])
        return cls(
            model,
            feature_config=feature_config,
            deck=[int(value) for value in metadata.get("deck", [])],
            expert_id=int(metadata.get("expert_id", 1)),
            card_metadata=metadata.get("card_metadata") or {},
            device=map_location,
        )
