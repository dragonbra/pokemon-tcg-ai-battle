from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .features.compiler import compiler_sha256
from .model.registry import create_model
from .model.variants import ModelConfig, SemanticGoalPolicy
from .runtime.session import PolicySession
from .training.batching import collate_records


CHECKPOINT_SCHEMA = "semantic_goal_inference_checkpoint_v1"
EXPECTED_COMPILER_SHA256 = (
    "60064dffb6d2780959589a83d57596bcc41f5834cd9146c24259a55c6503cf1d"
)


class SemanticGoalInference:
    def __init__(
        self,
        model: SemanticGoalPolicy,
        deck: Sequence[int],
    ) -> None:
        self.model = model.eval()
        self.deck = tuple(int(card_id) for card_id in deck)
        self.session: PolicySession | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        deck: Sequence[int],
    ) -> SemanticGoalInference:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("wrong 0013 inference checkpoint schema")
        if payload.get("variant") != "M0":
            raise ValueError("0013 candidate requires the M0 variant")
        if payload.get("feature_compiler_sha256") != EXPECTED_COMPILER_SHA256:
            raise ValueError("checkpoint feature compiler digest mismatch")
        if compiler_sha256() != EXPECTED_COMPILER_SHA256:
            raise ValueError("packaged feature compiler digest mismatch")
        config_value = payload.get("model_config")
        if not isinstance(config_value, Mapping):
            raise ValueError("checkpoint is missing model_config")
        model = create_model("M0", ModelConfig(**dict(config_value)))
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, deck)

    def reset(self) -> None:
        self.session = None

    @staticmethod
    def _actor(observation: Mapping[str, Any]) -> int:
        current = observation.get("current")
        actor = current.get("yourIndex") if isinstance(current, Mapping) else None
        if isinstance(actor, bool) or not isinstance(actor, int):
            raise ValueError("observation is missing current.yourIndex")
        return actor

    def select(self, observation: Mapping[str, Any]) -> list[int]:
        actor = self._actor(observation)
        if self.session is None:
            self.session = PolicySession.new_game(actor, self.deck)
        elif self.session.actor != actor:
            raise ValueError("actor changed without a game reset")

        decision = self.session.observe(observation)
        batch = collate_records(
            [
                {
                    "_compiled_features": decision.tensors,
                    "ordered_action": (),
                    "action_termination": "forced_max",
                }
            ]
        )
        with torch.inference_mode():
            result = self.model.deterministic_actions(batch)
        if not result.legal[0]:
            raise RuntimeError("M0 decoder failed to produce a legal full action")
        action = list(result.sequences[0])
        self.session.record_action(action)
        return action
