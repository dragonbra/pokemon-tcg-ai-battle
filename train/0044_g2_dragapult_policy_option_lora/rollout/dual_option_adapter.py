"""Project-local CUDA adapter preserving policy/value Option isolation."""

from __future__ import annotations

from typing import Any, Sequence

from ptcg_cuda_engine.semantic0031_bridge import Semantic0031DeviceAdapter


class DualOptionSemantic0031DeviceAdapter(Semantic0031DeviceAdapter):
    """Expose LoRA policy options to decode and base options to Value."""

    def __init__(
        self, actor_critic: Any, registered_deck: Sequence[int] | None,
        *, max_select: int = 64,
    ) -> None:
        self.actor_critic = actor_critic
        self._value_options: dict[int, Any] = {}
        super().__init__(actor_critic.actor, registered_deck, max_select=max_select)

    def _encode_options(self, batch: Any, state: Any) -> Any:
        option_encoder = self.model.option_encoder
        inputs = self._encode_option_inputs(batch, state)
        prefix = option_encoder.encode_prefix(batch, state, inputs)
        value_options = option_encoder.encode_final(batch, state, prefix)
        policy_options = self.actor_critic.policy_option_lora(
            option_encoder, batch, state, prefix
        )
        if self.actor_critic.policy_option_lora.is_zero_delta():
            policy_options = value_options + (
                policy_options - policy_options.detach()
            )
        self._value_options[id(policy_options)] = value_options
        return policy_options

    def take_value_options(self, policy_options: Any) -> Any:
        try:
            return self._value_options.pop(id(policy_options))
        except KeyError as error:
            raise RuntimeError(
                "0044 CUDA routing lost the policy/value Option branch binding"
            ) from error


__all__ = ["DualOptionSemantic0031DeviceAdapter"]
