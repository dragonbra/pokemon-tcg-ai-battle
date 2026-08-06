"""Vectorized frozen 0022 decoder heads over shared 0033 POD encodings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from ..contract import PodNativeBatch
from .action_decoder import ActionDecoder, OrderedActions


SCHEMA = "0022_league_decoder_model_only_v1"
DECODER_NAMES = (
    "pointer_key.weight",
    "pointer_query.weight",
    "option_bias.weight",
    "option_bias.bias",
    "decoder_init.weight",
    "decoder_init.bias",
    "decoder.weight_ih",
    "decoder.weight_hh",
    "decoder.bias_ih",
    "decoder.bias_hh",
    "stop.0.weight",
    "stop.0.bias",
    "stop.2.weight",
    "stop.2.bias",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenHeadIdentity:
    deck_id: str
    deck_sha256: str
    checkpoint: str
    checkpoint_sha256: str
    foundation_sha256: str
    update: int


class FrozenDeckHeads(nn.Module):
    """Stack decoder parameters so every lane selects one head on-device."""

    def __init__(self, states: list[Mapping[str, Tensor]], identities: list[FrozenHeadIdentity]):
        super().__init__()
        if not states or len(states) != len(identities):
            raise ValueError("states and identities must be nonempty and aligned")
        self.identities = tuple(identities)
        for name in DECODER_NAMES:
            values = [state[f"decoder.{name}"].detach().clone() for state in states]
            self.register_buffer(name.replace(".", "__"), torch.stack(values))
        self.requires_grad_(False)

    @classmethod
    def from_0019_foundation(cls, checkpoint: Path) -> "FrozenDeckHeads":
        checkpoint = Path(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if int(payload.get("epoch", -1)) != 13:
            raise ValueError("shared frozen foundation must be 0019 epoch 13")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("0019 foundation checkpoint has no metadata")
        if metadata.get("project_id") != "0019_universal_winner_bc":
            raise ValueError("shared frozen foundation has the wrong project identity")
        state = payload.get("model")
        if not isinstance(state, Mapping):
            raise ValueError("0019 foundation checkpoint has no model state")
        missing = [name for name in DECODER_NAMES if name not in state]
        if missing:
            raise ValueError(f"0019 foundation is missing decoder tensors: {missing}")
        decoder_state = {f"decoder.{name}": state[name] for name in DECODER_NAMES}
        digest = _sha256(checkpoint)
        identity = FrozenHeadIdentity(
            deck_id="0019_epoch13_shared_foundation",
            deck_sha256="shared_exact_deck_conditioned_foundation",
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest,
            foundation_sha256=digest,
            update=int(payload.get("global_step", -1)),
        )
        return cls([decoder_state], [identity])

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: Iterable[Path],
        *,
        expected_deck_hashes: Mapping[str, str] | None = None,
    ) -> "FrozenDeckHeads":
        states: list[Mapping[str, Tensor]] = []
        identities: list[FrozenHeadIdentity] = []
        foundation: str | None = None
        for path in checkpoints:
            path = Path(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("schema_version") != SCHEMA:
                raise ValueError(f"unsupported frozen head schema: {path}")
            deck_id = str(payload.get("deck_id"))
            deck_hash = str(payload.get("deck_sha256"))
            if path.stem != deck_id:
                raise ValueError(f"checkpoint/deck identity mismatch: {path}")
            if expected_deck_hashes is not None and expected_deck_hashes.get(deck_id) != deck_hash:
                raise ValueError(f"checkpoint/deck hash mismatch: {deck_id}")
            current_foundation = str(payload.get("foundation_sha256"))
            if foundation is None:
                foundation = current_foundation
            elif current_foundation != foundation:
                raise ValueError("frozen heads mix foundation checkpoints")
            state = payload.get("state")
            if not isinstance(state, Mapping):
                raise ValueError(f"checkpoint has no model state: {path}")
            missing = [f"decoder.{name}" for name in DECODER_NAMES if f"decoder.{name}" not in state]
            if missing:
                raise ValueError(f"checkpoint missing decoder tensors: {missing}")
            states.append(state)
            identities.append(
                FrozenHeadIdentity(
                    deck_id=deck_id,
                    deck_sha256=deck_hash,
                    checkpoint=str(path),
                    checkpoint_sha256=_sha256(path),
                    foundation_sha256=current_foundation,
                    update=int(payload.get("update", -1)),
                )
            )
        return cls(states, identities)

    def _parameter(self, name: str, head_indices: Tensor) -> Tensor:
        return getattr(self, name.replace(".", "__")).index_select(0, head_indices.long())

    def _routed_parameter(
        self,
        name: str,
        head_indices: Tensor,
        focal_mask: Tensor,
        learner_parameter: Tensor,
    ) -> Tensor:
        frozen = self._parameter(name, head_indices)
        learner = learner_parameter.unsqueeze(0).expand_as(frozen)
        shape = (focal_mask.shape[0],) + (1,) * (frozen.ndim - 1)
        return torch.where(focal_mask.view(shape), learner, frozen)

    def _linear(self, x: Tensor, weight_name: str, bias_name: str | None, heads: Tensor) -> Tensor:
        weight = self._parameter(weight_name, heads)
        result = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)
        if bias_name is not None:
            result = result + self._parameter(bias_name, heads)
        return result

    def _linear_options(
        self, x: Tensor, weight_name: str, bias_name: str | None, heads: Tensor
    ) -> Tensor:
        weight = self._parameter(weight_name, heads)
        result = torch.bmm(x, weight.transpose(1, 2))
        if bias_name is not None:
            result = result + self._parameter(bias_name, heads).unsqueeze(1)
        return result

    def _gru(self, x: Tensor, hidden: Tensor, heads: Tensor) -> Tensor:
        gate_x = self._linear(x, "decoder.weight_ih", "decoder.bias_ih", heads)
        gate_h = self._linear(hidden, "decoder.weight_hh", "decoder.bias_hh", heads)
        input_reset, input_update, input_new = gate_x.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = gate_h.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        new = torch.tanh(input_new + reset * hidden_new)
        return new + update * (hidden - new)

    def greedy(
        self,
        batch: PodNativeBatch,
        options: Tensor,
        state_summary: Tensor,
        head_indices: Tensor,
        *,
        max_action_steps: int = 64,
    ) -> OrderedActions:
        if head_indices.shape != (batch.batch_size,):
            raise ValueError("head_indices must have shape [B]")
        hidden = torch.tanh(
            self._linear(state_summary, "decoder_init.weight", "decoder_init.bias", head_indices)
        )
        keys = self._linear_options(options, "pointer_key.weight", None, head_indices)
        option_bias = self._linear_options(
            options, "option_bias.weight", "option_bias.bias", head_indices
        ).squeeze(-1)
        sequences = torch.full(
            (batch.batch_size, max_action_steps), -1, dtype=torch.long, device=options.device
        )
        lengths = torch.zeros(batch.batch_size, dtype=torch.long, device=options.device)
        available = batch.option_mask.clone()
        active = torch.ones(batch.batch_size, dtype=torch.bool, device=options.device)
        rows = torch.arange(batch.batch_size, device=options.device)
        for step in range(max_action_steps):
            query = self._linear(hidden, "pointer_query.weight", None, head_indices)
            pointer = (query.unsqueeze(1) * keys).sum(-1) / options.shape[-1] ** 0.5
            pointer = (pointer + option_bias).masked_fill(
                ~(available & lengths.lt(batch.max_count).unsqueeze(1)),
                torch.finfo(pointer.dtype).min,
            )
            stop_hidden = torch.nn.functional.gelu(
                self._linear(hidden, "stop.0.weight", "stop.0.bias", head_indices)
            )
            stop = self._linear(stop_hidden, "stop.2.weight", "stop.2.bias", head_indices)
            stop = stop.masked_fill(
                ~lengths.ge(batch.min_count).unsqueeze(1), torch.finfo(stop.dtype).min
            )
            scores = torch.cat((pointer, stop), dim=1)
            scores = scores.masked_fill(~active.unsqueeze(1), torch.finfo(scores.dtype).min)
            scores[:, options.shape[1]] = torch.where(
                active, scores[:, options.shape[1]], torch.zeros_like(scores[:, options.shape[1]])
            )
            choice = scores.argmax(dim=1)
            selecting = active & choice.ne(options.shape[1])
            chosen = choice.clamp_max(options.shape[1] - 1)
            sequences[:, step] = torch.where(selecting, chosen, sequences[:, step])
            selected = options[rows, chosen]
            advanced = self._gru(selected, hidden, head_indices)
            hidden = torch.where(selecting.unsqueeze(1), advanced, hidden)
            available[rows, chosen] &= ~selecting
            lengths += selecting.long()
            active = selecting & lengths.lt(batch.max_count)
        legal = lengths.ge(batch.min_count) & lengths.le(batch.max_count)
        return OrderedActions(sequences, lengths, legal)

    def provenance(self) -> list[dict[str, Any]]:
        return [asdict(identity) for identity in self.identities]

    def grouped_greedy(
        self,
        batch: PodNativeBatch,
        options: Tensor,
        state_summary: Tensor,
        *,
        max_action_steps: int = 64,
    ) -> OrderedActions:
        heads = len(self.identities)
        if batch.batch_size % heads:
            raise ValueError("batch must contain an equal fixed capacity per frozen head")
        capacity = batch.batch_size // heads

        def grouped(name: str) -> Tensor:
            return getattr(self, name.replace(".", "__"))

        def linear(x: Tensor, weight_name: str, bias_name: str | None = None) -> Tensor:
            weight = grouped(weight_name)
            result = torch.matmul(x.unsqueeze(-2), weight.transpose(1, 2).unsqueeze(1)).squeeze(-2)
            if bias_name is not None:
                result = result + grouped(bias_name).unsqueeze(1)
            return result

        def linear_options(
            x: Tensor, weight_name: str, bias_name: str | None = None
        ) -> Tensor:
            weight = grouped(weight_name)
            result = torch.matmul(x, weight.transpose(1, 2).unsqueeze(1))
            if bias_name is not None:
                result = result + grouped(bias_name).unsqueeze(1).unsqueeze(1)
            return result

        state = state_summary.view(heads, capacity, -1)
        option_tokens = options.view(heads, capacity, options.shape[1], options.shape[2])
        option_mask = batch.option_mask.view(heads, capacity, -1)
        minimum = batch.min_count.view(heads, capacity)
        maximum = batch.max_count.view(heads, capacity)
        hidden = torch.tanh(linear(state, "decoder_init.weight", "decoder_init.bias"))
        keys = linear_options(option_tokens, "pointer_key.weight")
        option_bias = linear_options(
            option_tokens, "option_bias.weight", "option_bias.bias"
        ).squeeze(-1)
        sequences = torch.full(
            (heads, capacity, max_action_steps), -1, dtype=torch.long, device=options.device
        )
        lengths = torch.zeros(heads, capacity, dtype=torch.long, device=options.device)
        available = option_mask.clone()
        active = torch.ones(heads, capacity, dtype=torch.bool, device=options.device)
        rows_h = torch.arange(heads, device=options.device).view(-1, 1)
        rows_c = torch.arange(capacity, device=options.device).view(1, -1)
        for step in range(max_action_steps):
            query = linear(hidden, "pointer_query.weight")
            pointer = (query.unsqueeze(2) * keys).sum(-1) / options.shape[-1] ** 0.5
            pointer = (pointer + option_bias).masked_fill(
                ~(available & lengths.lt(maximum).unsqueeze(2)),
                torch.finfo(pointer.dtype).min,
            )
            stop_hidden = torch.nn.functional.gelu(
                linear(hidden, "stop.0.weight", "stop.0.bias")
            )
            stop = linear(stop_hidden, "stop.2.weight", "stop.2.bias").masked_fill(
                ~lengths.ge(minimum).unsqueeze(2), torch.finfo(pointer.dtype).min
            )
            scores = torch.cat((pointer, stop), dim=2)
            scores = scores.masked_fill(~active.unsqueeze(2), torch.finfo(scores.dtype).min)
            scores[:, :, options.shape[1]] = torch.where(
                active,
                scores[:, :, options.shape[1]],
                torch.zeros_like(scores[:, :, options.shape[1]]),
            )
            choice = scores.argmax(dim=2)
            selecting = active & choice.ne(options.shape[1])
            chosen = choice.clamp_max(options.shape[1] - 1)
            sequences[:, :, step] = torch.where(
                selecting, chosen, sequences[:, :, step]
            )
            selected = option_tokens[rows_h, rows_c, chosen]
            gate_x = linear(selected, "decoder.weight_ih", "decoder.bias_ih")
            gate_h = linear(hidden, "decoder.weight_hh", "decoder.bias_hh")
            ix_r, ix_z, ix_n = gate_x.chunk(3, dim=-1)
            ih_r, ih_z, ih_n = gate_h.chunk(3, dim=-1)
            reset = torch.sigmoid(ix_r + ih_r)
            update = torch.sigmoid(ix_z + ih_z)
            new = torch.tanh(ix_n + reset * ih_n)
            advanced = new + update * (hidden - new)
            hidden = torch.where(selecting.unsqueeze(2), advanced, hidden)
            available[rows_h, rows_c, chosen] &= ~selecting
            lengths += selecting.long()
            active = selecting & lengths.lt(maximum)
        flat_lengths = lengths.reshape(-1)
        return OrderedActions(
            sequences.reshape(batch.batch_size, max_action_steps),
            flat_lengths,
            flat_lengths.ge(batch.min_count) & flat_lengths.le(batch.max_count),
        )

    def routed_generate(
        self,
        batch: PodNativeBatch,
        options: Tensor,
        state_summary: Tensor,
        head_indices: Tensor,
        focal_mask: Tensor,
        learner: ActionDecoder,
        *,
        stochastic_focal: bool,
    ) -> OrderedActions:
        def parameter(frozen_name: str, learner_parameter: Tensor) -> Tensor:
            return self._routed_parameter(
                frozen_name, head_indices, focal_mask, learner_parameter
            )

        def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
            result = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)
            return result if bias is None else result + bias

        def linear_options(
            x: Tensor, weight: Tensor, bias: Tensor | None = None
        ) -> Tensor:
            result = torch.bmm(x, weight.transpose(1, 2))
            return result if bias is None else result + bias.unsqueeze(1)

        initial_weight = parameter("decoder_init.weight", learner.initial.weight)
        initial_bias = parameter("decoder_init.bias", learner.initial.bias)
        key_weight = parameter("pointer_key.weight", learner.key.weight)
        query_weight = parameter("pointer_query.weight", learner.query.weight)
        option_bias_weight = parameter("option_bias.weight", learner.option_bias.weight)
        option_bias_bias = parameter("option_bias.bias", learner.option_bias.bias)
        weight_ih = parameter("decoder.weight_ih", learner.recurrent.weight_ih)
        weight_hh = parameter("decoder.weight_hh", learner.recurrent.weight_hh)
        bias_ih = parameter("decoder.bias_ih", learner.recurrent.bias_ih)
        bias_hh = parameter("decoder.bias_hh", learner.recurrent.bias_hh)
        stop0_weight = parameter("stop.0.weight", learner.stop[0].weight)
        stop0_bias = parameter("stop.0.bias", learner.stop[0].bias)
        stop2_weight = parameter("stop.2.weight", learner.stop[2].weight)
        stop2_bias = parameter("stop.2.bias", learner.stop[2].bias)

        hidden = torch.tanh(linear(state_summary, initial_weight, initial_bias))
        keys = linear_options(options, key_weight)
        option_bias = linear_options(options, option_bias_weight, option_bias_bias).squeeze(-1)
        sequences = torch.full(
            (batch.batch_size, learner.config.max_action_steps),
            -1,
            dtype=torch.long,
            device=options.device,
        )
        lengths = torch.zeros(batch.batch_size, dtype=torch.long, device=options.device)
        available = batch.option_mask.clone()
        active = torch.ones(batch.batch_size, dtype=torch.bool, device=options.device)
        logprob = torch.zeros(batch.batch_size, dtype=options.dtype, device=options.device)
        rows = torch.arange(batch.batch_size, device=options.device)
        for step in range(learner.config.max_action_steps):
            query = linear(hidden, query_weight)
            pointer = (query.unsqueeze(1) * keys).sum(-1) / options.shape[-1] ** 0.5
            pointer = (pointer + option_bias).masked_fill(
                ~(available & lengths.lt(batch.max_count).unsqueeze(1)),
                torch.finfo(pointer.dtype).min,
            )
            stop_hidden = torch.nn.functional.gelu(
                linear(hidden, stop0_weight, stop0_bias)
            )
            stop = linear(stop_hidden, stop2_weight, stop2_bias).masked_fill(
                ~lengths.ge(batch.min_count).unsqueeze(1), torch.finfo(pointer.dtype).min
            )
            scores = torch.cat((pointer, stop), dim=1)
            scores = scores.masked_fill(~active.unsqueeze(1), torch.finfo(scores.dtype).min)
            scores[:, options.shape[1]] = torch.where(
                active, scores[:, options.shape[1]], torch.zeros_like(scores[:, options.shape[1]])
            )
            greedy = scores.argmax(dim=1)
            if stochastic_focal:
                noise = torch.empty_like(scores).exponential_().log().neg()
                sampled = (scores + noise).argmax(dim=1)
                choice = torch.where(focal_mask, sampled, greedy)
                chosen_logprob = torch.log_softmax(scores.float(), dim=1).gather(
                    1, choice.unsqueeze(1)
                ).squeeze(1)
                logprob += torch.where(active & focal_mask, chosen_logprob, 0.0)
            else:
                choice = greedy
            selecting = active & choice.ne(options.shape[1])
            chosen = choice.clamp_max(options.shape[1] - 1)
            sequences[:, step] = torch.where(selecting, chosen, sequences[:, step])
            selected = options[rows, chosen]
            gate_x = linear(selected, weight_ih, bias_ih)
            gate_h = linear(hidden, weight_hh, bias_hh)
            ix_r, ix_z, ix_n = gate_x.chunk(3, dim=-1)
            ih_r, ih_z, ih_n = gate_h.chunk(3, dim=-1)
            reset = torch.sigmoid(ix_r + ih_r)
            update = torch.sigmoid(ix_z + ih_z)
            new = torch.tanh(ix_n + reset * ih_n)
            advanced = new + update * (hidden - new)
            hidden = torch.where(selecting.unsqueeze(1), advanced, hidden)
            available[rows, chosen] &= ~selecting
            lengths += selecting.long()
            active = selecting & lengths.lt(batch.max_count)
        legal = lengths.ge(batch.min_count) & lengths.le(batch.max_count)
        return OrderedActions(sequences, lengths, legal, logprob)


__all__ = ["FrozenDeckHeads", "FrozenHeadIdentity", "SCHEMA"]
