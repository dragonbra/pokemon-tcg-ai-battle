from __future__ import annotations

import copy
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from evaluation.packages.loader import SubmissionPackage

from ..policy.actor_critic import load_source_actor_critic
from ..policy.batching import collate_feature_batches, move_batch
from ..policy.online_runtime import OnlineCausalEncoder
from .resident_pool import OpponentRequest


_DECODER_MODULE_NAMES = (
    "pointer_key",
    "pointer_query",
    "option_bias",
    "decoder_init",
    "decoder",
    "stop",
)
_CANDIDATE_POLICY_NAME = "__candidate__"


def _parameter_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


@dataclass(frozen=True)
class LeagueSampledAction:
    indices: tuple[int, ...]
    stopped: bool
    log_prob: float
    entropy: float
    value: float


class _DecoderHead(nn.Module):
    """One independently trainable copy of the Foundation action decoder."""

    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        for name in _DECODER_MODULE_NAMES:
            setattr(self, name, copy.deepcopy(getattr(actor, name)))
        self.d_model = int(actor.config.d_model)
        self.max_action_steps = int(actor.config.max_action_steps)

    def greedy(self, batch: dict[str, Tensor], state: Tensor, options: Tensor) -> list[list[int]]:
        batch_size, option_count, _ = options.shape
        maximum = batch["max_count"].clamp_max(self.max_action_steps)
        maximum_steps = int(maximum.max().item())
        sequences = torch.full(
            (batch_size, maximum_steps),
            -1,
            dtype=torch.long,
            device=options.device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        hidden = torch.tanh(self.decoder_init(state))
        keys = self.pointer_key(options)
        chosen = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
        finished = maximum == 0
        rows = torch.arange(batch_size, device=options.device)
        for step in range(maximum_steps):
            finished |= lengths >= maximum
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / math.sqrt(self.d_model)
            pointer = pointer + self.option_bias(options).squeeze(-1)
            pointer = pointer.masked_fill(
                ~batch["option_mask"] | chosen,
                torch.finfo(pointer.dtype).min,
            )
            best_score, best_option = pointer.max(1)
            stop_score = self.stop(hidden).squeeze(1)
            stopping = ~finished & (lengths >= batch["min_count"]) & (
                stop_score >= best_score
            )
            active = ~finished & ~stopping
            selected_is_legal = batch["option_mask"][rows, best_option]
            active &= selected_is_legal
            sequences[:, step] = torch.where(active, best_option, -1)
            selected = options[rows, best_option]
            advanced = self.decoder(selected, hidden)
            hidden = torch.where(active.unsqueeze(1), advanced, hidden)
            chosen.scatter_(
                1,
                best_option.unsqueeze(1),
                chosen.gather(1, best_option.unsqueeze(1)) | active.unsqueeze(1),
            )
            lengths += active
            finished |= stopping | ~selected_is_legal
        if (lengths < batch["min_count"]).any():
            raise RuntimeError("shared Foundation decoder violated official minCount")
        return [
            [int(value) for value in sequences[index, : lengths[index]].tolist()]
            for index in range(batch_size)
        ]

    def sample(
        self,
        batch: dict[str, Tensor],
        state: Tensor,
        options: Tensor,
        values: Tensor,
    ) -> list[LeagueSampledAction]:
        batch_size, option_count, _ = options.shape
        keys = self.pointer_key(options)
        hidden = torch.tanh(self.decoder_init(state))
        chosen = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
        minimum = batch["min_count"]
        maximum = batch["max_count"].clamp_max(self.max_action_steps)
        sequences = torch.full(
            (batch_size, self.max_action_steps),
            -1,
            dtype=torch.long,
            device=options.device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=options.device)
        finished = maximum == 0
        log_prob = torch.zeros(batch_size, device=options.device)
        entropy = torch.zeros(batch_size, device=options.device)
        rows = torch.arange(batch_size, device=options.device)
        for step in range(int(maximum.max().item())):
            active = ~finished & (lengths < maximum)
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / math.sqrt(self.d_model)
            pointer = pointer + self.option_bias(options).squeeze(-1)
            pointer = pointer.masked_fill(
                ~batch["option_mask"] | chosen,
                torch.finfo(pointer.dtype).min,
            )
            stop_score = self.stop(hidden).masked_fill(
                ~(lengths >= minimum).unsqueeze(1),
                torch.finfo(pointer.dtype).min,
            )
            logits = torch.cat((pointer, stop_score), dim=1).float()
            probabilities = torch.softmax(logits, dim=1)
            token = torch.multinomial(probabilities, 1).squeeze(1)
            token_log_prob = torch.log_softmax(logits, dim=1).gather(
                1, token.unsqueeze(1)
            ).squeeze(1)
            token_entropy = -(probabilities * torch.log_softmax(logits, dim=1)).sum(1)
            log_prob += torch.where(active, token_log_prob, 0.0)
            entropy += torch.where(active, token_entropy, 0.0)
            choosing_stop = active & (token == option_count)
            choosing_option = active & ~choosing_stop
            safe_option = token.clamp_max(option_count - 1)
            sequences[:, step] = torch.where(choosing_option, safe_option, -1)
            selected = options[rows, safe_option]
            advanced = self.decoder(selected, hidden)
            hidden = torch.where(choosing_option.unsqueeze(1), advanced, hidden)
            chosen.scatter_(
                1,
                safe_option.unsqueeze(1),
                chosen.gather(1, safe_option.unsqueeze(1)) | choosing_option.unsqueeze(1),
            )
            lengths += choosing_option
            stopped |= choosing_stop
            finished |= choosing_stop | (lengths >= maximum)
        if (lengths < minimum).any():
            raise RuntimeError("shared Foundation sampler violated official minCount")
        return [
            LeagueSampledAction(
                indices=tuple(
                    int(value)
                    for value in sequences[index, : lengths[index]].tolist()
                ),
                stopped=bool(stopped[index].item()),
                log_prob=float(log_prob[index].item()),
                entropy=float(entropy[index].item()),
                value=float(values[index].item()),
            )
            for index in range(batch_size)
        ]


def _stack_parameter(modules: list[nn.Module], path: str) -> nn.Parameter:
    values = []
    for module in modules:
        current: Any = module
        parts = path.split(".")
        for part in parts[:-1]:
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        values.append(getattr(current, parts[-1]).detach().clone())
    return nn.Parameter(torch.stack(values))


class _StackedDecoderHeads(nn.Module):
    """Independent expert parameters executed with row-routed batched operations."""

    def __init__(self, actor: nn.Module, policy_names: list[str]) -> None:
        super().__init__()
        heads = [_DecoderHead(actor) for _ in policy_names]
        self.policy_ids = {name: index for index, name in enumerate(policy_names)}
        self.d_model = int(actor.config.d_model)
        self.max_action_steps = int(actor.config.max_action_steps)
        self.pointer_key_weight = _stack_parameter(heads, "pointer_key.weight")
        self.pointer_query_weight = _stack_parameter(heads, "pointer_query.weight")
        self.option_bias_weight = _stack_parameter(heads, "option_bias.weight")
        self.option_bias_bias = _stack_parameter(heads, "option_bias.bias")
        self.decoder_init_weight = _stack_parameter(heads, "decoder_init.weight")
        self.decoder_init_bias = _stack_parameter(heads, "decoder_init.bias")
        self.decoder_weight_ih = _stack_parameter(heads, "decoder.weight_ih")
        self.decoder_weight_hh = _stack_parameter(heads, "decoder.weight_hh")
        self.decoder_bias_ih = _stack_parameter(heads, "decoder.bias_ih")
        self.decoder_bias_hh = _stack_parameter(heads, "decoder.bias_hh")
        self.stop_0_weight = _stack_parameter(heads, "stop.0.weight")
        self.stop_0_bias = _stack_parameter(heads, "stop.0.bias")
        self.stop_2_weight = _stack_parameter(heads, "stop.2.weight")
        self.stop_2_bias = _stack_parameter(heads, "stop.2.bias")

    @staticmethod
    def _linear(
        value: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        if value.ndim == 2:
            output = torch.bmm(weight, value.unsqueeze(2)).squeeze(2)
            return output if bias is None else output + bias
        output = torch.einsum("bni,boi->bno", value, weight)
        return output if bias is None else output + bias.unsqueeze(1)

    def _gru(self, value: Tensor, hidden: Tensor, policy_ids: Tensor) -> Tensor:
        input_gates = self._linear(
            value,
            self.decoder_weight_ih[policy_ids],
            self.decoder_bias_ih[policy_ids],
        )
        hidden_gates = self._linear(
            hidden,
            self.decoder_weight_hh[policy_ids],
            self.decoder_bias_hh[policy_ids],
        )
        input_reset, input_update, input_new = input_gates.chunk(3, dim=1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        new = torch.tanh(input_new + reset * hidden_new)
        return new + update * (hidden - new)

    def decode(
        self,
        batch: dict[str, Tensor],
        state: Tensor,
        options: Tensor,
        policy_ids: Tensor,
        sample_mask: Tensor,
        values: Tensor,
    ) -> list[LeagueSampledAction | list[int]]:
        key_weight = self.pointer_key_weight[policy_ids]
        query_weight = self.pointer_query_weight[policy_ids]
        option_bias_weight = self.option_bias_weight[policy_ids]
        option_bias_bias = self.option_bias_bias[policy_ids]
        hidden = torch.tanh(
            self._linear(
                state,
                self.decoder_init_weight[policy_ids],
                self.decoder_init_bias[policy_ids],
            )
        )
        keys = self._linear(options, key_weight)
        option_bias = self._linear(
            options, option_bias_weight, option_bias_bias
        ).squeeze(2)
        batch_size, option_count, _ = options.shape
        minimum = batch["min_count"]
        maximum = batch["max_count"].clamp_max(self.max_action_steps)
        sequences = torch.full(
            (batch_size, self.max_action_steps),
            -1,
            dtype=torch.long,
            device=options.device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=options.device)
        chosen = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
        finished = maximum == 0
        log_prob = torch.zeros(batch_size, device=options.device)
        entropy = torch.zeros(batch_size, device=options.device)
        rows = torch.arange(batch_size, device=options.device)
        for step in range(int(maximum.max().item())):
            finished |= lengths >= maximum
            query = self._linear(hidden, query_weight)
            pointer = (query.unsqueeze(1) * keys).sum(2) / math.sqrt(self.d_model)
            pointer = (pointer + option_bias).masked_fill(
                ~batch["option_mask"] | chosen,
                torch.finfo(pointer.dtype).min,
            )
            stop_hidden = torch.nn.functional.gelu(
                self._linear(
                    hidden,
                    self.stop_0_weight[policy_ids],
                    self.stop_0_bias[policy_ids],
                )
            )
            stop_score = self._linear(
                stop_hidden,
                self.stop_2_weight[policy_ids],
                self.stop_2_bias[policy_ids],
            ).squeeze(1)

            best_score, greedy_option = pointer.max(1)
            greedy_stop = (lengths >= minimum) & (stop_score >= best_score)
            logits = torch.cat((pointer, stop_score.unsqueeze(1)), dim=1).float()
            logits[:, -1] = logits[:, -1].masked_fill(
                lengths < minimum, torch.finfo(logits.dtype).min
            )
            probabilities = torch.softmax(logits, dim=1)
            sampled_token = torch.multinomial(probabilities, 1).squeeze(1)
            token = torch.where(sample_mask, sampled_token, greedy_option)
            choosing_stop = ~finished & torch.where(
                sample_mask,
                token == option_count,
                greedy_stop,
            )
            choosing_option = ~finished & ~choosing_stop
            safe_option = token.clamp_max(option_count - 1)

            token_log_prob = torch.log_softmax(logits, dim=1).gather(
                1, sampled_token.unsqueeze(1)
            ).squeeze(1)
            token_entropy = -(probabilities * torch.log_softmax(logits, dim=1)).sum(1)
            sampled_active = sample_mask & ~finished
            log_prob += torch.where(sampled_active, token_log_prob, 0.0)
            entropy += torch.where(sampled_active, token_entropy, 0.0)
            stopped |= choosing_stop & sample_mask
            sequences[:, step] = torch.where(choosing_option, safe_option, -1)
            selected = options[rows, safe_option]
            advanced = self._gru(selected, hidden, policy_ids)
            hidden = torch.where(choosing_option.unsqueeze(1), advanced, hidden)
            chosen.scatter_(
                1,
                safe_option.unsqueeze(1),
                chosen.gather(1, safe_option.unsqueeze(1))
                | choosing_option.unsqueeze(1),
            )
            lengths += choosing_option
            finished |= choosing_stop
        if (lengths < minimum).any():
            raise RuntimeError("stacked Foundation decoder violated official minCount")
        result: list[LeagueSampledAction | list[int]] = []
        for index in range(batch_size):
            indices = tuple(
                int(value)
                for value in sequences[index, : lengths[index]].tolist()
            )
            if bool(sample_mask[index].item()):
                result.append(
                    LeagueSampledAction(
                        indices=indices,
                        stopped=bool(stopped[index].item()),
                        log_prob=float(log_prob[index].item()),
                        entropy=float(entropy[index].item()),
                        value=float(values[index].item()),
                    )
                )
            else:
                result.append(list(indices))
        return result


class SharedFoundationOpponentPool:
    """One resident Foundation encoder with one routed Decoder head per deck."""

    def __init__(
        self,
        packages: list[SubmissionPackage],
        *,
        device: torch.device,
        include_candidate: bool = False,
    ) -> None:
        if not packages:
            raise ValueError("shared Foundation pool requires at least one package")
        self.device = device
        started = time.perf_counter()
        model, metadata = load_source_actor_critic()
        actor = model.actor
        self.config = actor.config
        policy_names = [package.name for package in packages]
        if include_candidate:
            policy_names.append(_CANDIDATE_POLICY_NAME)
        self.heads = _StackedDecoderHeads(actor, policy_names)
        self.candidate_value_head = (
            copy.deepcopy(model.value_head) if include_candidate else None
        )
        # The resident Foundation owns encoder parameters only; decoder parameters live in heads.
        for name in _DECODER_MODULE_NAMES:
            setattr(actor, name, nn.Identity())
        self.encoder = actor
        self.encoder.to(device).eval()
        self.heads.to(device).eval()
        if self.candidate_value_head is not None:
            self.candidate_value_head.to(device).eval()
        self.packages = {package.name: package for package in packages}
        self.source_version = str(metadata["version"])
        self.include_candidate = include_candidate
        self._session_state: dict[tuple[str, str], OnlineCausalEncoder] = {}
        self.load_seconds = time.perf_counter() - started
        self.request_count = 0
        self.batch_count = 0
        self.max_batch_size = 0
        self.batch_size_histogram: Counter[int] = Counter()
        self.feature_seconds = 0.0
        self.collate_transfer_seconds = 0.0
        self.encoder_seconds = 0.0
        self.decoder_seconds = 0.0
        self.last_select_seconds = 0.0

    def supports(self, package_name: str) -> bool:
        return package_name in self.packages

    def close_session(self, session_id: str) -> None:
        stale = [key for key in self._session_state if key[0] == session_id]
        for key in stale:
            del self._session_state[key]

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def select(self, requests: list[OpponentRequest]) -> list[list[int]]:
        if not requests:
            return []
        feature_started = time.perf_counter()
        encoded = []
        for request in requests:
            package = self.packages.get(request.package_name)
            if package is None:
                raise ValueError(
                    "opponent is not in shared Foundation pool: "
                    f"{request.package_name}"
                )
            actor = int((request.observation.get("current") or {}).get("yourIndex", -1))
            state_key = (request.session_id, request.package_name)
            encoder = self._session_state.get(state_key)
            if encoder is None or encoder.actor != actor:
                encoder = OnlineCausalEncoder(actor, package.deck, self.config)
                self._session_state[state_key] = encoder
            item = encoder.encode(request.observation)
            item["source_id"] = torch.tensor([0], dtype=torch.long)
            encoded.append(item)
        self.feature_seconds += time.perf_counter() - feature_started

        return self._select_encoded(
            [request.package_name for request in requests],
            encoded,
        )

    def select_mixed(
        self,
        candidate_features: list[dict[str, Tensor]],
        opponent_requests: list[OpponentRequest],
        *,
        sample_candidate: bool = False,
    ) -> tuple[list[LeagueSampledAction | list[int]], list[list[int]]]:
        if not self.include_candidate:
            raise RuntimeError("shared Foundation pool was not configured for candidate routing")
        feature_started = time.perf_counter()
        opponent_features = []
        for request in opponent_requests:
            package = self.packages.get(request.package_name)
            if package is None:
                raise ValueError(
                    "opponent is not in shared Foundation pool: "
                    f"{request.package_name}"
                )
            actor = int((request.observation.get("current") or {}).get("yourIndex", -1))
            state_key = (request.session_id, request.package_name)
            encoder = self._session_state.get(state_key)
            if encoder is None or encoder.actor != actor:
                encoder = OnlineCausalEncoder(actor, package.deck, self.config)
                self._session_state[state_key] = encoder
            item = encoder.encode(request.observation)
            item["source_id"] = torch.tensor([0], dtype=torch.long)
            opponent_features.append(item)
        self.feature_seconds += time.perf_counter() - feature_started
        policy_names = [_CANDIDATE_POLICY_NAME] * len(candidate_features) + [
            request.package_name for request in opponent_requests
        ]
        actions = self._select_encoded(
            policy_names,
            candidate_features + opponent_features,
            sample_candidate=sample_candidate,
        )
        split = len(candidate_features)
        return actions[:split], actions[split:]

    def _select_encoded(
        self,
        policy_names: list[str],
        encoded: list[dict[str, Tensor]],
        *,
        sample_candidate: bool = False,
    ) -> list[LeagueSampledAction | list[int]]:
        if not encoded:
            self.last_select_seconds = 0.0
            return []
        select_started = time.perf_counter()
        transfer_started = time.perf_counter()
        batch = move_batch(collate_feature_batches(encoded), self.device)
        self._synchronize()
        self.collate_transfer_seconds += time.perf_counter() - transfer_started

        encoder_started = time.perf_counter()
        with torch.inference_mode():
            state, options = self.encoder.encode(batch)
        self._synchronize()
        self.encoder_seconds += time.perf_counter() - encoder_started

        decoder_started = time.perf_counter()
        with torch.inference_mode():
            policy_ids = torch.tensor(
                [self.heads.policy_ids[name] for name in policy_names],
                dtype=torch.long,
                device=self.device,
            )
            sample_mask = torch.tensor(
                [
                    sample_candidate and name == _CANDIDATE_POLICY_NAME
                    for name in policy_names
                ],
                dtype=torch.bool,
                device=self.device,
            )
            values = (
                self.candidate_value_head(state).squeeze(1)
                if self.candidate_value_head is not None
                else torch.zeros(state.size(0), device=self.device)
            )
            result = self.heads.decode(
                batch,
                state,
                options,
                policy_ids,
                sample_mask,
                values,
            )
        self._synchronize()
        self.decoder_seconds += time.perf_counter() - decoder_started

        self.request_count += len(encoded)
        self.batch_count += 1
        self.max_batch_size = max(self.max_batch_size, len(encoded))
        self.batch_size_histogram[len(encoded)] += 1
        self.last_select_seconds = time.perf_counter() - select_started
        return result

    def audit(self) -> dict[str, Any]:
        decoder_bytes = _parameter_bytes(self.heads)
        return {
            "kind": "shared_foundation_routed_decoders",
            "device": str(self.device),
            "source_version": self.source_version,
            "resident_count": len(self.packages),
            "head_count": len(self.heads.policy_ids),
            "candidate_head_included": self.include_candidate,
            "package_names": sorted(self.packages),
            "load_seconds": self.load_seconds,
            "encoder_parameter_bytes": _parameter_bytes(self.encoder),
            "decoder_parameter_bytes_total": decoder_bytes,
            "decoder_parameter_bytes_each": decoder_bytes // len(self.heads.policy_ids),
            "request_count": self.request_count,
            "batch_count": self.batch_count,
            "max_batch_size": self.max_batch_size,
            "mean_batch_size": self.request_count / self.batch_count if self.batch_count else 0.0,
            "batch_size_histogram": {
                str(size): count for size, count in sorted(self.batch_size_histogram.items())
            },
            "feature_seconds": self.feature_seconds,
            "collate_transfer_seconds": self.collate_transfer_seconds,
            "encoder_seconds": self.encoder_seconds,
            "decoder_seconds": self.decoder_seconds,
        }


__all__ = ["LeagueSampledAction", "SharedFoundationOpponentPool"]
