from __future__ import annotations

import copy
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from evaluation.packages.loader import SubmissionPackage
from evaluation.runner.worker import _load_agent_module

from ..policy.batching import collate_feature_batches, move_batch
from .catalog import OpponentInferenceKind, classify_opponent


def _id_only_collate(module: Any) -> Any:
    for loader_name in ("_load_train_module", "_load_policy_module"):
        loader = getattr(module, loader_name, None)
        if callable(loader):
            provider = loader()
            collate = getattr(provider, "collate_examples", None)
            if callable(collate):
                return collate
    raise RuntimeError("ID-only package exposes no supported collate provider")


def _decoder_hidden(model: Any, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.tanh(model.decoder_init(state))
    layers = int(getattr(model.config, "decoder_layers", 1))
    if layers > 1:
        hidden = hidden.view(state.size(0), layers, model.config.d_model)
        return hidden, hidden[:, -1]
    return hidden, hidden


def _selection_bounds(
    requests: list[OpponentRequest],
    device: torch.device,
    max_action_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    minimum = torch.tensor(
        [int((request.observation.get("select") or {}).get("minCount", 0)) for request in requests],
        dtype=torch.long,
        device=device,
    )
    maximum = torch.tensor(
        [int((request.observation.get("select") or {}).get("maxCount", 0)) for request in requests],
        dtype=torch.long,
        device=device,
    ).clamp_max(max_action_steps)
    if (minimum < 0).any() or (maximum < minimum).any():
        raise ValueError("official runtime supplied invalid selection bounds")
    return minimum, maximum


def _add_deck_conditioning(row: dict[str, Any], deck: list[int]) -> None:
    if not deck:
        return
    if len(deck) != 60:
        raise ValueError("resident opponent deck conditioning requires exactly 60 cards")
    counts = Counter(deck)
    row.update(
        deck_ids=deck,
        deck_counts=[counts[card] for card in deck],
        sample_weight=1.0,
        deck_hash_code=0,
    )


@dataclass(frozen=True)
class OpponentRequest:
    session_id: str
    package_name: str
    observation: dict[str, Any]


class _ResidentPolicy:
    def __init__(
        self,
        package: SubmissionPackage,
        kind: OpponentInferenceKind,
        device: torch.device,
    ) -> None:
        self.package = package
        self.kind = kind
        self.device = device
        self.module = _load_agent_module(package, "0018_resident_gpu", ())
        self.policy = getattr(self.module, "_POLICY", None)
        self.runtime = None
        self._session_state: dict[str, Any] = {}
        self._prepare()

    def _prepare(self) -> None:
        if self.kind == OpponentInferenceKind.SOURCE_CAUSAL:
            if self.policy is None:
                raise RuntimeError("source-causal package does not expose _POLICY")
            model = getattr(self.policy, "actor", None) or getattr(self.policy, "model", None)
            if model is None:
                raise RuntimeError("source-causal package does not expose a model")
            model.to(self.device).eval()
            return
        if self.kind == OpponentInferenceKind.FULL_ACTION:
            if self.policy is None or not hasattr(self.policy, "model"):
                raise RuntimeError("full-action package does not expose _POLICY.model")
            self.policy.model.to(self.device).eval()
            self.policy.device = self.device
            return
        if self.kind == OpponentInferenceKind.ID_ONLY:
            if hasattr(self.module, "_ensure_model"):
                model, _codec, _torch = self.module._ensure_model()
            elif hasattr(self.module, "_runtime"):
                self.runtime = self.module._runtime()
                model = self.runtime.model
                if hasattr(self.runtime, "device"):
                    self.runtime.device = self.device
            else:
                raise RuntimeError("ID-only package exposes no supported runtime")
            model.to(self.device).eval()
            return
        raise RuntimeError(f"unsupported resident inference kind: {self.kind}")

    def close_session(self, session_id: str) -> None:
        self._session_state.pop(session_id, None)

    def select_many(self, requests: list[OpponentRequest]) -> list[list[int]]:
        if self.kind == OpponentInferenceKind.SOURCE_CAUSAL:
            return self._select_source(requests)
        if self.kind == OpponentInferenceKind.FULL_ACTION:
            return self._select_full_action(requests)
        if self.kind == OpponentInferenceKind.ID_ONLY:
            return self._select_id_only(requests)
        raise RuntimeError(f"unsupported resident inference kind: {self.kind}")

    def _select_source(self, requests: list[OpponentRequest]) -> list[list[int]]:
        encoded = []
        source_id = int(getattr(self.policy, "source_id", 0))
        model = getattr(self.policy, "actor", None) or getattr(self.policy, "model", None)
        config = getattr(model, "config")
        encoder_type = type(getattr(self.policy, "encoder", None))
        if encoder_type is type(None):
            encoder_type = self.policy.select.__func__.__globals__["OnlineCausalEncoder"]
        for request in requests:
            actor = int((request.observation.get("current") or {}).get("yourIndex", -1))
            encoder = self._session_state.get(request.session_id)
            if encoder is None or encoder.actor != actor:
                encoder = encoder_type(actor, self.policy.deck, config)
                self._session_state[request.session_id] = encoder
            item = encoder.encode(request.observation)
            item["source_id"] = torch.tensor([source_id], dtype=torch.long)
            encoded.append(item)
        batch = move_batch(collate_feature_batches(encoded), self.device)
        with torch.inference_mode():
            decoded = model.deterministic_action_tensors(batch)
        return [
            [int(value) for value in decoded.sequences[index, : decoded.lengths[index]].tolist()]
            for index in range(len(requests))
        ]

    def _select_full_action(self, requests: list[OpponentRequest]) -> list[list[int]]:
        # The existing policy already moves encoded tensors to its declared device.
        actions = []
        for request in requests:
            history = self._session_state.setdefault(request.session_id, [])
            self.policy.history = history
            with torch.inference_mode():
                actions.append(list(self.policy.select(copy.deepcopy(request.observation))))
            self._session_state[request.session_id] = list(self.policy.history)
        return actions

    def _select_id_only(self, requests: list[OpponentRequest]) -> list[list[int]]:
        if self.runtime is not None and hasattr(self.runtime, "ledger"):
            actions = []
            initial = copy.deepcopy(self.runtime.ledger)
            for request in requests:
                self.runtime.ledger = self._session_state.setdefault(
                    request.session_id, copy.deepcopy(initial)
                )
                actions.append(list(self.runtime.decode(request.observation)))
                self._session_state[request.session_id] = self.runtime.ledger
            return actions
        return self._decode_id_only_batch(requests)

    def _decode_id_only_batch(self, requests: list[OpponentRequest]) -> list[list[int]]:
        if self.runtime is None:
            model, codec, _torch = self.module._ensure_model()
            deck = [int(card) for card in getattr(self.module, "my_deck", [])]
            collate = _id_only_collate(self.module)
        else:
            model = self.runtime.model
            codec = self.runtime.codec
            deck = [int(card) for card in self.runtime.deck_ids]
            collate = self.runtime.decode.__func__.__globals__["collate_examples"]
        rows = []
        for request in requests:
            select = request.observation.get("select") or {}
            options = select.get("option") or []
            minimum = max(0, int(select.get("minCount", 0) or 0))
            if not options or len(options) > model.config.max_options:
                raise ValueError("resident ID-only request exceeds model option capacity")
            row = codec.encode(request.observation, list(range(minimum)))
            if row is None:
                raise ValueError("resident ID-only codec rejected an engine observation")
            _add_deck_conditioning(row, deck)
            rows.append(row)
        batch = collate(rows)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            state, option_values = model.encode(batch)
            keys = model.pointer_key(option_values)
            hidden, top = _decoder_hidden(model, state)
            available = batch["option_mask"].clone()
            chosen = torch.zeros_like(available)
            batch_size, option_count = available.shape
            minimum, maximum = _selection_bounds(
                requests, self.device, model.config.max_action_steps
            )
            lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            sequences = torch.full(
                (batch_size, int(maximum.max().item())),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            finished = maximum == 0
            rows_index = torch.arange(batch_size, device=self.device)
            for step in range(sequences.size(1)):
                finished |= lengths >= maximum
                pointer = (
                    model.pointer_query(top).unsqueeze(1) * keys
                ).sum(-1) / math.sqrt(model.config.d_model)
                pointer = pointer + model.option_bias(option_values).squeeze(-1)
                pointer = pointer.masked_fill(
                    ~available | chosen, torch.finfo(pointer.dtype).min
                )
                best_score, best_option = pointer.max(1)
                stop_score = model.stop(top).squeeze(1)
                stopping = ~finished & (lengths >= minimum) & (stop_score >= best_score)
                active = ~finished & ~stopping
                sequences[:, step] = torch.where(active, best_option, -1)
                selected = option_values[rows_index, best_option]
                if hasattr(model, "decode_step"):
                    advanced = model.decode_step(selected, hidden)
                    hidden = torch.where(active[:, None, None], advanced, hidden)
                    top = hidden[:, -1]
                else:
                    advanced = model.decoder(selected, hidden)
                    hidden = torch.where(active[:, None], advanced, hidden)
                    top = hidden
                chosen.scatter_(
                    1,
                    best_option.unsqueeze(1),
                    chosen.gather(1, best_option.unsqueeze(1)) | active.unsqueeze(1),
                )
                lengths += active
                finished |= stopping
        return [
            [int(value) for value in sequences[index, : lengths[index]].tolist()]
            for index in range(len(requests))
        ]

    def resident_parameter_bytes(self) -> int:
        if self.kind == OpponentInferenceKind.ID_ONLY:
            if self.runtime is not None:
                model = self.runtime.model
            else:
                model, _codec, _torch = self.module._ensure_model()
        else:
            model = getattr(self.policy, "actor", None) or getattr(self.policy, "model", None)
        return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


class ResidentOpponentPool:
    """Load compatible opponent policies once and keep their weights on one GPU."""

    def __init__(
        self,
        packages: list[SubmissionPackage],
        *,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("resident opponent inference requires a CUDA device")
        self.device = device
        self._policies: dict[str, _ResidentPolicy] = {}
        self.failures: dict[str, str] = {}
        self.request_count = 0
        self.batch_count = 0
        self.max_batch_size = 0
        self.batch_size_histogram: Counter[int] = Counter()
        self.inference_seconds = 0.0
        for package in packages:
            kind = classify_opponent(package)
            if kind == OpponentInferenceKind.CPU:
                continue
            try:
                self._policies[package.name] = _ResidentPolicy(package, kind, device)
            except BaseException as error:
                self.failures[package.name] = f"{type(error).__name__}: {error}"

    def supports(self, package_name: str) -> bool:
        return package_name in self._policies

    def close_session(self, session_id: str) -> None:
        for policy in self._policies.values():
            policy.close_session(session_id)

    def select(self, requests: list[OpponentRequest]) -> list[list[int]]:
        grouped: dict[str, list[tuple[int, OpponentRequest]]] = defaultdict(list)
        for index, request in enumerate(requests):
            grouped[request.package_name].append((index, request))
        result: list[list[int] | None] = [None] * len(requests)
        started = time.perf_counter()
        for package_name, indexed in grouped.items():
            policy = self._policies.get(package_name)
            if policy is None:
                raise ValueError(f"opponent is not resident: {package_name}")
            actions = policy.select_many([request for _, request in indexed])
            for (index, _request), action in zip(indexed, actions, strict=True):
                result[index] = action
            self.batch_count += 1
            self.max_batch_size = max(self.max_batch_size, len(indexed))
            self.batch_size_histogram[len(indexed)] += 1
        self.inference_seconds += time.perf_counter() - started
        self.request_count += len(requests)
        if any(action is None for action in result):
            raise RuntimeError("resident opponent response ordering is incomplete")
        return [action for action in result if action is not None]

    def audit(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "resident_count": len(self._policies),
            "resident": {
                name: {
                    "kind": policy.kind.value,
                    "parameter_bytes": policy.resident_parameter_bytes(),
                }
                for name, policy in sorted(self._policies.items())
            },
            "load_failures": dict(sorted(self.failures.items())),
            "request_count": self.request_count,
            "batch_count": self.batch_count,
            "max_batch_size": self.max_batch_size,
            "mean_batch_size": (
                self.request_count / self.batch_count if self.batch_count else 0.0
            ),
            "batch_size_histogram": {
                str(size): count for size, count in sorted(self.batch_size_histogram.items())
            },
            "inference_seconds": self.inference_seconds,
        }


__all__ = ["OpponentRequest", "ResidentOpponentPool"]
