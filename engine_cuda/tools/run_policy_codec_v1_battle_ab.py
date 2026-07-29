from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CUDA_ENGINE_ROOT.parent
for module_root in (
    REPO_ROOT / "tools",
    CUDA_ENGINE_ROOT / "python",
    CUDA_ENGINE_ROOT / "tools",
):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from probe_actual_policy_residency import checkpoint_state, load_module  # noqa: E402
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    EntityPointerPolicyV1DeviceAdapter,
    IDOnlyPointerPolicyDeviceAdapter,
)
from pure_policy_codec_v1 import PolicyCodecV1  # noqa: E402
from pure_policy_model_v1 import (  # noqa: E402
    collate_decision_records,
    load_policy_checkpoint,
)
from seeded_cpp_shim import SeededCppBattle, SeededCppLib  # noqa: E402


DEFAULT_LIB = REPO_ROOT / "tmp" / "seeded_cpp_shim_policy_codec_v1" / "libcg_seeded.so"
DEFAULT_OPPONENT_POOL = CUDA_ENGINE_ROOT / "configs" / "strongest_bc_opponent_pool_v1.json"
DEFAULT_TARGET = (
    REPO_ROOT / "arena_agents" / "agent_pure_lucario_v1_bc512_e4_probe" / "policy.pt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_deck(path: Path) -> list[int]:
    deck = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path} contains {len(deck)} cards, expected 60")
    return deck


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def terminal_winner(game_result: int) -> int | None:
    if game_result == 1:
        return 0
    if game_result == 2:
        return 1
    return None


def normalize_action(obs: Mapping[str, Any], action: Any) -> list[int]:
    select = obs.get("select") or {}
    options = list(select.get("option") or [])
    minimum = max(0, int(select.get("minCount") or 0))
    maximum = max(minimum, int(select.get("maxCount") or minimum))
    output: list[int] = []
    if isinstance(action, list):
        for value in action:
            if isinstance(value, int) and 0 <= value < len(options) and value not in output:
                output.append(value)
    output = output[:maximum]
    if len(output) < minimum:
        output.extend(index for index in range(len(options)) if index not in output)
        output = output[:minimum]
    return output


def canonical_actions(actions: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    values = actions.detach().cpu().tolist()
    sizes = lengths.detach().cpu().tolist()
    return [list(map(int, row[: int(size)])) for row, size in zip(values, sizes)]


def pad_axis_one(value: torch.Tensor, capacity: int, fill: int | float | bool = 0) -> torch.Tensor:
    if value.ndim < 2 or value.shape[1] > capacity:
        raise ValueError(f"cannot pad tensor {tuple(value.shape)} to axis-one capacity {capacity}")
    if value.shape[1] == capacity:
        return value
    shape = list(value.shape)
    shape[1] = capacity
    output = torch.full(shape, fill, dtype=value.dtype, device=value.device)
    output[:, : value.shape[1]] = value
    return output


def semantic_actions_equal(
    left: list[int],
    right: list[int],
    equivalence: list[int],
) -> bool:
    if len(left) != len(right):
        return False
    for left_index, right_index in zip(left, right):
        if left_index == right_index:
            continue
        if not (0 <= left_index < len(equivalence) and 0 <= right_index < len(equivalence)):
            return False
        group = int(equivalence[left_index])
        if group < 0 or group != int(equivalence[right_index]):
            return False
    return True


def require_legal_model_action(
    observation: Mapping[str, Any],
    action: list[int],
    *,
    policy_name: str,
) -> list[int]:
    normalized = normalize_action(observation, action)
    if normalized != action:
        raise RuntimeError(
            f"illegal raw action from {policy_name}: raw={action} normalized={normalized}"
        )
    return action


class FrozenPolicyCodecV1Backends:
    def __init__(self, checkpoint: Path, device: torch.device, *, cpu_threads: int) -> None:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.checkpoint = checkpoint
        self.device = device
        self.codec = PolicyCodecV1()
        self.cpu_model, self.payload = load_policy_checkpoint(checkpoint, "cpu")
        self.gpu_model, _ = load_policy_checkpoint(checkpoint, device)
        self.cpu_model.requires_grad_(False).eval()
        self.gpu_model.requires_grad_(False).eval()
        self.gpu_adapter = EntityPointerPolicyV1DeviceAdapter(self.gpu_model, max_select=80)
        config = self.gpu_model.config
        self.entity_capacity = int(getattr(config, "max_entities", 192))
        self.option_capacity = int(getattr(config, "max_options", 128))

    def _collate(self, observations: list[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
        records: list[dict[str, Any]] = []
        equivalence: list[list[int]] = []
        for observation in observations:
            encoded = self.codec.encode(observation)
            row = encoded.to_record()
            row.update(
                teacher_action=[],
                value_target=0.0,
                next_prize_target=0.0,
                finish_reason=0,
                opponent_class=0,
            )
            records.append(row)
            equivalence.append([int(value) for value in row["option_equiv"]])
        batch = collate_decision_records(records)
        for name in ("entity_cat", "entity_num", "entity_parent", "entity_mask"):
            batch[name] = pad_axis_one(batch[name], self.entity_capacity)
        for name in ("option_cat", "option_num", "option_equiv", "option_mask"):
            fill = -1 if name == "option_equiv" else 0
            batch[name] = pad_axis_one(batch[name], self.option_capacity, fill)
        return batch, equivalence

    @torch.inference_mode()
    def act_cpu(self, observations: list[dict[str, Any]]) -> tuple[list[list[int]], list[list[int]]]:
        if not observations:
            return [], []
        batch, equivalence = self._collate(observations)
        actions, _logprob, _entropy, _value = self.cpu_model.sample_decode_batch(
            batch,
            mode="greedy",
            compute_entropy=False,
        )
        return [
            require_legal_model_action(obs, action, policy_name="target_cpu")
            for obs, action in zip(observations, actions)
        ], equivalence

    @torch.inference_mode()
    def act_gpu(self, observations: list[dict[str, Any]]) -> list[list[int]]:
        if not observations:
            return []
        batch, _equivalence = self._collate(observations)
        gpu_batch = {name: value.to(self.device) for name, value in batch.items()}
        actions, lengths = self.gpu_adapter.act_device(gpu_batch)
        rows = canonical_actions(actions, lengths)
        return [
            require_legal_model_action(obs, action, policy_name="target_gpu")
            for obs, action in zip(observations, rows)
        ]


@dataclass(frozen=True)
class OpponentSpec:
    name: str
    archetype: str
    kind: str
    directory: Path
    adapter: str
    codec: str
    checkpoints: tuple[Path, ...]
    right_weight: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpponentSpec":
        kind = str(raw["kind"])
        directory = resolve_repo_path(str(raw["directory"]))
        raw_checkpoints = raw.get("checkpoints")
        if raw_checkpoints is None:
            raw_checkpoints = [raw.get("checkpoint", "policy.pt")]
        checkpoints = tuple(
            (Path(value).resolve() if Path(value).is_absolute() else (directory / value).resolve())
            for value in raw_checkpoints
        )
        spec = cls(
            name=str(raw["name"]),
            archetype=str(raw["archetype"]),
            kind=kind,
            directory=directory,
            adapter=str(raw.get("adapter", "")),
            codec=str(raw.get("codec", "")),
            checkpoints=checkpoints,
            right_weight=(None if raw.get("right_weight") is None else float(raw["right_weight"])),
        )
        if kind not in {"policy_codec_v1", "idonly_single", "idonly_probability_ensemble"}:
            raise ValueError(f"unsupported opponent kind: {kind}")
        if not (directory / "deck.csv").is_file() or not all(path.is_file() for path in checkpoints):
            raise FileNotFoundError(f"incomplete opponent package: {directory}")
        if kind == "idonly_probability_ensemble" and (
            len(checkpoints) != 2 or spec.right_weight is None
        ):
            raise ValueError("probability ensemble requires two checkpoints and right_weight")
        return spec


class FrozenIDOnlyOpponent:
    """One frozen BC model with independent per-battle belief state."""

    def __init__(self, policy: OpponentSpec, module_index: int) -> None:
        self.policy = policy
        self.name = policy.name
        self.checkpoint = policy.checkpoints[0]
        source = self.checkpoint.parent / "idonly_policy.py"
        self.module = load_module(source, f"_battle_ab_opponent_{module_index}")
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        config = self.module.ModelConfig(**payload["model_config"])
        self.model = self.module.IDOnlyPointerPolicy(config)
        self.model.load_state_dict(checkpoint_state(payload), strict=True)
        self.model.requires_grad_(False).eval()
        self.adapter = IDOnlyPointerPolicyDeviceAdapter(
            self.model,
            max_select=int(config.max_action_steps),
        )
        self.codec = self.module.IDOnlyCodec(config)
        self.deck = read_deck(self.checkpoint.parent / "deck.csv")
        counts = Counter(self.deck)
        self.deck_ids = list(self.deck)
        self.deck_counts = [counts[card] for card in self.deck]
        self.ledgers: dict[str, Any] = {}

    def _row(self, battle_key: str, observation: dict[str, Any]) -> dict[str, Any] | None:
        select = observation.get("select") or {}
        minimum = max(0, int(select.get("minCount") or 0))
        codec_observation = observation
        if self.policy.adapter == "marnie_compact_v5":
            codec_observation, _audit = self.module.project_public_information(observation)
        row = self.codec.encode(codec_observation, list(range(minimum)))
        if row is None:
            return None
        row.update(
            deck_ids=self.deck_ids,
            deck_counts=self.deck_counts,
            sample_weight=1.0,
            deck_hash_code=0,
            prize_source_code=0,
        )
        if self.policy.adapter in {"marnie_prize_pointer_v4", "marnie_compact_v5"}:
            ledger = self.ledgers.setdefault(battle_key, self.module.PrizeLedger(self.deck_ids))
            prize = ledger.observe(codec_observation)
            if self.policy.adapter == "marnie_compact_v5":
                compact = self.module.compact_probability_observation(prize, draw_horizon=3)
                row.update(
                    resource_card=list(compact.card_ids),
                    resource_registered_count=list(compact.registered_counts),
                    resource_visible_count=list(compact.visible_counts),
                    resource_probability=[list(values) for values in compact.features],
                    resource_source_code=compact.source_code,
                )
            else:
                row.update(
                    prize_card=list(prize.card_ids),
                    prize_num=[list(values) for values in prize.features],
                    prize_source_code=prize.source_code,
                )
        return row

    @torch.inference_mode()
    def act_many(self, requests: list[tuple[str, dict[str, Any]]]) -> list[list[int]]:
        output: list[list[int] | None] = [None] * len(requests)
        valid_rows: list[dict[str, Any]] = []
        valid_indices: list[int] = []
        for index, (battle_key, observation) in enumerate(requests):
            row = self._row(battle_key, observation)
            if row is None:
                output[index] = normalize_action(observation, [])
            else:
                valid_rows.append(row)
                valid_indices.append(index)
        if valid_rows:
            batch = self.module.collate_examples(valid_rows)
            batch["max_count"] = torch.tensor(
                [int((requests[index][1].get("select") or {}).get("maxCount") or 0) for index in valid_indices],
                dtype=torch.long,
            )
            actions, lengths = self.adapter.act_device(batch)
            for index, action in zip(valid_indices, canonical_actions(actions, lengths)):
                output[index] = require_legal_model_action(
                    requests[index][1],
                    action,
                    policy_name=self.name,
                )
        return [list(action or []) for action in output]


class FrozenPolicyCodecV1Opponent:
    def __init__(self, spec: OpponentSpec, target: FrozenPolicyCodecV1Backends) -> None:
        if spec.checkpoints != (target.checkpoint,):
            raise ValueError("PolicyCodecV1 mirror must use the frozen target checkpoint")
        self.name = spec.name
        self.deck = read_deck(spec.directory / "deck.csv")
        self.target = target

    def act_many(self, requests: list[tuple[str, dict[str, Any]]]) -> list[list[int]]:
        actions, _equivalence = self.target.act_cpu([observation for _key, observation in requests])
        return actions


class FrozenProbabilityEnsembleOpponent:
    def __init__(self, spec: OpponentSpec, module_index: int) -> None:
        self.name = spec.name
        self.spec = spec
        self.deck = read_deck(spec.directory / "deck.csv")
        self.module = load_module(
            spec.directory / "idonly_policy.py",
            f"_battle_ab_ensemble_{module_index}",
        )
        payloads = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in spec.checkpoints
        ]
        if payloads[0]["model_config"] != payloads[1]["model_config"]:
            raise ValueError("ensemble checkpoints have different model configs")
        self.config = self.module.ModelConfig(**payloads[0]["model_config"])
        self.models = []
        for payload in payloads:
            model = self.module.IDOnlyPointerPolicy(self.config)
            model.load_state_dict(checkpoint_state(payload), strict=True)
            model.requires_grad_(False).eval()
            self.models.append(model)
        self.right_weight = float(spec.right_weight)
        self.codec = self.module.IDOnlyCodec(self.config)
        counts = Counter(self.deck)
        self.deck_ids = list(self.deck)
        self.deck_counts = [counts[card] for card in self.deck]

    def _logits(
        self,
        model: Any,
        option_values: torch.Tensor,
        keys: torch.Tensor,
        hidden: torch.Tensor,
        available: torch.Tensor,
        chosen: torch.Tensor,
        min_count: torch.Tensor,
        active: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        pointer = (
            model.pointer_query(hidden).unsqueeze(1) * keys
        ).sum(-1) / math.sqrt(model.config.d_model)
        pointer = pointer + model.option_bias(option_values).squeeze(-1)
        pointer = pointer.masked_fill(
            ~available | chosen | ~active.unsqueeze(1),
            torch.finfo(pointer.dtype).min,
        )
        stop = model.stop(hidden).squeeze(-1)
        stop = stop.masked_fill(
            ~active | min_count.gt(step),
            torch.finfo(stop.dtype).min,
        )
        return torch.cat([pointer, stop.unsqueeze(1)], dim=1)

    @torch.inference_mode()
    def act_many(self, requests: list[tuple[str, dict[str, Any]]]) -> list[list[int]]:
        if not requests:
            return []
        rows: list[dict[str, Any]] = []
        for _battle_key, observation in requests:
            select = observation.get("select") or {}
            minimum = max(0, int(select.get("minCount") or 0))
            row = self.codec.encode(observation, list(range(minimum)))
            if row is None:
                raise RuntimeError(f"ensemble codec rejected a live observation for {self.name}")
            row.update(
                deck_ids=self.deck_ids,
                deck_counts=self.deck_counts,
                sample_weight=1.0,
                deck_hash_code=0,
            )
            rows.append(row)
        batch = self.module.collate_examples(rows)
        min_count = batch["min_count"].long()
        max_count = torch.tensor(
            [int((observation.get("select") or {}).get("maxCount") or 0) for _key, observation in requests],
            dtype=torch.long,
        ).clamp(min=0, max=int(self.config.max_action_steps))
        encoded = [model.encode(batch) for model in self.models]
        states = [item[0] for item in encoded]
        options = [item[1] for item in encoded]
        keys = [model.pointer_key(value) for model, value in zip(self.models, options)]
        hidden = [torch.tanh(model.decoder_init(state)) for model, state in zip(self.models, states)]
        option_mask = batch["option_mask"].bool()
        batch_size, option_count = option_mask.shape
        chosen = torch.zeros_like(option_mask)
        active = max_count.gt(0)
        actions = torch.full(
            (batch_size, int(self.config.max_action_steps)),
            -1,
            dtype=torch.long,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long)
        row_ids = torch.arange(batch_size)
        for step in range(int(self.config.max_action_steps)):
            logits = [
                self._logits(
                    model,
                    option_values,
                    option_keys,
                    model_hidden,
                    option_mask,
                    chosen,
                    min_count,
                    active,
                    step,
                )
                for model, option_values, option_keys, model_hidden in zip(
                    self.models, options, keys, hidden
                )
            ]
            probabilities = (
                (1.0 - self.right_weight) * torch.softmax(logits[0], dim=1)
                + self.right_weight * torch.softmax(logits[1], dim=1)
            )
            choice = probabilities.argmax(dim=1)
            chosen_valid = active & choice.lt(option_count)
            safe_choice = choice.clamp(min=0, max=option_count - 1)
            actions[:, step] = torch.where(chosen_valid, safe_choice, -1)
            chosen[row_ids, safe_choice] |= chosen_valid
            lengths += chosen_valid.long()
            next_hidden = []
            for model, option_values, model_hidden in zip(self.models, options, hidden):
                selected = option_values.gather(
                    1,
                    safe_choice.view(-1, 1, 1).expand(-1, 1, option_values.size(-1)),
                ).squeeze(1)
                candidate = model.decoder(selected, model_hidden)
                next_hidden.append(torch.where(chosen_valid.unsqueeze(1), candidate, model_hidden))
            hidden = next_hidden
            active = chosen_valid & lengths.lt(max_count)
        decoded = canonical_actions(actions, lengths)
        return [
            require_legal_model_action(observation, action, policy_name=self.name)
            for (_key, observation), action in zip(requests, decoded)
        ]


def make_opponent(
    spec: OpponentSpec,
    target: FrozenPolicyCodecV1Backends,
    module_index: int,
) -> Any:
    if spec.kind == "policy_codec_v1":
        return FrozenPolicyCodecV1Opponent(spec, target)
    if spec.kind == "idonly_probability_ensemble":
        return FrozenProbabilityEnsembleOpponent(spec, module_index)
    return FrozenIDOnlyOpponent(spec, module_index)


@dataclass
class Branch:
    key: str
    game: SeededCppBattle
    steps: int = 0
    result: int = 0
    finish_reason: int = 0
    error: str = ""

    @property
    def done(self) -> bool:
        return bool(self.error or self.result)


@dataclass
class Pair:
    pair_id: int
    seed: int
    evaluated_player: int
    initial_digest: int
    cpu: Branch
    gpu: Branch
    diverged: bool = False
    first_divergence: dict[str, Any] | None = None
    comparable_decisions: int = 0
    raw_action_differences: int = 0
    semantic_action_differences: int = 0
    invariant_errors: list[str] = field(default_factory=list)


def refresh_branch(branch: Branch) -> None:
    if branch.done:
        return
    snapshot = branch.game.snapshot()
    branch.result = int(snapshot.meta.game_result)
    branch.finish_reason = int(snapshot.meta.finish_reason)


def close_pairs(pairs: list[Pair]) -> None:
    for pair in pairs:
        pair.cpu.game.close()
        pair.gpu.game.close()


def play_chunk(
    *,
    shim: SeededCppLib,
    target: FrozenPolicyCodecV1Backends,
    target_deck: list[int],
    opponent: Any,
    opponent_index: int,
    game_start: int,
    games: int,
    base_seed: int,
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    pairs: list[Pair] = []
    stats = Counter()
    try:
        for offset in range(games):
            game_index = game_start + offset
            pair_id = opponent_index * 1_000_000 + game_index
            setup_seed = base_seed + opponent_index * 1_000_003 + game_index * 1009
            continuation_seed = setup_seed + 25_214_903_917
            evaluated_player = game_index % 2
            decks = (
                (target_deck, opponent.deck)
                if evaluated_player == 0
                else (opponent.deck, target_deck)
            )
            with shim.start_battle(*decks, seed=setup_seed) as root:
                branches = [root.clone(), root.clone()]
                for branch in branches:
                    branch.reseed(continuation_seed)
                digests = [branch.digest() for branch in branches]
                if digests[0] != digests[1]:
                    raise RuntimeError(f"initial branch digest mismatch for pair {pair_id}: {digests}")
            pairs.append(
                Pair(
                    pair_id=pair_id,
                    seed=setup_seed,
                    evaluated_player=evaluated_player,
                    initial_digest=digests[0],
                    cpu=Branch(f"{pair_id}:cpu", branches[0]),
                    gpu=Branch(f"{pair_id}:gpu", branches[1]),
                )
            )

        for _tick in range(max_steps):
            for pair in pairs:
                refresh_branch(pair.cpu)
                refresh_branch(pair.gpu)
            if all(pair.cpu.done and pair.gpu.done for pair in pairs):
                break

            cpu_requests: list[tuple[Pair, dict[str, Any]]] = []
            gpu_requests: list[tuple[Pair, dict[str, Any]]] = []
            opponent_requests: list[tuple[Pair, Branch, dict[str, Any]]] = []
            snapshots: dict[str, Any] = {}
            for pair in pairs:
                for backend, branch in (("cpu", pair.cpu), ("gpu", pair.gpu)):
                    if branch.done:
                        continue
                    if branch.steps >= max_steps:
                        branch.error = "max_steps"
                        continue
                    snapshot = branch.game.snapshot()
                    snapshots[branch.key] = snapshot
                    observation = branch.game.observation()
                    if int(snapshot.meta.select_player) == pair.evaluated_player:
                        if backend == "cpu":
                            cpu_requests.append((pair, observation))
                        else:
                            gpu_requests.append((pair, observation))
                    else:
                        opponent_requests.append((pair, branch, observation))

            cpu_observations = [observation for _pair, observation in cpu_requests]
            gpu_observations = [observation for _pair, observation in gpu_requests]
            cpu_actions, cpu_equivalence = target.act_cpu(cpu_observations)
            gpu_actions = target.act_gpu(gpu_observations)
            opponent_actions = opponent.act_many(
                [(branch.key, observation) for _pair, branch, observation in opponent_requests]
            )
            actions: dict[str, list[int]] = {}
            equivalence_by_pair: dict[int, list[int]] = {}
            for (pair, _observation), action, equivalence in zip(
                cpu_requests, cpu_actions, cpu_equivalence
            ):
                actions[pair.cpu.key] = action
                equivalence_by_pair[pair.pair_id] = equivalence
                stats["cpu_target_decisions"] += 1
            for (pair, _observation), action in zip(gpu_requests, gpu_actions):
                actions[pair.gpu.key] = action
                stats["gpu_target_decisions"] += 1
            for (_pair, branch, _observation), action in zip(opponent_requests, opponent_actions):
                actions[branch.key] = action
                stats["opponent_decisions"] += 1

            for pair in pairs:
                if pair.diverged or pair.cpu.done or pair.gpu.done:
                    continue
                cpu_snapshot = snapshots.get(pair.cpu.key)
                gpu_snapshot = snapshots.get(pair.gpu.key)
                if cpu_snapshot is None or gpu_snapshot is None:
                    pair.invariant_errors.append("one_branch_finished_before_divergence")
                    pair.diverged = True
                    continue
                if pair.cpu.game.digest() != pair.gpu.game.digest():
                    pair.invariant_errors.append("pre_action_digest_mismatch")
                    pair.diverged = True
                    continue
                cpu_action = actions[pair.cpu.key]
                gpu_action = actions[pair.gpu.key]
                cpu_is_target = int(cpu_snapshot.meta.select_player) == pair.evaluated_player
                gpu_is_target = int(gpu_snapshot.meta.select_player) == pair.evaluated_player
                if cpu_is_target != gpu_is_target:
                    pair.invariant_errors.append("acting_role_mismatch")
                    pair.diverged = True
                    continue
                if cpu_is_target:
                    pair.comparable_decisions += 1
                    stats["comparable_target_decisions"] += 1
                    if cpu_action != gpu_action:
                        pair.raw_action_differences += 1
                        stats["raw_action_differences"] += 1
                        semantic_equal = semantic_actions_equal(
                            cpu_action,
                            gpu_action,
                            equivalence_by_pair[pair.pair_id],
                        )
                        if not semantic_equal:
                            pair.semantic_action_differences += 1
                            stats["semantic_action_differences"] += 1
                        pair.first_divergence = {
                            "pair_id": pair.pair_id,
                            "seed": pair.seed,
                            "step": pair.cpu.steps,
                            "evaluated_player": pair.evaluated_player,
                            "cpu_action": cpu_action,
                            "gpu_action": gpu_action,
                            "same_option_equiv": semantic_equal,
                            "option_equiv": equivalence_by_pair[pair.pair_id],
                        }
                        pair.diverged = True
                elif cpu_action != gpu_action:
                    pair.invariant_errors.append("opponent_action_mismatch_before_divergence")
                    pair.diverged = True

            for pair in pairs:
                for branch in (pair.cpu, pair.gpu):
                    if branch.done or branch.key not in actions:
                        continue
                    error = int(branch.game.select(actions[branch.key]))
                    branch.steps += 1
                    if error:
                        branch.error = f"select_error:{error}"
                if not pair.diverged and not pair.cpu.done and not pair.gpu.done:
                    if pair.cpu.game.digest() != pair.gpu.game.digest():
                        pair.invariant_errors.append("post_action_digest_mismatch")
                        pair.diverged = True
        else:
            for pair in pairs:
                for branch in (pair.cpu, pair.gpu):
                    refresh_branch(branch)
                    if not branch.done:
                        branch.error = "max_steps"

        game_rows: list[dict[str, Any]] = []
        divergences: list[dict[str, Any]] = []
        for pair in pairs:
            refresh_branch(pair.cpu)
            refresh_branch(pair.gpu)
            cpu_winner = terminal_winner(pair.cpu.result)
            gpu_winner = terminal_winner(pair.gpu.result)
            cpu_won = cpu_winner == pair.evaluated_player if cpu_winner is not None else None
            gpu_won = gpu_winner == pair.evaluated_player if gpu_winner is not None else None
            complete = not pair.cpu.error and not pair.gpu.error and not pair.invariant_errors
            row = {
                "pair_id": pair.pair_id,
                "seed": pair.seed,
                "evaluated_player": pair.evaluated_player,
                "initial_digest": pair.initial_digest,
                "comparable_decisions": pair.comparable_decisions,
                "raw_action_differences": pair.raw_action_differences,
                "semantic_action_differences": pair.semantic_action_differences,
                "trajectory_diverged": pair.diverged,
                "cpu_result": pair.cpu.result,
                "gpu_result": pair.gpu.result,
                "cpu_finish_reason": pair.cpu.finish_reason,
                "gpu_finish_reason": pair.gpu.finish_reason,
                "cpu_won": cpu_won,
                "gpu_won": gpu_won,
                "same_outcome": cpu_won == gpu_won if complete else None,
                "cpu_steps": pair.cpu.steps,
                "gpu_steps": pair.gpu.steps,
                "cpu_error": pair.cpu.error,
                "gpu_error": pair.gpu.error,
                "invariant_errors": ";".join(pair.invariant_errors),
                "complete": complete,
            }
            game_rows.append(row)
            if pair.first_divergence is not None:
                divergences.append(pair.first_divergence)
        return game_rows, divergences, dict(stats)
    finally:
        close_pairs(pairs)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["complete"]]
    decided = [row for row in complete if isinstance(row["cpu_won"], bool) and isinstance(row["gpu_won"], bool)]
    cpu_wins = sum(row["cpu_won"] is True for row in decided)
    gpu_wins = sum(row["gpu_won"] is True for row in decided)
    discordant = [row for row in decided if row["cpu_won"] != row["gpu_won"]]
    comparable = sum(int(row["comparable_decisions"]) for row in rows)
    raw_differences = sum(int(row["raw_action_differences"]) for row in rows)
    semantic_differences = sum(int(row["semantic_action_differences"]) for row in rows)
    return {
        "pairs": len(rows),
        "complete_pairs": len(complete),
        "decided_pairs": len(decided),
        "errors": len(rows) - len(complete),
        "invariant_errors": sum(bool(row["invariant_errors"]) for row in rows),
        "comparable_target_decisions": comparable,
        "raw_action_differences": raw_differences,
        "raw_action_difference_rate": raw_differences / max(1, comparable),
        "semantic_action_differences": semantic_differences,
        "semantic_action_difference_rate": semantic_differences / max(1, comparable),
        "trajectory_divergences": sum(bool(row["trajectory_diverged"]) for row in rows),
        "outcome_discordances": len(discordant),
        "outcome_discordance_rate": len(discordant) / max(1, len(decided)),
        "cpu_wins": cpu_wins,
        "gpu_wins": gpu_wins,
        "cpu_win_rate": cpu_wins / max(1, len(decided)),
        "gpu_win_rate": gpu_wins / max(1, len(decided)),
        "gpu_minus_cpu_win_rate": (gpu_wins - cpu_wins) / max(1, len(decided)),
        "cpu_only_wins": sum(row["cpu_won"] is True and row["gpu_won"] is False for row in discordant),
        "gpu_only_wins": sum(row["cpu_won"] is False and row["gpu_won"] is True for row in discordant),
        "average_cpu_steps": sum(int(row["cpu_steps"]) for row in complete) / max(1, len(complete)),
        "average_gpu_steps": sum(int(row["gpu_steps"]) for row in complete) / max(1, len(complete)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seeded live-battle A/B for frozen PolicyCodecV1 CPU and CUDA semantics."
    )
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--opponent-pool", type=Path, default=DEFAULT_OPPONENT_POOL)
    parser.add_argument(
        "--opponent-archetypes",
        default="",
        help="Optional comma-separated subset for a replacement run; the source pool must still define all six.",
    )
    parser.add_argument("--games-per-opponent", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=2026073001)
    parser.add_argument("--engine-threads", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-outcome-discordance-rate", type=float, default=0.005)
    parser.add_argument("--max-win-rate-delta", type=float, default=0.005)
    parser.add_argument("--max-semantic-action-difference-rate", type=float, default=0.0005)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(
        args.games_per_opponent,
        args.chunk_size,
        args.max_steps,
        args.engine_threads,
        args.cpu_threads,
    ) <= 0:
        raise ValueError("game, chunk, step, and thread counts must be positive")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    lib = args.lib.resolve()
    opponent_pool_path = args.opponent_pool.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU arm")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(device)

    opponent_pool_raw = json.loads(opponent_pool_path.read_text(encoding="utf-8"))
    all_selected = [OpponentSpec.from_dict(row) for row in opponent_pool_raw["opponents"]]
    expected_archetypes = {"lucario", "cynthia", "kangaskhan_crustle", "marnie", "alakazam", "dragapult"}
    if len(all_selected) != 6 or {spec.archetype for spec in all_selected} != expected_archetypes:
        raise RuntimeError("battle A/B requires the six requested strongest-BC archetypes")
    forbidden = {"rule_lucario", "rule_crustal", "alakazam_rule", "alakazam_search"}
    if any(spec.kind.startswith("rule") or spec.archetype in forbidden for spec in all_selected):
        raise RuntimeError("rule/search opponents are forbidden in the pure-BC pool")
    requested_archetypes = {
        value.strip()
        for value in args.opponent_archetypes.split(",")
        if value.strip()
    }
    if requested_archetypes - expected_archetypes:
        raise ValueError(f"unknown requested archetypes: {sorted(requested_archetypes - expected_archetypes)}")
    selected = [
        (index, spec)
        for index, spec in enumerate(all_selected)
        if not requested_archetypes or spec.archetype in requested_archetypes
    ]

    target = FrozenPolicyCodecV1Backends(checkpoint, device, cpu_threads=args.cpu_threads)
    target_deck = read_deck(checkpoint.parent / "deck.csv")
    shim = SeededCppLib(str(lib))
    shim.set_batch_threads(args.engine_threads)
    source_paths = {
        "model": REPO_ROOT / "tools" / "pure_policy_model_v1.py",
        "codec": REPO_ROOT / "tools" / "pure_policy_codec_v1.py",
        "adapter": CUDA_ENGINE_ROOT / "python" / "ptcg_cuda_engine" / "policy_adapters.py",
    }
    freeze_manifest = {
        "schema_version": 1,
        "status": "frozen_for_battle_ab",
        "policy": "PolicyCodecV1 EntityPointerPolicyV1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "deck_sha256": sha256_file(checkpoint.parent / "deck.csv"),
        "engine": str(lib),
        "engine_sha256": sha256_file(lib),
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "float32",
        "decode": "greedy_argmax_device_adapter",
        "entity_capacity": target.entity_capacity,
        "option_capacity": target.option_capacity,
        "max_select": 80,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "opponent_pool_manifest": str(opponent_pool_path),
        "opponent_pool_manifest_sha256": sha256_file(opponent_pool_path),
        "opponent_pool": [
            {
                "name": spec.name,
                "archetype": spec.archetype,
                "kind": spec.kind,
                "checkpoints": [str(path) for path in spec.checkpoints],
                "checkpoint_sha256": [sha256_file(path) for path in spec.checkpoints],
                "adapter": spec.adapter,
                "codec": spec.codec,
                "right_weight": spec.right_weight,
            }
            for _index, spec in selected
        ],
        "excluded_opponents": ["rule Lucario", "rule Crustal", "standalone rule/search Alakazam"],
        "known_engine_divergence": "KNOWN_DIVERGENCE_660_1207",
    }
    freeze_path = out / "gpu_policy_freeze_manifest.json"
    write_json(freeze_path, freeze_manifest)

    all_rows: list[dict[str, Any]] = []
    all_divergences: list[dict[str, Any]] = []
    matchup_reports: list[dict[str, Any]] = []
    started = time.time()
    for opponent_index, spec in selected:
        opponent_started = time.time()
        opponent = make_opponent(spec, target, opponent_index)
        matchup_rows: list[dict[str, Any]] = []
        matchup_divergences: list[dict[str, Any]] = []
        runtime_stats = Counter()
        for game_start in range(0, args.games_per_opponent, args.chunk_size):
            games = min(args.chunk_size, args.games_per_opponent - game_start)
            rows, divergences, stats = play_chunk(
                shim=shim,
                target=target,
                target_deck=target_deck,
                opponent=opponent,
                opponent_index=opponent_index,
                game_start=game_start,
                games=games,
                base_seed=args.seed,
                max_steps=args.max_steps,
            )
            for row in rows:
                row.update(opponent_pool_index=opponent_index, opponent=spec.name)
            for row in divergences:
                row.update(opponent_pool_index=opponent_index, opponent=spec.name)
            matchup_rows.extend(rows)
            matchup_divergences.extend(divergences)
            runtime_stats.update(stats)
            print(
                json.dumps(
                    {
                        "event": "battle_ab_progress",
                        "opponent": spec.name,
                        "pairs_done": len(matchup_rows),
                        "pairs_target": args.games_per_opponent,
                    }
                ),
                flush=True,
            )
        report = aggregate(matchup_rows)
        report.update(
            opponent_pool_index=opponent_index,
            opponent=spec.name,
            archetype=spec.archetype,
            runtime_counts=dict(runtime_stats),
            elapsed_sec=round(time.time() - opponent_started, 3),
        )
        matchup_reports.append(report)
        all_rows.extend(matchup_rows)
        all_divergences.extend(matchup_divergences)
        del opponent

    totals = aggregate(all_rows)
    pass_checks = {
        "all_pairs_complete": totals["complete_pairs"] == totals["pairs"],
        "no_invariant_errors": totals["invariant_errors"] == 0,
        "outcome_discordance_within_limit": (
            totals["outcome_discordance_rate"] <= args.max_outcome_discordance_rate
        ),
        "win_rate_delta_within_limit": (
            abs(totals["gpu_minus_cpu_win_rate"]) <= args.max_win_rate_delta
        ),
        "semantic_action_difference_within_limit": (
            totals["semantic_action_difference_rate"]
            <= args.max_semantic_action_difference_rate
        ),
    }
    report = {
        "status": "pass" if all(pass_checks.values()) else "fail",
        "scope": "CPU official engine; CPU/GPU frozen PolicyCodecV1 policy A/B; no training",
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "games_per_opponent": args.games_per_opponent,
        "opponent_count": len(selected),
        "seed": args.seed,
        "engine_threads": args.engine_threads,
        "cpu_threads": args.cpu_threads,
        "thresholds": {
            "max_outcome_discordance_rate": args.max_outcome_discordance_rate,
            "max_win_rate_delta": args.max_win_rate_delta,
            "max_semantic_action_difference_rate": args.max_semantic_action_difference_rate,
        },
        "checks": pass_checks,
        "totals": totals,
        "matchups": matchup_reports,
        "first_divergences": all_divergences[:100],
        "elapsed_sec": round(time.time() - started, 3),
        "known_engine_divergence": "KNOWN_DIVERGENCE_660_1207 retained and not redefined",
    }
    write_csv(out / "games.csv", all_rows)
    write_json(out / "divergences.json", {"divergences": all_divergences})
    write_json(out / "report.json", report)
    print(json.dumps({"event": "battle_ab_done", **report}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
